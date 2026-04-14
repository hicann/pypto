# 使用 Shmem API 实现 Dispatch Kernel

## 1. Dispatch 功能描述

Dispatch 算子是分布式 MoE（Mixture of Experts）分布式训练中的关键算子，用于在专家并行（Expert Parallel, EP）场景下实现 token 的高效分发。其主要功能包括：

- **Token分发**: 根据每个 token 选择的 topk 个专家 ID，将 token 数据分发到对应的专家节点
- **辅助信息生成**: 为后续 combine 阶段生成辅助信息，包含发送 rank、token_id和k_offset
- **计数统计**: 统计每个专家接收的 token 数量，用于后续的内存管理和数据对齐
- **跨rank通信**: 基于共享内存（ShmemAPI）实现高效的跨 rank 数据交换

Dispatch 的核心任务是实现 token 的路由发送与按序整合

## 2. Dispatch kernel原型

```python
def moe_distributed_dispatch_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor], tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]]:
    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 3})
    def kernel(
        x: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        expert_ids: pypto.Tensor([batch_size, topk], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expand_x: pypto.Tensor([expand_x_row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for: pypto.Tensor([expand_x_row, info_size], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_token_nums: pypto.Tensor([expert_num_per_rank], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
    ):
        # kernel实现
        pass
        return kernel
```

## 3. 参数说明

### 3.1 标量参数

| 参数名 | 类型 | 说明 |
|--------|------|----------|
| batch_size | int | 批次大小，支持 8 |
| hidden_size | int | 隐藏层维度，固定为 5120 |
| moe_expert_num | int | 专家总数，固定为 160 |
| topk | int | 每个 token 选择的专家数量，固定为 8 |
| data_type | DT_BF16 | 数据类型，固定为 DT_BF16 |
| ep_world_size | int | 专家并行的 rank 数量，支持 4 或 8 |
| group_name | str | 通信组名称，长度范围[1, 128) |

**派生标量参数**:
- `expand_x_row = min(topk * batch_size * ep_world_size, batch_size * moe_expert_num)`: 预留的接收 tensor 的行数
- `expert_num_per_rank = moe_expert_num // ep_world_size`: 每个 rank 负责的专家数量
- `info_size = 64`: 三元组信息的长度
- `cum_sum_row_size = align_up(moe_expert_num, 256)`: cumsum tensor的行数（256字节对齐）
- `count_size = 8`: 计数 tensor 的列数

### 3.2 输入Tensor

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| x | [batch_size, hidden_size] | DT_BF16 | 输入 token 数据，每行代表一个 token |
| expert_ids | [batch_size, topk] | DT_INT32 | 每个 token 选择的 topk 个专家 ID |

### 3.3 输出Tensor

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| expand_x | [expand_x_row, hidden_size] | DT_BF16 | 扩展后的 token 数据，按专家分组存储 |
| assist_info_for_combine | [expand_x_row, info_size] | DT_INT32 | 辅助信息，用于 combine 阶段恢复 token 顺序 |
| expert_token_nums | [expert_num_per_rank] | DT_INT32 | 每个专家接收的 token 数量 |
| recv_counts | [1] | DT_INT32 | 本 rank 接收的总token数量 |

## 4. 计算逻辑伪代码

```
# 获取当前 rank
this_rank = get_current_rank()

# 创建共享内存区域
# 用于接收 token 的共享区域
shmem_data = create_shmem_tensor([moe_expert_num * batch_size, hidden_size], dtype)
# 用于接收三元组的共享区域
shmem_info = create_shmem_tensor([moe_expert_num * batch_size, info_size], DT_INT32)
# 用于接收计数的共享区域
shmem_count = create_shmem_tensor([cum_sum_row_size, count_size], DT_INT32)

# 阶段1: 计算发送偏移
expert_ids_vec = reshape(expert_ids, [1, batch_size * topk])
offset_table = zeros([batch_size, topk], DT_INT32)
for index in range(batch_size * topk):
    row_index, col_index = divmod(index, topk)
    remote_expert_id = expert_ids[row_index, col_index]
    token_offset = calc_occurrences(expert_ids_vec, remote_expert_id, index)
    offset_table[row_index, col_index] = token_offset

# 阶段2: 发送 token 与辅助信息
for index in range(batch_size * topk):
    row_index, col_index = divmod(index, topk)
    
    # 构造辅助信息
    moe_info = zeros([1, info_size], DT_INT32)
    moe_info[0, info_size-3] = this_rank
    moe_info[0, info_size-2] = row_index
    moe_info[0, info_size-1] = col_index
    
    # 计算目标位置
    remote_expert_id = expert_ids[row_index, col_index]
    remote_rank_id = remote_expert_id // expert_num_per_rank
    remote_expert_offset = remote_expert_id % expert_num_per_rank
    token_offset = offset_table[row_index, col_index]
    dst_offset = (remote_expert_offset * ep_world_size + this_rank) * batch_size + token_offset
    
    # 发送数据
    shmem_put(x[row_index:row_index+1, :], dst_offset, shmem_data, remote_rank_id)
    shmem_put(moe_info, dst_offset, shmem_info, remote_rank_id)
    shmem_signal(shmem_data, 1, ADD)

# 阶段3: 发送计数信息
for expert_id in range(moe_expert_num):
    expert_offset = calc_occurrences(expert_ids_vec, expert_id, batch_size * topk)
    remote_rank_id = expert_id // expert_num_per_rank
    remote_expert_offset = expert_id % expert_num_per_rank
    total_count = expert_offset[-1]
    dst_offset = remote_expert_offset * ep_world_size + this_rank + 1
    
    shmem_put(total_count, dst_offset, shmem_count, remote_rank_id)
    shmem_signal(shmem_count, 1, ADD)

# 阶段4: 等待并计算 cumsum
wait_until(shmem_data.signal == batch_size * topk * ep_world_size)
wait_until(shmem_count.signal == moe_expert_num)

local_expert_recv_count = shmem_get(shmem_count, this_rank)
cum_sum_input = shmem_get(shmem_count, this_rank)
cum_sum_result = cumsum(cum_sum_input, axis=0)

recv_counts[0] = cum_sum_result[expert_num_per_rank * ep_world_size, 0]

for expert_id in range(expert_num_per_rank):
    start_row = expert_id * ep_world_size + 1
    end_row = start_row + ep_world_size
    expert_valid_cnt = cum_sum_input[start_row:end_row, :]
    expert_valid_cum_sum = cumsum(expert_valid_cnt, axis=0)
    expert_token_nums[expert_id] = expert_valid_cum_sum[-1, 0]

# 阶段5: 接收数据
for index in range(moe_expert_num):
    cur_count = local_expert_recv_count[index + 1, 0]
    offset = cum_sum_result[index, 0]
    
    local_data = shmem_get(shmem_data, this_rank, [index * batch_size, 0], valid_shape=[cur_count, hidden_size])
    local_info = shmem_get(shmem_info, this_rank, [index * batch_size, 0], valid_shape=[cur_count, info_size])
    
    expand_x[offset:offset+cur_count, :] = local_data
    assist_info_for_combine[offset:offset+cur_count, :info_size] = local_info
```

## 5. 计算流程讲解

### 5.1 偏移计算准备

**目的**: 计算每个 token 在目标专家共享内存区域中的写入偏移量，避免多个 token 写入同一位置。

**实现步骤**:
1. 将`expert_ids`从 [batch_size, topk] 重塑为 [1, batch_size * topk] 的向量形式
2. 创建`offset_table`与 expert_ids 同 shape, 描述 expert_ids 每个专家为本卡发送的第几个
3. 遍历所有 token（batch_size * topk个），对每个 token：
   - 获取该 token 选择的专家 ID：`remote_expert_id = expert_ids[row_index, col_index]`
   - 调用`dispatch_calc_occurrences`函数计算该位置下, 该 token 对应的目标专家为发送给该专家的第几个
   - 将计算结果存储到`offset_table[row_index, col_index]`

**dispatch_calc_occurrences函数原理**:
- 通过比较 expert_ids 向量中与目标 expert_id 相等的元素数量
- 使用 cumsum 累计相等数量
- 返回当前 token 在该专家接收序列中的偏移量

**示例**:
假设batch_size=2, topk=2, expert_ids=[[0,1],[0,1]]，则：
- tokenId = 0、topk = 0 的发送专家 expert_id = 0，该 token 是第 0 个发送给 expert0 的 token，offset = 0
- tokenId = 0、topk = 1 的发送专家 expert_id = 1，该 token 是第 0 个发送给 expert1 的 token，offset = 0
- tokenId = 1、topk = 0 的发送专家 expert_id = 0，该 token 是第 1 个发送给 expert0 的 token，offset = 1
- tokenId = 1、topk = 1 的发送专家 expert_id = 1，该 token 是第 1 个发送给 expert1 的 token，offset = 1

### 5.2 数据发送阶段

**目的**: 将 token 数据和辅助信息发送到目标rank的共享内存区域。

**实现步骤**:
1. 遍历所有 token（batch_size * topk个）
2. 对每个 token：
   - **构造辅助信息**:
     - `moe_info[0, info_size-3] = this_rank`: 记录发送 rank
     - `moe_info[0, info_size-2] = row_index`: 记录 token_id
     - `moe_info[0, info_size-1] = col_index`: 记录 k_offset
   
   - **计算目标位置**:
     - `remote_rank_id = remote_expert_id // expert_num_per_rank`: 目标 rank
     - `remote_expert_offset = remote_expert_id % expert_num_per_rank`: 目标 rank 内的 expert 偏移
     - `token_offset = offset_table[row_index, col_index]`: 该 expert 内的 token 偏移
     - `dst_offset = (remote_expert_offset * ep_world_size + this_rank) * batch_size + token_offset`: 最终共享内存偏移
   
   - **发送数据**:
     - 使用`shmem_put`发送token数据到`shmem_data`
     - 使用`shmem_put`发送辅助信息到`shmem_info`
     - 使用`shmem_signal`发送信号，信号值累加（ADD 操作），通知接收端本卡 expert_ids 该位置的发送任务已执行完毕

**shmem_data与shmem_signal布局**:

```
[expert_0 * batch_size, hidden_size]  <- expert 0 从各个 rank 接收的 token
[expert_1 * batch_size, hidden_size]  <- expert 1 从各个 rank 接收的 token
...
[expert_N * batch_size, hidden_size]  <- expert N 从各个 rank 接收的 token
```

每个 expert 的 batch_size 个槽位按 rank 交错排列，便于后续接收。

### 5.3 有效数据数目发送阶段

**目的**: 统计每个 expert 从本 rank 接收的 token 数量，并发送到目标 expert 的 count 共享区。

**实现步骤**:
1. 遍历所有 expert_id（0到 moe_expert_num -1）
2. 对每个 expert_id：
   - 调用`dispatch_calc_occurrences`计算该 expert 从本 rank 接收的 token 总数
   - 计算目标 rank 和 expert 偏移
   - 使用`shmem_put`发送计数信息到`shmem_count`
   - 使用`shmem_signal`发送信号

**shmem_count布局**:

```
[0, count_size]: 额外预留位置，用于计算 CumSum 操作时存储 0 号专家的偏移
[1, count_size]: expert 0从 rank_id = 0接收的 token 数量
[2, count_size]: expert 0从 rank_id = 1接收的 token 数量
...
[ep_world_size, count_size]: expert 0 从 rank_id = ep_world_size - 1 接收的 token 数量
...
[expert_a + ep_world_size * expert_a + 1, count_size]: expert a 从 rank_id = 0 接收的 token 数量
...
[expert_a + ep_world_size * expert_a + ep_world_size, count_size]: expert a 从 rank_id = ep_world_size - 1 接收的 token 数量
...
```

每个 expert 占用 ep_world_size 行，记录从各个 rank 接收的数量，有序用于做 CumSum 操作计算每一条接收 token 在结果的行偏移

### 5.4 CumSum 阶段

**目的**: 等待所有数据准备好，计算每个 expert 在输出 tensor 中的行偏移量。

**实现步骤**:
1. **等待同步**:
   - 使用`shmem_wait_until`等待所有rank的数据信号：`shmem_data.signal == batch_size * topk * ep_world_size`，表示全局所有卡 expert_ids 的发送任务全部执行完
   - 使用`shmem_wait_until`等待所有rank的计数信号：`shmem_count.signal == moe_expert_num`，表示全局所有卡发送所有专家 count 的任务全部执行完
   - 等待完成后清理信号（`clear_signal=True`）

2. **获取计数信息**:
   - 使用`shmem_get`从`shmem_count`获取本rank的计数信息
   - 存储到`local_expert_recv_count`和`cum_sum_input`

3. **计算cumsum**:
   - 对`cum_sum_input`按行计算 cumsum，得到`cum_sum_result`
   - `cum_sum_result[i, 0]`有 expert_num_per_rank * ep_world_size 行，表示某个专家接收的每个卡的首条 token 在expand_x 的行偏移

4. **计算接收总数**:
   - `recv_counts[0] = cum_sum_result[expert_num_per_rank * ep_world_size, 0]`: 本 rank 接收的总 token 数量

5. **计算每个expert的接收数量**:
   - 遍历本rank负责的expert（0 到 expert_num_per_rank-1）
   - 对每个expert，从`cum_sum_input`中提取所有 rank 发送给该expert的数量
   - 计算cumsum得到该expert的总接收数量
   - 存储到`expert_token_nums[expert_id]`

**示例**:
假设 ep_world_size=4，expert_num_per_rank=40，则：
- expert 0 的计数信息在`cum_sum_input[1:5, 0]`（从 rank 0,1,2,3接收的 token 数量）
- expert 0 的总接收数量为`sum(cum_sum_input[1:5, 0])`
- expert 0 的 0-3 卡的接收数据在 expand_x 中的行偏移为`cum_sum_result[1:5, 0]`

### 5.5 数据接收阶段

**目的**: 从共享内存中接收 token 数据和三元组信息，写入输出 expand_x。

**实现步骤**:
1. 遍历本卡所有专家 expert_id（0到expert_num_per_rank-1）的每一个 rank 接收区域
2. 对每个区域：
   - **获取接收数量和偏移**:
     - `cur_count = local_expert_recv_count[index + 1, 0]`: 本卡某 expert 从该 rank 接收的 token 数量
     - `offset = cum_sum_result[index, 0]`: 本卡某 expert 在输出 expand_x 中的行偏移量
   
   - **接收token数据**:
     - 使用`shmem_get`从`shmem_data`接收数据
     - 源偏移：`[index * batch_size, 0]`
     - 使用`valid_shape=[cur_count, hidden_size]`只接收该区域内的有效数据
     - 写入输出：`expand_x[offset:offset+cur_count, :]`
   
   - **接收辅助信息**:
     - 使用`shmem_get`从`shmem_info`接收数据
     - 源偏移：`[index * batch_size, 0]`
     - 使用`valid_shape=[cur_count, info_size]`只接收该区域内的有效数据
     - 写入输出：`assist_info_for_combine[offset:offset+cur_count, :info_size]`

**输出tensor布局**:

```
expand_x布局:
[0:expert_0_count, :]          <- expert 0 的 token 数据
[expert_0_count:expert_1_count, :]  <- expert 1 的 token 数据
...
[sum(prev_counts):, :]        <- expert N 的 token 数据

assist_info_for_combine布局:
[0:expert_0_count, :]          <- expert 0 的辅助信息
[expert_0_count:expert_1_count, :]  <- expert 1 的辅助信息
...
[sum(prev_counts):, :]        <- expert N 的辅助信息
```

每个 expert 的 token 数据按 expert_id 再按 rank_id 顺序连续存储
