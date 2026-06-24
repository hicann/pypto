# 使用Shmem API实现Dispatch Kernel

## Dispatch功能描述

Dispatch算子是分布式MoE（Mixture of Experts）分布式训练中的关键算子，用于在专家并行（Expert Parallel, EP）场景下实现token的高效分发。其主要功能包括：

- **Token分发**:根据每个token选择的topk个专家ID，将token数据分发到对应的专家节点
- **辅助信息生成**:为后续combine阶段生成辅助信息，包含发送rank、token_id和k_offset
- **计数统计**:统计每个专家接收的token数量，用于后续的内存管理和数据对齐
- **跨rank通信**:基于共享内存（ShmemAPI）实现高效的跨rank数据交换

Dispatch的核心任务是实现token的路由发送与按序整合

## Dispatch kernel原型

```python
def moe_distributed_dispatch_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]]:
    @pypto.frontend.jit()
    def kernel(
        x: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        expert_ids: pypto.Tensor([batch_size, topk], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        x_active_mask: pypto.Tensor([batch_size], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expand_x: pypto.Tensor([expand_x_row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for_combine: pypto.Tensor([expand_x_row, info_out_size], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_token_nums: pypto.Tensor([expert_num_per_rank], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
    ):
      # kernel实现
      pass
    return kernel
```

## 参数说明

### 标量参数

| 参数名 | 类型 | 说明 |
|--------|------|----------|
| batch_size | int | 批次大小，支持8 |
| hidden_size | int | 隐藏层维度，固定为5120 |
| moe_expert_num | int | 专家总数，固定为160 |
| topk | int | 每个token选择的专家数量，固定为8 |
| data_type | DT_BF16 | 数据类型，固定为DT_BF16 |
| ep_world_size | int | 专家并行的rank数量，支持4或8 |
| group_name | str | 通信组名称，长度范围[1, 128) |

**派生标量参数**:

- `expand_x_row = min(topk * batch_size * ep_world_size, batch_size * moe_expert_num)`:预留的接收tensor的行数
- `expert_num_per_rank = moe_expert_num // ep_world_size`:每个rank负责的专家数量
- `info_size = 4`:发送到共享内存中的辅助信息长度（rank_id, token_id, k_offset, padding）
- `info_out_size = 3`:最终输出辅助信息的长度（rank_id, token_id, k_offset）
- `cum_sum_row_size = align_up(moe_expert_num, 256)`: cumsum tensor的行数（256字节对齐）
- `count_size = 8`:计数tensor的列数
- `total_send_tasks = batch_size * topk`:总发送任务数

### 输入Tensor

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| x | [batch_size, hidden_size] | DT_BF16 | 输入token数据，每行代表一个token |
| expert_ids | [batch_size, topk] | DT_INT32 | 每个token选择的topk个专家ID |
| x_active_mask | [batch_size] | DT_INT32 | Token有效性掩码，前部连续为1（有效），后部为0（无效） |

### 输出Tensor

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| expand_x | [expand_x_row, hidden_size] | DT_BF16 | 扩展后的token数据，按专家分组存储 |
| assist_info_for_combine | [expand_x_row, info_out_size] | DT_INT32 | 辅助信息，用于combine阶段恢复token顺序 |
| expert_token_nums | [expert_num_per_rank] | DT_INT32 | 每个专家接收的token数量 |
| recv_counts | [1] | DT_INT32 | 本rank接收的总token数量 |

## 计算逻辑伪代码

```python
# 获取当前rank
this_rank = get_current_rank()

# 创建共享内存区域
# 用于接收token的共享区域
shmem_data = create_shmem_tensor([moe_expert_num * batch_size, hidden_size], dtype)
# 用于接收辅助信息的共享区域
shmem_info = create_shmem_tensor([moe_expert_num * batch_size, info_size], DT_INT32)
# 用于接收计数的共享区域
shmem_count = create_shmem_tensor([cum_sum_row_size, count_size], DT_INT32)
# 同步信号
shmem_barrier_signal = create_shmem_signal()

# 清空count共享区
shmem_clear_data(shmem_count)

# 阶段1:计算发送偏移（one_hot + cumsum方式，无需感知mask）
expert_ids_flat = reshape(expert_ids, [batch_size * topk])

# one_hot编码+ cumsum计算每个token在目标专家中的偏移
one_hot_table = one_hot(expert_ids_flat, moe_expert_num)
one_hot_table_int32 = cast(one_hot_table, INT32)
cumsum_table = cumsum(one_hot_table_int32, dim=0)
cumsum_table_int32 = cast(cumsum_table, INT32)

# 阶段2:发送token与辅助信息（含计数，仅活跃token参与）
for index in range(batch_size * topk):
    if x_active_mask[index // topk] == 1:
        token_id = index // topk
        k_offset = index % topk
        
        # 构造辅助信息
        moe_info = zeros([1, info_size], DT_INT32)
        moe_info[0, 0] = this_rank
        moe_info[0, 1] = token_id
        moe_info[0, 2] = k_offset
        
        # 计算目标位置
        remote_expert_id = expert_ids[token_id, k_offset]
        remote_rank_id = remote_expert_id // expert_num_per_rank
        remote_expert_offset = remote_expert_id % expert_num_per_rank
        
        # token_offset直接由cumsum_table给出，无需感知mask
        token_offset = cumsum_table_int32[index - 1, remote_expert_id] if index > 0 else 0
        dst_offset = (remote_expert_offset * ep_world_size + this_rank) * batch_size + token_offset
        
        # 发送token数据
        shmem_put(x[token_id:token_id+1, :], dst_offset, shmem_data, remote_rank_id)
        
        # 发送辅助信息
        shmem_put(moe_info, dst_offset, shmem_info, remote_rank_id)
        
        # 原子ADD计数（替代阶段3的单独发送）
        count = full([1, 1], 1, INT32)
        shmem_put(count, [remote_expert_offset * ep_world_size + this_rank + 1, 0],
                  shmem_count, remote_rank_id, put_op=ATOMIC_ADD)
    
    # 发送信号（广播，统一通知接收端）
    shmem_signal(shmem_data, target_pe=-1, sig_op=ADD)

# 阶段3:等待并计算cumsum
wait_until(shmem_data.signal == batch_size * topk * ep_world_size)
clear_signal(shmem_data)

# 获取本rank的计数信息
cum_sum_input = shmem_get(shmem_count, this_rank)
cum_sum_result = cumsum(cum_sum_input, axis=0)
cum_sum_result_int32 = cast(cum_sum_result, INT32)

recv_counts[0] = cum_sum_result_int32[expert_num_per_rank * ep_world_size, 0]

# 阶段4:计算每个expert的接收数量
for expert_id in range(expert_num_per_rank):
    start_row = expert_id * ep_world_size + 1
    end_row = start_row + ep_world_size
    expert_valid_cnt = cum_sum_input[start_row:end_row, :]
    expert_valid_cum_sum = cumsum(expert_valid_cnt, axis=0)
    expert_valid_cum_sum_int32 = cast(expert_valid_cum_sum, INT32)
    expert_token_nums[expert_id] = expert_valid_cum_sum_int32[ep_world_size - 1, 0]
    
    # 阶段5:接收数据（per expert + per rank）
    for rank_id in range(ep_world_size):
        index = expert_id * ep_world_size + rank_id
        cur_count = cum_sum_input[index + 1, 0]
        offset = cum_sum_result_int32[index, 0]
        
        local_data = shmem_get(shmem_data, this_rank, [batch_size, hidden_size],
                               [index * batch_size, 0], valid_shape=[cur_count, hidden_size])
        local_info = shmem_get(shmem_info, this_rank, [batch_size, info_out_size],
                               [index * batch_size, 0], valid_shape=[cur_count, info_out_size])
        
        expand_x[offset:offset+cur_count, :] = local_data
        assist_info_for_combine[offset:offset+cur_count, :] = local_info
```

## 计算流程讲解

### 偏移计算准备

**目的**:计算每个token在目标专家共享内存区域中的写入偏移量，避免多个token写入同一位置。当前实现采用`one_hot + cumsum`方式，天然兼容mask场景，无需感知mask。

**实现步骤**:

1. 将`expert_ids`从 [batch_size, topk] 重塑为 [batch_size * topk] 的一维向量
2. 对扁平后的expert_ids做`one_hot`编码，得到shape [total_send_tasks, moe_expert_num] 的one-hot矩阵
3. 沿第0维做`cumsum`，得到cumsum_table，其中`cumsum_table[i, e]`表示前i+1个token中发送给专家e的总次数
4. 对任意active token（index位置），其发送给目标expert_id的偏移量为`cumsum_table[index - 1, expert_id]`

**one_hot + cumsum的优势**:

- 通过one_hot将expert_id映射为稀疏向量，cumsum自然统计每个expert的累计发送次数
- 当引入x_active_mask时，仅活跃token的one_hot编码为非零值，非活跃token编码为全零向量，cumsum会自动跳过无效token，无需显式处理mask
- 偏移计算与mask解耦：token_offset = cumsum_table[index - 1, remote_expert_id]，公式本身不涉及mask

**示例（含mask）**:
假设batch_size=8, topk=8，expert_ids矩阵如下（每个token选择不同的expert组合），x_active_mask=[1,1,1,1,1,1,0,0]：

```.txt
token0: expert_ids=[0,1,2,3,0,1,2,3]  # 选expert0-3各2次
token1: expert_ids=[0,1,2,3,0,1,2,3]  # 选expert0-3各2次
token2: expert_ids=[0,1,2,3,0,1,2,3]  # 选expert0-3各2次
token3: expert_ids=[0,1,2,3,0,1,2,3]  # 选expert0-3各2次
token4: expert_ids=[0,1,2,3,0,1,2,3]  # 选expert0-3各2次
token5: expert_ids=[0,1,2,3,0,1,2,3]  # 选expert0-3各2次
token6: expert_ids=[0,1,2,3,0,1,2,3]  # 选expert0-3各2次（非活跃）
token7: expert_ids=[0,1,2,3,0,1,2,3]  # 选expert0-3各2次（非活跃）
```

- token_id=0-5活跃，token_id=6-7非活跃
- expert_ids_flat展开后长度为64，前48个对应活跃token（0-47），后16个对应非活跃token（48-63）

**cumsum_table构建过程**（假设只有4个expert）：

通过one_hot编码expert_ids_flat，并按列累加，得到cumsum_table（64行×4列）。cumsum_table[i,j]表示"从index=0到index=i，expert j被请求的累计次数"。

**部分cumsum_table值示例**：

| index | 对应的token-k | expert_id | cumsum_table各列值（expert0,1,2,3） |
|-------|--------------|-----------|-----------------------------------|
| 0 | token0-k0 | 0 | [**1**, 0, 0, 0] |
| 1 | token0-k1 | 1 | [1, **1**, 0, 0] |
| 2 | token0-k2 | 2 | [1, 1, **1**, 0] |
| 3 | token0-k3 | 3 | [1, 1, 1, **1**] |
| 4 | token0-k4 | 0 | [**2**, 1, 1, 1] |
| 7 | token0-k7 | 3 | [2, 2, 2, **2**] |
| 8 | token1-k0 | 0 | [**3**, 2, 2, 2] |
| 15 | token1-k7 | 3 | [4, 4, 4, **4**] |
| 47 | token5-k7 | 3 | [12, 12, 12, **12**] |
| 48 | token6-k0 | 0 | [**12**, 12, 12, 12]（非活跃，不增加） |
| 63 | token7-k7 | 3 | [12, 12, 12, **12**]（非活跃，不增加） |

**token_offset计算示例**：

| index | token | k | expert_id | token_offset计算 | 结果 |
|-------|-------|---|-----------|-----------------|------|
| 0 | token0 | 0 | 0 | 第一个位置，直接赋值 | **token_offset=0** |
| 1 | token0 | 1 | 1 | cumsum_table[0,1] = 0 | **token_offset=0** |
| 4 | token0 | 4 | 0 | cumsum_table[3,0] = 1 | **token_offset=1** |
| 8 | token1 | 0 | 0 | cumsum_table[7,0] = 2 | **token_offset=2** |
| 48 | token6 | 0 | 0 | cumsum_table[47,0] = 12 | **token_offset=12**（非活跃，不发送） |

> [!NOTE]说明
> 活跃token（token0-5）：6个token × 每token选每个expert 2次 = 每个expert被请求12次
> 非活跃token（token6-7）：虽然expert_ids存在且计算了token_offset，但发送阶段会检查mask并跳过，不实际发送数据

### 数据发送阶段（含mask适配与计数）

**目的**:将活跃token数据和辅助信息发送到目标rank的共享内存区域，并通过原子ADD操作为每个expert计数。

**实现步骤**:

1. 遍历所有token（batch_size * topk个）
2. 对每个token，首先检查`x_active_mask[token_id] == 1`，仅活跃token参与发送
3. 对每个活跃token：
   - **构造辅助信息**:
     - `moe_info[0, 0] = this_rank`:记录发送rank
     - `moe_info[0, 1] = token_id`:记录token_id
     - `moe_info[0, 2] = k_offset`:记录k_offset
   
   - **计算目标位置**:
     - `remote_rank_id = remote_expert_id // expert_num_per_rank`:目标rank
     - `remote_expert_offset = remote_expert_id % expert_num_per_rank`:目标rank内的expert偏移
     - `token_offset = cumsum_table_int32[index - 1, remote_expert_id] (index>0)或0`:该expert内的token偏移，因one_hot+cumsum已自动跳过无效token，无需感知mask
     - `dst_offset = (remote_expert_offset * ep_world_size + this_rank) * batch_size + token_offset`:最终共享内存偏移
   
   - **发送数据**:
     - 使用`shmem_put`发送token数据到`shmem_data`
     - 使用`shmem_put`发送辅助信息到`shmem_info`
     - 使用`shmem_put`(put_op=ATOMIC_ADD)发送计数1到`shmem_count`的目标位置：
       - `dst_count_offset = remote_expert_offset * ep_world_size + this_rank + 1`
       - 该操作原子累加，替代了原先独立的阶段3
4. 每次迭代都调用`shmem_signal`发送信号，`target_pe=-1`（广播模式），信号值累加（ADD操作），通知所有接收端发送任务已执行完毕

**mask约束说明**:

- `x_active_mask`要求前部连续为1（活跃），后部为0（非活跃）
- 该约束确保了数据发送的有序性：发送循环按index递增遍历，活跃token连续排列，非活跃token自动跳过

**shmem_data布局**:

```.txt
[expert_0 * batch_size, hidden_size]  <- expert 0从各个rank接收的token
[expert_1 * batch_size, hidden_size]  <- expert 1从各个rank接收的token
...
[expert_N * batch_size, hidden_size]  <- expert N从各个rank接收的token
```

每个expert的batch_size个槽位按rank交错排列，便于后续接收。

**计数机制说明**:

- 在发送token数据时，同时以原子ADD方式向`shmem_count`写入计数值1
- 目标位置`[remote_expert_offset * ep_world_size + this_rank + 1, 0]`标识了"本rank发送给某个expert的token数量"
- 多个token发送到同一个(expert, rank)时，原子ADD保证计数正确累加
- 该方式将原先独立的"计数发送阶段"融合到数据发送循环中，无需再次遍历所有expert

### 基于shmem_put + AtomicType.ADD的计数机制

**目的**:在发送token数据的同时，通过`shmem_put` + `AtomicType.ADD`原子累加操作，为每个expert统计从本rank接收的token数量。

**实现方式**:
计数不再作为独立阶段，而是融合在数据发送循环（5.2）中：

1. 每个活跃token发送数据时，通过`shmem_put(put_op=AtomicType.ADD)`向`shmem_count`写入常数值1
2. 目标位置为`[remote_expert_offset * ep_world_size + this_rank + 1, 0]`
3. `shmem_put`的`put_op=AtomicType.ADD`参数使写入操作为原子累加：远端`shmem_count`对应位置的值每次加1
4. 多个token发送到同一个(expert, rank)时，原子ADD保证最终值为该方向的token总数，无需任何锁或额外同步

**关键设计差异**（与旧方案对比）:

- 旧方案：独立阶段遍历所有expert，调用`calc_occurrences`汇总count，再通过`shmem_put` + `shmem_signal`发送
- 新方案：不依赖`shmem_signal`通知计数完成；每个token在发送数据时原子累加1，发送循环结束后`shmem_count`即为完整计数结果

**shmem_count布局**:

```.txt
[0, count_size]:额外预留位置，用于计算CumSum操作时存储0号专家的偏移
[1, count_size]: expert 0从rank_id = 0接收的token数量
[2, count_size]: expert 0从rank_id = 1接收的token数量
...
[ep_world_size, count_size]: expert 0从rank_id = ep_world_size - 1接收的token数量
...
[expert_a * ep_world_size + 1, count_size]: expert a从rank_id = 0接收的token数量
...
[expert_a * ep_world_size + ep_world_size, count_size]: expert a从rank_id = ep_world_size - 1接收的token数量
...
```

每个expert占用ep_world_size行，记录从各个rank接收的数量，有序用于做CumSum操作计算每一条接收token在结果的行偏移。

### CumSum阶段

**目的**:等待所有数据准备好，计算每个expert在输出tensor中的行偏移量。

**实现步骤**:

1. **等待同步**:
   - 使用`shmem_wait_until`等待数据信号：`shmem_data.signal == batch_size * topk * ep_world_size`，表示全局所有卡的发送任务全部执行完
   - 等待完成后清理信号（`clear_signal=True`）
   - 注意：shmem_count的累加已在发送循环中通过`shmem_put(put_op=AtomicType.ADD)`完成，无需单独的计数信号或额外等待

2. **获取计数信息**:
   - 使用`shmem_get`从`shmem_count`获取本rank的计数信息
   - 存储到`cum_sum_input`，shape为 [cum_sum_row_size, count_size]

3. **计算cumsum**:
   - 对`cum_sum_input`按行计算cumsum，得到`cum_sum_result`
   - `cum_sum_result[i, 0]`有expert_num_per_rank * ep_world_size行，表示某个expert接收的每个rank的首条token在expand_x的行偏移

4. **计算接收总数**:
   - `recv_counts[0] = cum_sum_result[expert_num_per_rank * ep_world_size, 0]`:本rank接收的总token数量

5. **计算每个expert的接收数量**:
   - 遍历本rank负责的expert（0到expert_num_per_rank-1）
   - 对每个expert，从`cum_sum_input`中提取所有rank发送给该expert的数量
   - 计算cumsum得到该expert的总接收数量
   - 存储到`expert_token_nums[expert_id]`

**示例**:
假设ep_world_size=4，expert_num_per_rank=40，则：

- expert 0的计数信息在`cum_sum_input[1:5, 0]`（从rank 0,1,2,3接收的token数量）
- expert 0的总接收数量为`sum(cum_sum_input[1:5, 0])`
- expert 0的0-3卡的接收数据在expand_x中的行偏移为`cum_sum_result[1:5, 0]`

### 数据接收阶段（per expert + per rank嵌套循环）

**目的**:从共享内存中接收token数据和辅助信息，写入输出expand_x。

**实现步骤**:

1. 外层循环遍历本卡所有expert（0到expert_num_per_rank-1）
2. 内层循环遍历每个expert下所有rank（0到ep_world_size-1），其中index = expert_id * ep_world_size + rank_id
3. 对每个(expert, rank)区域：

   - **获取接收数量和偏移**:
     - `cur_count = cum_sum_input[index + 1, 0]`:本卡某expert从该rank接收的token数量
     - `offset = cum_sum_result[index, 0]`:该区域在输出expand_x中的行偏移量
   
   - **接收token数据**:
     - 使用`shmem_get`从`shmem_data`接收数据
     - 源偏移：`[index * batch_size, 0]`
     - 使用`valid_shape=[cur_count, hidden_size]`只接收该区域内的有效数据
     - 写入输出：`expand_x[offset:offset+cur_count, :]`
   
   - **接收辅助信息**:
     - 使用`shmem_get`从`shmem_info`接收数据
     - 源偏移：`[index * batch_size, 0]`
     - 使用`valid_shape=[cur_count, info_out_size]`只接收该区域内的有效数据
     - 写入输出：`assist_info_for_combine[offset:offset+cur_count, :]`

**输出tensor布局**:

```.txt
expand_x布局:
[0:expert_0_count, :]          <- expert 0的token数据
[expert_0_count:expert_1_count, :]  <- expert 1的token数据
...
[sum(prev_counts):, :]        <- expert N的token数据

assist_info_for_combine布局:
[0:expert_0_count, :]          <- expert 0的辅助信息
[expert_0_count:expert_1_count, :]  <- expert 1的辅助信息
...
[sum(prev_counts):, :]        <- expert N的辅助信息
```

每个expert的token数据按expert_id再按rank_id顺序连续存储
