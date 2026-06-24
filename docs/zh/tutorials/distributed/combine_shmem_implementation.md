# 使用Shmem API实现Combine Kernel

## Combine功能概述

Combine是MoE（Mixture of Experts）分布式训练中的关键算子，与Dispatch算子形成逆操作关系：

- **Dispatch阶段**：将输入token根据expert_ids分路由到各个专家所在的rank
- **Combine阶段**：将专家处理后的token收集回原始rank，并按照expert_scales进行加权合并

Combine的核心任务是实现token的逆向路由和加权聚合。

## Combine Kernel原型

```python
def moe_distributed_combine_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor], None]:
    batch_size = moe_case.batch_size
    # ...
    stitch_function_max_num = 128 if batch_size in (1, 8) else 10

    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": stitch_function_max_num})
    def kernel(
        expand_x: pypto.Tensor([row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for_combine: pypto.Tensor([row, 3], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_scales: pypto.Tensor([batch_size, topk], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
        x_active_mask: pypto.Tensor([batch_size], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        out: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
    ):
        # kernel实现
        pass
    return kernel
```

## 参数说明

### 标量参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `batch_size` | int | 批大小，支持1、8或256 |
| `hidden_size` | int | 隐藏层维度，固定为5120 |
| `moe_expert_num` | int | 专家总数，固定为160 |
| `topk` | int | 每个token选择的专家数，固定为8 |
| `ep_world_size` | int | Expert parallel的rank数，支持2、4、8或16 |
| `group_name` | str | 通信域名称，长度1-128 |

### 输入Tensor

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| `expand_x` | `[row, hidden_size]` | DT_BF16 | 专家处理后的token，`row = min(topk * batch_size * ep_world_size, batch_size * moe_expert_num)`，有效token数为`recv_counts[0]` |
| `assist_info_for_combine` | `[row, 3]` | DT_INT32 | 辅助信息，每行包含 [rank_id, token_id, k_offset]，用于标识token的原始位置 |
| `recv_counts` | `[1]` | DT_INT32 | 当前rank接收到的token总数，也就是`expand_x`里的有效token数 |
| `expert_scales` | `[batch_size, topk]` | DT_FP32 | 每个token对应topk个专家的权重，用于加权合并 |
| `x_active_mask` | `[batch_size]` | DT_INT32 | 标识哪些token为活跃状态，值为1表示活跃，0表示不活跃，注意：1必须排在0之前，即活跃token必须连续排列在前部，例如`{1, 1, 0}`为合法输入，而`{1, 0, 1}`为非法输入 |

### 输出Tensor

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| `out` | `[batch_size, hidden_size]` | DT_BF16 | 合并后的输出，每个token是其topk个专家输出的加权和 |

## 计算逻辑伪代码

```python
def combine_logic(expand_x, assist_info_for_combine, recv_counts, expert_scales, x_active_mask):
    batch_size, topk = expert_scales.shape
    hidden_size = expand_x.shape[1]
    out = zeros([batch_size, hidden_size])

    # 临时存储每个token的topk个专家输出
    moe_expert_tokens = zeros([batch_size, topk, hidden_size])

    # 阶段1：发送token回原始rank
    for row_index in range(recv_counts[0]):
        rank_id = assist_info_for_combine[row_index, 0]
        token_id = assist_info_for_combine[row_index, 1]
        k_offset = assist_info_for_combine[row_index, 2]

        # 将token发送到原始rank
        send_to_rank(rank_id, token_id, k_offset, expand_x[row_index])

    # 阶段2：接收所有发送给当前rank的token（仅处理活跃token）
    for token_id in range(batch_size):
        if x_active_mask[token_id] == 1:
            # 等待所有topk个专家的token都到达
            for k_offset in range(topk):
                moe_expert_tokens[token_id, k_offset] = receive_token(token_id, k_offset)

    # 阶段3：加权（仅处理活跃token）
    for token_id in range(batch_size):
        if x_active_mask[token_id] == 1:
            out[token_id] = sum(
                expert_scales[token_id, k_offset] * moe_expert_tokens[token_id, k_offset]
                for k_offset in range(topk)
            )

    return out
```

## 计算流程详解

### 发送阶段（Send Phase）

每个rank将自己接收到的token发送回原始rank：

```python
recv_counts_scalar = recv_counts[0]
for row_index in range(recv_counts_scalar):
    rank_id = assist_info_for_combine[row_index, 0]
    token_id = assist_info_for_combine[row_index, 1]
    k_offset = assist_info_for_combine[row_index, 2]

    # 步骤1：发送数据到目标rank的Shmem
    pypto.set_vec_tile_shapes(1, hidden_size)
    expand_x_tile = expand_x[row_index:row_index + 1, ...]
    shmem_put_out = pypto.distributed.shmem_put(
        expand_x_tile,
        [topk * token_id + k_offset, 0],  # 目标位置
        shmem_data,
        rank_id,  # 目标rank
    )

    # 步骤2：发送信号通知目标rank
    pypto.distributed.shmem_signal(
        shmem_data,
        0,  # src_pe
        1,  # signal value，表示有一个专家输出到达
        [1, hidden_size],
        [token_id, 0],  # 信号位置
        target_pe=rank_id,
        sig_op=pypto.AtomicType.ADD,  # 累加信号
        pred=[shmem_put_out],  # 依赖数据发送完成
    )
```

**关键点**：

- 使用`shmem_put`将token数据写入目标rank的共享内存
- 使用`shmem_signal`发送信号，信号值累加（ADD操作）
- 信号位置`[token_id, 0]`对应每个token的信号计数器

### 接收阶段（Receive Phase）

每个rank等待并接收所有发送给它的活跃token：

```python
my_pe = pypto.distributed.my_symbolic_pe(group_name)
for token_id in pypto.loop(batch_size, name='MOE_DISTRIBUTED_RECEIVE', idx_name='token_id'):
    # 仅处理活跃token
    if x_active_mask[token_id] == 1:
        # 步骤1：等待所有topk个专家的信号
        pypto.set_vec_tile_shapes(1, hidden_size)
        wait_until_out = pypto.distributed.shmem_wait_until(
            shmem_data,
            0,  # src_pe
            topk,  # 等待信号值达到topk
            [1, hidden_size],
            [token_id, 0],  # 等待位置
            cmp=pypto.OpType.EQ,
            clear_signal=True,  # 等待完成后清除信号
            pred=[expand_x],
        )

        # 步骤2：从Shmem读取所有topk个专家的输出
        pypto.set_vec_tile_shapes(topk, hidden_size)
        shmem_load_out = pypto.experimental.shmem_load(
            shmem_data,
            my_pe,
            [topk, hidden_size],
            [topk * token_id, 0],
            pred=[wait_until_out],
        )
```

**关键点**：

- 使用`if x_active_mask[token_id] == 1:`条件，仅对活跃token执行接收操作
- 使用`shmem_wait_until`等待信号值达到topk
- `clear_signal=True`确保信号被清除，避免影响后续操作
- 使用`pypto.experimental.shmem_load`一次性读取所有topk个专家的输出，减少任务下发次数
- `shmem_load`返回的Tensor形状为`[topk, hidden_size]`

### 合并阶段（Combine Phase）

使用expert_scales进行加权合并（仅对活跃token）：

```python
if x_active_mask[token_id] == 1:
    # 转换为FP32进行计算
    pypto.set_vec_tile_shapes(topk, hidden_size // 2)
    shmem_load_out_fp32 = pypto.cast(shmem_load_out, pypto.DT_FP32)

    # 将expert_scales reshape为 [topk, 1] 以便逐元素乘法
    expert_scales_tile = expert_scales[token_id:(token_id + 1), :]
    expert_scales_tile_reshaped = pypto.reshape(expert_scales_tile, [topk, 1])

    # 加权：每个专家输出乘以对应权重
    mul_out = pypto.mul(shmem_load_out_fp32, expert_scales_tile_reshaped)

    # 沿topk维度求和
    sum_out_fp32 = pypto.sum(mul_out, dim=0, keepdim=True)

    # 转换回BF16
    sum_out_bf16 = pypto.cast(sum_out_fp32, expand_x.dtype)

    out[token_id:, :] = sum_out_bf16
```

**关键点**：

- 使用`mul` + `sum`实现加权求和：先逐元素乘法，再沿topk维度求和
- `expert_scales` reshape为`[topk, 1]`，与`shmem_load_out_fp32` `[topk, hidden_size]`做逐元素乘法
- `mul`结果shape: `[topk, hidden_size]`，`sum`后shape: `[1, hidden_size]`
- 仅对`x_active_mask == 1`的活跃token执行合并计算

## 完整Kernel代码

```python
def moe_distributed_combine_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor], None]:
    batch_size = moe_case.batch_size
    hidden_size = moe_case.hidden_size
    moe_expert_num = moe_case.moe_expert_num
    topk = moe_case.topk
    data_type = moe_case.data_type
    ep_world_size = moe_case.ep_world_size
    row = min(topk * batch_size * ep_world_size, batch_size * moe_expert_num)

    stitch_function_max_num = 128 if batch_size in (1, 8) else 10

    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": stitch_function_max_num})
    def kernel(
        expand_x: pypto.Tensor([row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for_combine: pypto.Tensor([row, 3], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_scales: pypto.Tensor([batch_size, topk], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
        x_active_mask: pypto.Tensor([batch_size], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        out: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
    ):
        # 创建Shmem_data
        shmem_data = pypto.distributed.create_shmem_tensor(
            group_name,
            ep_world_size,
            expand_x.dtype,
            [topk * batch_size, hidden_size],
        )

        # 发送token
        recv_counts_scalar = recv_counts[0]
        for row_index in pypto.loop(recv_counts_scalar, name='MOE_DISTRIBUTED_SEND', idx_name='row_index'):
            rank_id = assist_info_for_combine[row_index, 0]
            token_id = assist_info_for_combine[row_index, 1]
            k_offset = assist_info_for_combine[row_index, 2]

            pypto.set_vec_tile_shapes(1, hidden_size)
            expand_x_tile = expand_x[row_index:row_index + 1, ...]
            shmem_put_out = pypto.distributed.shmem_put(
                expand_x_tile,
                [topk * token_id + k_offset, 0],
                shmem_data,
                rank_id,
            )

            pypto.distributed.shmem_signal(
                shmem_data,
                0,
                1,
                [1, hidden_size],
                [token_id, 0],
                target_pe=rank_id,
                sig_op=pypto.AtomicType.ADD,
                pred=[shmem_put_out],
            )

        # 接收token
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        for token_id in pypto.loop(batch_size, name='MOE_DISTRIBUTED_RECEIVE', idx_name='token_id'):
            if x_active_mask[token_id] == 1:
                pypto.set_vec_tile_shapes(1, hidden_size)
                wait_until_out = pypto.distributed.shmem_wait_until(
                    shmem_data,
                    0,
                    topk,
                    [1, hidden_size],
                    [token_id, 0],
                    cmp=pypto.OpType.EQ,
                    clear_signal=True,
                    pred=[expand_x],
                )

                pypto.set_vec_tile_shapes(topk, hidden_size)
                shmem_load_out = pypto.experimental.shmem_load(
                    shmem_data,
                    my_pe,
                    [topk, hidden_size],
                    [topk * token_id, 0],
                    pred=[wait_until_out],
                )

                pypto.set_vec_tile_shapes(topk, hidden_size // 2)
                shmem_load_out_fp32 = pypto.cast(shmem_load_out, pypto.DT_FP32)
                expert_scales_tile = expert_scales[token_id:(token_id + 1), :]
                expert_scales_tile_reshaped = pypto.reshape(expert_scales_tile, [topk, 1])
                mul_out = pypto.mul(shmem_load_out_fp32, expert_scales_tile_reshaped)
                sum_out_fp32 = pypto.sum(mul_out, dim=0, keepdim=True)
                sum_out_bf16 = pypto.cast(sum_out_fp32, expand_x.dtype)

                out[token_id:, :] = sum_out_bf16

    return kernel
```

## 总结

Combine kernel通过Shmem API实现了高效的跨rank token收集和合并：

1. **发送阶段**：使用`shmem_put`发送数据，`shmem_signal`发送通知
2. **接收阶段**：使用`shmem_wait_until`等待所有信号，`pypto.experimental.shmem_load`批量读取数据，仅处理活跃token
3. **合并阶段**：使用`mul` + `sum`进行加权求和，仅对活跃token合并

这种设计充分利用了Shmem的信号机制和批量传输能力，实现了高性能的分布式MoE combine操作。
