# 使用 Shmem API 实现 Combine Kernel

## 1. Combine 功能概述

Combine 是 MoE（Mixture of Experts）分布式训练中的关键算子，与 Dispatch 算子形成逆操作关系：

- **Dispatch 阶段**：将输入 token 根据 expert_ids 分路由到各个专家所在的 rank
- **Combine 阶段**：将专家处理后的 token 收集回原始 rank，并按照 expert_scales 进行加权合并

Combine 的核心任务是实现 token 的逆向路由和加权聚合。

## 2. Combine Kernel 原型

```python
def moe_distributed_combine_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor], None]:
    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 3})
    def kernel(
        expand_x: pypto.Tensor([row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for_combine: pypto.Tensor([row, 3], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_scales: pypto.Tensor([batch_size, topk], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
        out: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
    ):
        # kernel 实现
        pass
    return kernel
```

## 3. 参数说明

### 3.1 标量参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `batch_size` | int | 批大小，支持 8 或 256 |
| `hidden_size` | int | 隐藏层维度，固定为 5120 |
| `moe_expert_num` | int | 专家总数，固定为 160 |
| `topk` | int | 每个 token 选择的专家数，固定为 8 |
| `ep_world_size` | int | Expert parallel 的 rank 数，支持 4 或 8 |
| `group_name` | str | 通信域名称，长度 1-128 |

### 3.2 输入 Tensor

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| `expand_x` | `[row, hidden_size]` | DT_BF16 | 专家处理后的 token，row = min(topk * batch_size * ep_world_size, batch_size * moe_expert_num)，其中有效 token 数为 `recv_counts[0]` |
| `assist_info_for_combine` | `[row, 3]`` | DT_INT32 | 辅助信息，每行包含 [rank_id, token_id, k_offset]，用于标识 token 的原始位置 |
| `recv_counts` | `[1]` | DT_INT32 | 当前 rank 接收到的 token 总数，也就是 `expand_x` 里的有效 token 数 |
| `expert_scales` | `[batch_size, topk]` | DT_FP32 | 每个 token 对应 topk 个专家的权重，用于加权合并 |

### 3.3 输出 Tensor

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| `out` | `[batch_size, hidden_size]` | DT_BF16 | 合并后的输出，每个 token 是其 topk 个专家输出的加权和 |

## 4. 计算逻辑伪代码

```python
def combine_logic(expand_x, assist_info_for_combine, recv_counts, expert_scales):
    batch_size, topk = expert_scales.shape
    hidden_size = expand_x.shape[1]
    out = zeros([batch_size, hidden_size])

    # 临时存储每个 token 的 topk 个专家输出
    moe_expert_tokens = zeros([batch_size, topk, hidden_size])

    # 阶段 1：发送 token 回原始 rank
    for row_index in range(recv_counts[0]):
        rank_id = assist_info_for_combine[row_index, 0]
        token_id = assist_info_for_combine[row_index, 1]
        k_offset = assist_info_for_combine[row_index, 2]

        # 将 token 发送到原始 rank
        send_to_rank(rank_id, token_id, k_offset, expand_x[row_index])

    # 阶段 2：接收所有发送给当前 rank 的 token
    for token_id in range(batch_size):
        # 等待所有 topk 个专家的 token 都到达
        for k_offset in range(topk):
            moe_expert_tokens[token_id, k_offset] = receive_token(token_id, k_offset)

    # 阶段 3：加权。
    for token_id in range(batch_size):
        out[token_id] = sum(
            expert_scales[token_id, k_offset] * moe_expert_tokens[token_id, k_offset]
            for k_offset in range(topk)
        )

    return out
```

## 5. 计算流程详解

### 5.1 发送阶段（Send Phase）

每个 rank 将自己接收到的 token 发送回原始 rank：

```python
recv_counts_scalar = recv_counts[0]
for row_index in range(recv_counts_scalar):
    rank_id = assist_info_for_combine[row_index, 0]
    token_id = assist_info_for_combine[row_index, 1]
    k_offset = assist_info_for_combine[row_index, 2]

    # 步骤 1：发送数据到目标 rank 的 Shmem
    pypto.set_vec_tile_shapes(1, hidden_size)
    expand_x_tile = expand_x[row_index:row_index + 1, ...]
    shmem_put_out = pypto.distributed.shmem_put(
        expand_x_tile,
        [topk * token_id + k_offset, 0],  # 目标位置
        shmem_data,
        rank_id,  # 目标 rank
    )

    # 步骤 2：发送信号通知目标 rank
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
- 使用 `shmem_put` 将 token 数据写入目标 rank 的共享内存
- 使用 `shmem_signal` 发送信号，信号值累加（ADD 操作）
- 信号位置 `[token_id, 0]` 对应每个 token 的信号计数器

### 5.2 接收阶段（Receive Phase）

每个 rank 等待并接收所有发送给它的 token：

```python
my_pe = pypto.distributed.my_symbolic_pe(group_name)
for token_id in range(batch_size):
    # 步骤 1：等待所有 topk 个专家的信号
    pypto.set_vec_tile_shapes(1, hidden_size)
    wait_until_out = pypto.distributed.shmem_wait_until(
        shmem_data,
        0,  # src_pe
        topk,  # 等待信号值达到 topk
        [1, hidden_size],
        [token_id, 0],  # 等待位置
        cmp=pypto.OpType.EQ,
        clear_signal=True,  # 等待完成后清除信号
        pred=[expand_x],
    )

    # 步骤 2：从 Shmem 读取所有 topk 个专家的输出
    pypto.set_vec_tile_shapes(topk, hidden_size)
    shmem_get_out = pypto.distributed.shmem_get(
        shmem_data,
        my_pe,
        [topk, hidden_size],
        [topk * token_id, 0],  # 读取位置
        pred=[wait_until_out],  # 依赖等待完成
    )
    shmem_get_out = shmem_get_out.view([topk, hidden_size], [0, 0], valid_shape=[topk, hidden_size])
```

**关键点**：
- 使用 `shmem_wait_until` 等待信号值达到 topk
- `clear_signal=True` 确保信号被清除，避免影响后续操作
- 使用 `shmem_get` 一次性读取所有 topk 个专家的输出，减少任务下发次数
- `shmem_get` 返回的 Tensor 形状为 `[topk, hidden_size]`
- 使用 `view()` 方法指定有效数据形状，确保后续计算正确

### 5.3 合并阶段（Combine Phase）

使用 expert_scales 进行加权合并：

```python
# 转换为 FP32 进行计算
pypto.set_vec_tile_shapes(topk // 2, hidden_size)
shmem_get_out_fp32 = pypto.cast(shmem_get_out, pypto.DT_FP32)

# 配置 Cube 矩阵乘法
k_tile_shape = align_up(topk, 16)
l0b_size = 65536
n_tile_shape = l0b_size // pypto.bytes_of(pypto.DT_FP32) // k_tile_shape
pypto.set_cube_tile_shapes([1, 1], [k_tile_shape, k_tile_shape], [n_tile_shape, n_tile_shape])

# 加权求和：out[token_id] = sum(expert_scales[token_id, k_offset] * tokens[token_id, k_offset])
expert_scales_tile = expert_scales[token_id:token_id + 1, :topk]
matmul_out_fp32 = expert_scales_tile.matmul(shmem_get_out_fp32, pypto.DT_FP32)

# 转换回 BF16
matmul_out_fp16 = pypto.cast(matmul_out_fp32, expand_x.dtype)
out[token_id:, :] = matmul_out_fp16
```

**关键点**：
- 使用 Cube 矩阵乘法加速加权求和
- `expert_scales` shape: `[1, topk]`，`shmem_get_out` shape: `[topk, hidden_size]`
- 矩阵乘法结果 shape: `[1, hidden_size]`

## 6. 完整 Kernel 代码

```python
def moe_distributed_combine_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor], None]:
    batch_size = moe_case.batch_size
    hidden_size = moe_case.hidden_size
    moe_expert_num = moe_case.moe_expert_num
    topk = moe_case.topk
    data_type = moe_case.data_type
    ep_world_size = moe_case.ep_world_size
    row = min(topk * batch_size * ep_world_size, batch_size * moe_expert_num)

    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 3})
    def kernel(
        expand_x: pypto.Tensor([row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for_combine: pypto.Tensor([row, 3], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_scales: pypto.Tensor([batch_size, topk], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
        out: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
    ):
        # 创建 Shmem_data
        shmem_data = pypto.distributed.create_shmem_tensor(
            group_name,
            ep_world_size,
            expand_x.dtype,
            [topk * batch_size, hidden_size],
        )

        # 发送 token
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

        # 接收 token
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        for token_id in range(batch_size):
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
            shmem_get_out = pypto.distributed.shmem_get(
                shmem_data,
                my_pe,
                [topk, hidden_size],
                [topk * token_id, 0],
                pred=[wait_until_out],
            )
            shmem_get_out = shmem_get_out.view([topk, hidden_size], [0, 0], valid_shape=[topk, hidden_size])

            pypto.set_vec_tile_shapes(topk // 2, hidden_size)
            shmem_get_out_fp32 = pypto.cast(shmem_get_out, pypto.DT_FP32)

            k_tile_shape = align_up(topk, 16)
            l0b_size = 65536
            n_tile_shape = l0b_size // pypto.bytes_of(pypto.DT_FP32) // k_tile_shape
            pypto.set_cube_tile_shapes([1, 1], [k_tile_shape, k_tile_shape], [n_tile_shape, n_tile_shape])
            expert_scales_tile = expert_scales[token_id:token_id + 1, :topk]
            matmul_out_fp32 = expert_scales_tile.matmul(shmem_get_out_fp32, pypto.DT_FP32)

            matmul_out_fp16 = pypto.cast(matmul_out_fp32, expand_x.dtype)

            out[token_id:, :] = matmul_out_fp16

    return kernel
```

## 7. 总结

Combine kernel 通过 Shmem API 实现了高效的跨 rank token 收集和合并：

1. **发送阶段**：使用 `shmem_put` 发送数据，`shmem_signal` 发送通知
2. **接收阶段**：使用 `shmem_wait_until` 等待所有信号，`shmem_get` 批量读取数据
3. **合并阶段**：使用 Cube 矩阵乘法进行加权求和

这种设计充分利用了 Shmem 的信号机制和批量传输能力，实现了高性能的分布式 MoE combine 操作。
