# Matmul AllReduce + Add + RMS Norm 融合算子

## 背景介绍

在推理/训练场景下，利用分布式场景对计算进行加速是十分常见的。目前主流的模型训练/推理过程中都会通过集合通信库 (CCL) 来实现多卡环境下的数据交换以减少算子的训练/推理时间。MatmulAllReduce 算子通常被用在 DecoderOnly 架构下 TP 并行的多头注意力 (MHA) 网络层中，通过将每个卡上对应注意力头的权重 $W_O$ 预先按行切分并分配到卡上，实现完整的多头注意力计算，如下：

$$
MHA(x) = Concat(head_0, head_1, \cdots, head_n) W_O
$$

另一方面，由于 PreNorm 在训练过程中更加稳定，训练收敛速度也比 PostNorm 更快，从理论上证明相对较优，因此目前主流的模型架构都采用 PreNorm 的方式对数据进行归一化。即先将注意力输出进行残差连接，然后进行数据归一化。

$$
PreNorm(x) = x + norm(MHA(x))
$$

RMS Norm 由于其计算简单，计算效率更高，且保持了 LayerNorm 的大部分优势如平移不变性、数值稳定性，作为 PerNorm 中的归一化层在 Qwen3/DeepSeek 等大模型架构中被广泛使用。其计算如下：

$$
RMSNorm(x)=\frac{x}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma + \beta
$$

本文将通过 PyPTO 实现上述融合算子。

## 开发介绍

本文以每张卡上的数据类型为 bfloat16，每个输入左矩阵为 $BS \times D$，权重矩阵为 $H \times D$，首先给出整体的代码实现：

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128,
                     "stitch_cfgcache_size": 100000000},
)
def matmul_allreduce_add_rmsnorm_kernel(
    in_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    matmul_weight: pypto.Tensor(),
    residual: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    gamma: pypto.Tensor(),
    bias: pypto.Tensor(),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    residual_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    eps,
    group_name,
    world_size,
):
    batch_size = in_tensor.shape[0]
    hidden_size = matmul_weight.shape[0]

    in_tensor_mean_coff = 1.0 / hidden_size
    view_row_shape = 8
    bs_loop = (batch_size + view_row_shape - 1) // view_row_shape

    pypto.set_vec_tile_shapes(hidden_size)
    gamma_2d = pypto.reshape(gamma, [1, hidden_size], inplace=True)
    bias_2d = pypto.reshape(bias, [1, hidden_size], inplace=True)

    for bs_idx in pypto.loop(bs_loop, name="LOOP_MM_ALLREDUCE_ADD_RMSNORM", idx_name="bs_idx"):
        # 1. create shmem tesnor
        shmem_shape = [view_row_shape, hidden_size]
        shmem_tensor = pypto.distributed.create_shmem_tensor(
            group_name, world_size, pypto.DT_FP32, shmem_shape)
        shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, world_size)
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        for _ in pypto.loop(1, name="LOOP_MM_AR_ARMS_L0", idx_name="_"):
            in_tensor_tile = pypto.view(
                in_tensor, (view_row_shape, in_tensor.shape[1]), [bs_idx * view_row_shape, 0],
                valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]])

            # 2. clear data
            pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
            data_clear_out = pypto.distributed.shmem_clear_data(
                shmem_tensor, shmem_shape, [0, 0], pred=[in_tensor_tile])
            signal_clear_out = pypto.distributed.shmem_clear_signal(
                shmem_tensor, pred=[in_tensor_tile])
            barrier_out = pypto.distributed.shmem_barrier_all(
                shmem_barrier_signal, [data_clear_out, signal_clear_out])

            # 3. matmul
            pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512])
            matmul_result = pypto.matmul(in_tensor_tile, matmul_weight, pypto.DT_FP32, b_trans=True)

            # 4. allreduce
            pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
            for dyn_idx in range(world_size):
                put_out = pypto.distributed.shmem_put(matmul_result, [0, 0], shmem_tensor, dyn_idx,
                    put_op=pypto.AtomicType.ADD, pred=[barrier_out])
                pypto.distributed.shmem_signal(shmem_tensor, dyn_idx, 1, shmem_shape,
                    [0, 0], target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])
            wait_until_out = pypto.distributed.shmem_wait_until(shmem_tensor, my_pe, world_size,
                shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[in_tensor_tile])
            pypto.set_vec_tile_shapes(1, hidden_size)
            all_reduce_out = pypto.experimental.shmem_load(
                shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out], valid_shape=shmem_shape
            )

            # 5. Add RmsNorm
            residual_tile = pypto.view(
                residual, (view_row_shape, hidden_size), [bs_idx * view_row_shape, 0],
                valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), hidden_size])

            # add
            residual_tile_fp32 = pypto.cast(residual_tile, pypto.DT_FP32)
            add_out = pypto.add(all_reduce_out, residual_tile_fp32)

            # rms norm
            square = pypto.mul(add_out, add_out)
            mean_res = pypto.mul(square, in_tensor_mean_coff)
            reduce_asum = pypto.sum(mean_res, -1, True)
            reduce_sum = pypto.add(reduce_asum, eps)
            reduce_sqrt = pypto.sqrt(reduce_sum)
            res_div = pypto.div(add_out, reduce_sqrt)

            hidden_bf16 = pypto.tensor([view_row_shape, hidden_size], pypto.DT_BF16, "hidden_bf16")
            residual_bf16_tmp = pypto.cast(add_out, in_tensor.dtype)
            for tmp_idx in range(view_row_shape):
                gamma_2d_fp32 = pypto.cast(gamma_2d, pypto.DT_FP32)
                bias_2d_fp32 = pypto.cast(bias_2d, pypto.DT_FP32)
                res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])
                res = pypto.mul(res_div_single, gamma_2d_fp32)
                res_add = pypto.add(res, bias_2d_fp32)
                in_tensor_norm = pypto.cast(res_add, in_tensor.dtype)
                hidden_bf16[tmp_idx:tmp_idx + 1] = in_tensor_norm

            residual_out[bs_idx * pypto.symbolic_scalar(view_row_shape):] = residual_bf16_tmp
            out_tensor[bs_idx * pypto.symbolic_scalar(view_row_shape):] = hidden_bf16
```

在上述代码中，各个输入参数的含义如下：

- `in_tensor`：输入左矩阵，形状大小为 $BS \times D$，数据类型为 bfloat16；
- `matmul_weight`：权重矩阵，形状大小为 $H \times D$，数据类型为 bfloat16；
- `residual`：原始输入对应的残差连接项，形状大小为 $BS \times H$，数据类型为 bfloat16；
- `gamma`：RMS Norm 中的缩放因子 $\gamma$，形状大小为 $1 \times H$，数据类型为 bfloat16；
- `bias`：RMS Norm 中的偏移项 $\beta$，形状大小为 $1 \times H$，数据类型为 bfloat16；
- `out_tensor`：输出矩阵，形状大小为 $BS \times H$；
- `residual_out`：残差连接后的中间结果，输出项；
- `eps`：RMS Norm 中的 $\epsilon$ 参数，保证除数不为零；
- `group_name`：通信域名称
- `world_size`：通信域中的进程数

上述代码实现主要分为以下几步：

1. 按照行对输入左矩阵进行切分，得到一个形状大小 $8 \times D$ 的切块矩阵；
2. 将切块矩阵与权重矩阵进行矩阵乘法，得到一个形状大小为 $8 \times H$ 的中间结果；
3. 将该中间结果广播给通信域中的其他卡，并通过 AtomicAdd 操作完成规约和的计算；
4. 通过阻塞等信号完成获取规约和的结果；
5. 从原始输入的残差连接项对应的矩阵中切分出一块 $8 \times H$ 的切块矩阵；
6. 将该矩阵域规约和的结果相加，得到残差连接后的结果；
7. 计算上述结果的 RMS Norm。

## MatmulAllReduce 卡间通信

为了支持细粒度的切分，目前 pypto 框架中主要通过事件/信号同步的方式来完成卡间通信。并将卡间通信过程抽象为以下操作：

- **create_shmem_tensor**：在当前 rank 上创建一片共享数据缓冲区内存以供本 rank 以及远端 rank 进行读写；
- **create_shmem_signal**：在当前 rank 上创建一片共享信号缓冲区内存以记录当前 rank 与远端 rank 之间的通信状态【内存是否读写完成】；
- **shmem_put**：将当前 rank 对应的数据写入远端 rank 对应的数据缓冲区内存；
- **shmem_signal**：通知远端 rank 当前 rank 已经完成其数据缓冲区内存的写入，并在远端 rank 的信号缓冲区中写入指定值；
- **shmem_wait_until**：等待所有远端 rank 完成当前 rank 的数据缓冲区内存写入操作，即判断当前 rank 的共享信号缓冲区中是否为指定值；
- **shmem_load**：从某个 rank 上读取数据缓冲区的内存，并加载到当前 rank 的内存中。

由于数据/信号缓冲区的大小有上限，默认数据缓冲区的大小为 200M，信号缓冲区的大小为 1M。因此在进行通信时，需要对输入数据进行切分，上述代码中通过控制切分块行数为 8 实现一个指定的块切分以保证某场景下不会超过数据/信号缓冲区的上限。实际值应该由用户场景自行决定。

在以行对输入左矩阵进行切分后，其与权重矩阵的矩阵乘结果需要通过 Put 操作与其他卡进行数据通信过程，因此需要一片共享数据缓冲区进行通信。

```python
shmem_tensor = pypto.distributed.create_shmem_tensor(
            group_name, world_size, pypto.DT_FP32, shmem_shape)
```

上述代码创建了一个对应通信域名称为 `group_name`，通信域中进程数为 `world_size` 的通信域，通信域的数据缓冲区大小与 `shmem_shape` 一致，数据类型为 DT_FP32 即单精度浮点类型。需要注意，由于在每个进程上都会执行上述代码，因此每个进程都具备一片上述的数据缓冲区作为本进程对应通信域下的数据缓冲区作为共享内存。返回结果 `shmem_tensor` 中既包含了数据缓冲区，也绑定了该数据缓冲区对应的信号缓冲区。

为了保证每个切片间互不干扰，代码实现首先通过 `shmem_clear_data`/`shmem_clear_signal` 将当前切块对应的共享数据/信号缓冲区的内存数据置为 0，并通过 `shmem_barrier_all` 等待共享信号缓冲区以及信号区的置 0 操作全部完成。

随后通过 matmul 算子将做切块矩阵域权重矩阵做矩阵乘法计算，由于权重矩阵的形状大小为 $D \times H$，而切块矩阵的形状大小为 $8 \times H$，因此权重矩阵需要进行转置变为 $H \times D$ 后进行计算，在代码实现中通过 `b_trans=True` 进行配置。并且在矩阵乘法中通过指定输出类型为 DT_FP32 保证其与数据缓冲区的大小一致。

矩阵乘法完成后，代码实现通过广播的方式将矩阵乘法结果告知通信域中所有 rank 并写入其对应的数据缓冲区中。

```python
for dyn_idx in range(world_size):
    put_out = pypto.distributed.shmem_put(matmul_result, [0, 0], shmem_tensor, dyn_idx,
        put_op=pypto.AtomicType.ADD, pred=[barrier_out])
    pypto.distributed.shmem_signal(shmem_tensor, dyn_idx, 1, shmem_shape,
        [0, 0], target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])
wait_until_out = pypto.distributed.shmem_wait_until(shmem_tensor, my_pe, world_size,
    shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[in_tensor_tile])
pypto.set_vec_tile_shapes(1, hidden_size)
all_reduce_out = pypto.experimental.shmem_load(
    shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out], valid_shape=shmem_shape
)
```

通过 `dyn_idx` 对通信域中所有其他 rank 进行遍历，并将结果通过 `shmem_put` 写入该 rank 的数据缓冲区中，写入时的偏移为 [0， 0] (该场景下数据缓冲区的大小为三维，形状为 [8, H])。并通过 `AtomicType.ADD` 指定写入模式为在原内存的值上做累加。由于远端 rank 需要知道该写操作是否完成，因此在 `shmem_put` 操作后需要通过 `shmem_signal` 告知远端 rank 该操作已完成，该方法的参数与 `shmem_put` 类似，不做赘述，需要注意的时该方法的参数 `1` 为写入远端 rank 信号区内存的值。

广播完成后，当前进程所在的 rank 需要从自身的数据缓冲区中获取最终结果，而在这之前必须保证所有远端 rank 的内存写入操作完成，因此需要通过 `shmem_wait_until` 操作阻塞流程，直至所有内存写入完成后才进行后续操作。由于在上述 `shmem_signal` 过程中代码实现会将远端 rank 所在的信号区内存值加 1，当所有写入完成时最终信号区的值会为通信域中的进程数，即通信域大小。因此 `shmem_wait_until` 中通过指定比较方式为 `pypto.OpType.EQ`，比较值为 `world_size` 确保所有 rank 广播操作完成。

广播操作完成后则通过 `shmem_load`/`shmem_get` 获取数据缓冲区的数据，并加载到当前 rank 的 UB/GM 当中。这里为了实现 UB 层级的内存复用，使用 `shmem_load` 将结果加载到 UB 中，以减少下面 残差连接 + RMS Norm 阶段的数据拷贝。

## 残差连接 + RMS Norm

### 残差连接

完成 MatmulAllReduce 后，其输出 `all_reduce_out` 被加载到 UB 中以供复用。由于 MatmulAllReduce 阶段通过行切分输入左矩阵，最终 `all_reduce_out` 的结果也对应输入左矩阵切片的行位置，再进行残差计算时 `residual` 也需要保持一致。因此通过 `view` 操作将切块对应后使用 `add` 操作完成残差连接的计算。同时由于矩阵输出的类型为 DT_FLOAT 而 `residual` 的类型为 DT_BF16(bloat16)，因此需要通过 `cast` 操作将类型统一。

```python
residual_tile = pypto.view(
                residual, (view_row_shape, hidden_size), [bs_idx * view_row_shape, 0],
                valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), hidden_size])

residual_tile_fp32 = pypto.cast(residual_tile, pypto.DT_FP32)
add_out = pypto.add(all_reduce_out, residual_tile_fp32)
```

### RMS Norm

RMS Norm 主要对残差连接层的计算结果 `add_out` 按照如下公式进行计算：

$$
RMSNorm(x)=\frac{x}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma + \beta
$$

## 切分逻辑

MatmulAllReduce + Add + RMS Norm 融合算子涉及到了通信算子、矩阵算子以及 Vector 算子。其中通信算子以及 Vector 算子共用 `set_vec_tile_shapes` 来指定切分大小。

为了保证程序使用的共享内存不超过数据缓冲区，上述实现并未直接分配一个指定大小的共享缓冲区，而是通过 `pypto.set_vec_tile_shapes(view_row_shape, hidden_size)` 按指定的 `view_row_shape=8` 进行切分，每个切分块最多只使用 $8 \cdot H \cdot sizeof(float)$ 大小的共享缓冲区。

而在矩阵计算过程中，通过设置 `pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512])` 将输入按照 [8, 128, 256] 和 [8, 256, 512] 进行切分，在 matmul 算子内部形成更多子图进行计算，其中 8 保证与 `view_row_shape` 一致，保证该维度能够一次计算完成，通过优化后面两个切分大小提高计算效率。

Add + RMS Norm 阶段通过 `pypto.set_vec_tile_shapes(1, hidden_size)` 让程序按行进行切分，并通过该大小进行各种后续计算。
