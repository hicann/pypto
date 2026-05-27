# PyPTO 通信算子切块设置指南

## 前置说明

本指南基于 PyPTO 通用切块机制，说明通信算子的特殊切块策略和约束。

建议先阅读以下文档了解切块基础知识：
- [Tiling配置](../development/tiling.md)：TileShape概念、设置方法、通用约束
- [set_vec_tile_shapes API文档](../../api/config/pypto-set_vec_tile_shapes.md)：接口使用说明

通信算子使用相同的 `set_vec_tile_shapes` 接口设置切块，但在**切分策略**和**约束条件**上有特殊处理，详见本指南第3节和第5节。

---

## 1. 背景介绍

在 PyPTO 分布式计算中，通信算子用于实现多卡间的数据传输与同步。由于硬件资源限制（如单次传输大小限制、内存带宽等）和性能优化需求，通信算子通常需要将大数据块划分为多个小块进行传输和操作，这种划分方式称为切块（Tiling）。

切块设置通过 **TileShape** 参数控制，它决定了数据如何被切分以及对应的操作如何展开。切块配置对通信算子的影响主要体现在：

- 功能正确性：不当的切块配置可能导致信号量等超时，精度错误等问题

- 执行性能：合理的切块配置可以提升并行度、减少同步开销、优化流水线排布

- 资源利用：合适的切块大小可以充分利用硬件传输带宽和计算资源

通信算子间存在严格的执行依赖关系（如数据写入完成后才能发送信号、信号到达后才能读取数据），切块配置需要在保证依赖关系正确的前提下，尽可能提升并行度和通信效率。

## 2. 通信算子支持切块情况

| 算子名称            | 切块支持 | 切块含义                 |
| ------------------- | -------- | ------------------------ |
| create_shmem_tensor | -        | -                        |
| create_shmem_signal | -        | -                        |
| shmem_view         | -        | -                        |
| shmem_put           | 支持     | 设置分块发送的数据大小   |
| shmem_get           | 支持     | 设置分块读取的数据大小   |
| shmem_signal        | 支持     | 分块写入信号量           |
| shmem_wait_until    | 支持     | 分块等待信号量           |
| shmem_barrier_all   | 支持     | -                        |
| shmem_clear_data    | -     | - |
| shmem_clear_signal  | -     | - |
| my_symbolic_pe      | -        | -                        |
| shmem_store         | 支持     | 设置分块发送的数据大小   |
| shmem_load          | 支持     | 设置分块读取的数据大小   |

## 3. 通信算子切块策略

切块主要针对 tensor 类型的数据。当前通信算子的输入和输出主要可以划分为两种类型的 tensor：

- **数据类型的 tensor**：用于在通信过程中进行数据交换，会发生实际的数据读取或写入。例如 `shmem_put` 中的 `src` 和 `dst`、`shmem_signal` 中的 `src` 等。它们都是在通信过程中会实际访问到的数据区域。

- **控制边类型的 tensor**：用于建立通信算子前后执行的依赖关系，控制通信算子的执行顺序。在通信过程中，不会对它们进行访问。例如 `shmem_put` 中的 `pred` 以及输出、`shmem_wait_until` 中的 `pred` 以及输出。

将数据类型的张量简称为**data tensor**，控制边类型的张量简称为**dummy tensor**。二者在通信过程中的语义与作用各不相同。

切块的主体对象为 data tensor，切块的核心目的是：在计算图展开阶段，将 data tensor对应的数据区域划分为多个子区域，使单个算子能够被展开为多个 tile 算子（tile_op），从而提升任务执行的并行度。此外，通信算子的执行依赖输入、输出 dummy tensor 建立严格的依赖关系。为在 tile_op 之间构建更细粒度的依赖关系，同样需要对 dummy tensor 进行切分，以此实现更优的流水排布，进一步提升整体并行效率。data tensor 和 dummy tensor 基于 TileShape 进行切分时，采取不同的切分策略。

### 3.1 数据类型的tensor切分策略

当前支持 **N 维切块**（N 可以是 2 - 4）。
假设 data tensor 的 shape 大小为 $[s_1, s_2, \ldots, s_N]$（维度 = N），TileShape 大小为 $[t_1, t_2, \ldots, t_N]$（维度 = N）。data tensor 会被切分为共 $n$ 个数据块。n 的计算方式如下：

$$
tile\_num\_dim_i = \left\lceil \frac{s_{i}}{t_i} \right\rceil \quad (i = 1, 2, \ldots, N) \\
n = \prod_{i=1}^{N} tile\_num\_dim_i
$$

其中：
- $tile\_num\_dim_i$：表示第 $i$ 维切分的数量
- $n$：表示总的数据块数量

切分之后的数据块 $d$ 相对于原数据的位置记为 `tile_index`（0 ≤ tile_index < n）。对于每个数据块 $d$，会更新其 shape 和 offset 信息，并将 $d$ 作为 tile_op 的输入或输出。shape 和 offset 信息更新如下：

**计算步骤**：

1. **计算每个维度索引**：

   ```
   dim_index[N-1] = tile_index % tile_num_dim_N
   dim_index[N-2] = (tile_index / tile_num_dim_N) % tile_num_dim_N-1
   ...
   dim_index[0] = tile_index / (tile_num_dim_2 × tile_num_dim_3 × ... × tile_num_dim_N)
   ```

2. **计算每个块的 offset 和 shape**：

   $$tile\_offset_i = dim\_index_i \times t_i$$
   $$tile\_shape_i = \min(s_i - tile\_offset_i, t_i)$$

**示例**：假设 data tensor shape 为 $[128, 512]$（维度=2），TileShape 为 $[64, 256]$（维度=2）：
- 第1维：$tile\_num\_dim_1 = \lceil 128/64 \rceil = 2$
- 第2维：$tile\_num\_dim_2 = \lceil 512/256 \rceil = 2$
- 总切块数：$n = 2 \times 2 = 4$

切块结果：
| tile_index | dim_indices | shape | offset |
|------------|-------------|-------|--------|
| 0 | [0, 0] | [64, 256] | [0, 0] |
| 1 | [0, 1] | [64, 256] | [0, 256] |
| 2 | [1, 0] | [64, 256] | [64, 0] |
| 3 | [1, 1] | [64, 256] | [64, 256] |


### 3.2 控制边类型的tensor切分策略

对 dummy tensor 切分时，遵循以下核心原则：**检查每个维度是否足够切分，满足条件就切分，不满足就不切分**。

#### 3.2.1 切分条件判断

假设 dummy tensor 的 shape 为 $[p_1, p_2, \ldots, p_N]$（维度 = N），TileShape 维度为 N，需要切分为 $tile\_num\_dim_1, tile\_num\_dim_2, \ldots, tile\_num\_dim_N$（对应 data tensor 的切块数量）。

**切分条件**：dummy tensor 的 维度数与切块维度一致，且每个维度大小必须大于或等于对应维度的切块数量：

$$
p_i \geq tile\_num\_dim_i \quad (i = 1, 2, \ldots, N)
$$

若 dummy tensor 维度不等于 N，或某个维度大小不足，则不满足切分条件。

#### 3.2.2 满足条件时的切分策略

当满足切分条件时，采用"平均分配+尾块优先"策略对 dummy tensor 进行 N 维切分：

**计算步骤**：

1. **计算每个维度索引**：从 `tile_index` 反推各维度的索引位置

   ```
   dim_indices[N-1] = tile_index % tile_num_dim_N
   dim_indices[N-2] = (tile_index / tile_num_dim_N) % tile_num_dim_N-1
   ...
   dim_indices[0] = tile_index / (tile_num_dim_2 × tile_num_dim_3 × ... × tile_num_dim_N)
   ```

2. **计算每个维度的基准大小和尾块数**：

   $$base_i = p_i / tile\_num\_dim_i \quad (向下取整)$$
   $$rem_i = p_i \bmod tile\_num\_dim_i$$

3. **计算每个块的 shape**：
   - 若 $dim\_indices[i] < rem_i$（属于前 $rem_i$ 个块）：$tile\_shape_i = base_i + 1$
   - 若 $dim\_indices[i] \geq rem_i$（属于后面的块）：$tile\_shape_i = base_i$

4. **计算每个块的 offset**（累加前面所有块的大小）：
   - 若 $dim\_indices[i] < rem_i$：$tile\_offset_i = dim\_indices[i] \times (base_i + 1)$
   - 若 $dim\_indices[i] \geq rem_i$：$tile\_offset_i = rem_i \times (base_i + 1) + (dim\_indices[i] - rem_i) \times base_i$

**示例**：假设 dummy tensor shape 为 $[10, 7]$，需要切分为 $tile\_num\_dim_1 = 3$（行）、$tile\_num\_dim_2 = 2$（列）。检查条件：
- 行方向：$p_1 = 10 \geq tile\_num\_dim_1 = 3$ ✓
- 列方向：$p_2 = 7 \geq tile\_num\_dim_2 = 2$ ✓

满足条件，进行切分：
- 行方向：$base\_row = 10 / 3 = 3$, $rem\_row = 10 \bmod 3 = 1$（前1个行块多1行）
- 列方向：$base\_col = 7 / 2 = 3$, $rem\_col = 7 \bmod 2 = 1$（前1个列块多1列）

切块矩阵：

| tile_index | dim_indices | shape | offset |
|------------|-------------|-------|--------|
| 0 | [0, 0] | [4, 4] | [0, 0] |
| 1 | [0, 1] | [4, 3] | [0, 4] |
| 2 | [1, 0] | [3, 4] | [4, 0] |
| 3 | [1, 1] | [3, 3] | [4, 4] |
| 4 | [2, 0] | [3, 4] | [7, 0] |
| 5 | [2, 1] | [3, 3] | [7, 4] |

#### 3.2.3 不满足条件时的处理策略

当 dummy tensor 的维度和切块维度不一致，或者某个维度大小小于对应的切块数量时（即 $p_i < tile\_num\_dim_i$），不满足切分条件，此时**不对 dummy tensor 进行切分**。

所有 tile_op 共用同一个 dummy tensor，依赖该 op 的算子需要等所有展开的 tile_op 执行完成后方可执行。

**示例**：假设 dummy tensor shape 为 $[2, 4]$，需要切分为 $tile\_num\_dim_1 = 3$（行）、$tile\_num\_dim_2 = 2$（列）。检查条件：
- 行方向：$p_1 = 2 < tile\_num\_dim_1 = 3$ ✗

不满足条件，不进行切分，所有 tile_op 共用原始 dummy tensor。

## 4. 具体样例

以双卡的 allreduce 通信为例，具体介绍切块对通信算子执行的影响。

**示例代码：**

```python
def allreduce_kernel(
    input_tensor: pypto.Tensor([128, 256], pypto.DT_INT32),
    output_tensor: pypto.Tensor([128, 256], pypto.DT_INT32),
    group_name,
    world_size = 2,
):
    # 创建shmem数据
    shmem_shape = [128, 256]
    shmem_tensor = pypto.distributed.create_shmem_tensor(
        group_name, world_size, pypto.DT_INT32, shmem_shape)

    # 数据发送和信号量写入
    pypto.set_vec_tile_shapes(64, 256)
    for dyn_idx in range(world_size):
        put_dummy = pypto.distributed.shmem_put(input_tensor, [0, 0], shmem_tensor, dyn_idx,
            put_op=pypto.AtomicType.ADD, pred=[input_tensor])
        pypto.distributed.shmem_signal(shmem_tensor, dyn_idx, 1, shmem_shape,
            [0, 0], target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_dummy])

    # 信号量等待
    my_pe = pypto.distributed.my_symbolic_pe(group_name)
    wait_until_dummy = pypto.distributed.shmem_wait_until(shmem_tensor, my_pe, world_size,
        shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[input_tensor])

    # 数据读取
    pypto.set_vec_tile_shapes(32, 256)
    all_reduce_out = pypto.distributed.shmem_get(
        shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_dummy], valid_shape=shmem_shape
    )
    output_tensor.move(all_reduce_out)

```

**代码解释：**

上述代码的执行流程可分为：创建 shmem 数据 → 数据发送和信号量写入 → 信号量等待 → 数据读取。

**数据发送和信号量写入阶段：**

本阶段中，每个处理单元（PE）将**依次向通信域内所有 PE 的共享内存张量（shmem tensor）** 写入数据，并通过写入信号量标记数据写入完成状态。共享内存信号量操作（shmem_signal）与共享内存数据写入操作（shmem_put）的执行顺序，由 `shmem_put` 的输出 `put_dummy` 进行严格依赖控制。

在上述用例中，输入数据形状 `[128, 256]`，切块形状 `TileShape [64, 256]`，框架将自动完成细粒度的算子切分与依赖管理：

- **shmem_put 切分**：单个 `shmem_put` 算子被切分为 `tile_shmem_put0`、`tile_shmem_put1`，实现数据的分块写入；
- **shmem_signal 切分**：单个 `shmem_signal` 算子被切分为 `tile_shmem_signal0`、`tile_shmem_signal1`，实现信号量的分块写入；
- **put_dummy 切分**：`put_dummy` 同步切分为两个独立节点，分别作为对应 `tile_shmem_put` 的输出、`tile_shmem_signal` 的输入，构建 tile_op 级别的依赖关系。

基于上述切分与依赖设计，`tile_shmem_signal0` 无需等待 `tile_shmem_put1` 执行完成，仅在 `tile_shmem_put0` 执行结束后即可触发执行，实现并行化的细粒度调度。

**信号量等待阶段：**

本阶段通过等待信号量达到条件值，确认通信域内各 PE 已完成目标共享内存张量（shmem tensor）的数据写入。

在上述用例中，输入数据形状 `[128, 256]`，切块形状 `TileShape [64, 256]`，框架将自动完成细粒度的算子切分与依赖管理：

- **shmem_wait_until 切分**：单个 `shmem_wait_until` 算子被切分为 `tile_shmem_wait_until0`、`tile_shmem_wait_until1`；
- **wait_until_dummy 切分**：`wait_until_dummy` 同步切分为 `tile_wait_until_dummy0`、`tile_wait_until_dummy1`，作为后继算子的控制边。

基于上述切分与依赖设计，支持数据分块独立就绪，单块数据写入完成即可通知后继算子读取，无需等待全部数据写入完成。

**数据读取阶段：**

本阶段从共享内存张量（shmem tensor）中读取数据，需在依赖的 `shmem_wait_until` 执行完成后运行，确保目标数据已写入指定地址。

在上述用例中，输入数据形状 `[128, 256]`，切块形状 `TileShape [32, 256]`，框架将自动完成细粒度的算子切分与依赖管理：

- **shmem_get 切分**：单个 `shmem_get` 算子被切分为 4 个 tile_op：`tile_shmem_get0`、`tile_shmem_get1`、`tile_shmem_get2`、`tile_shmem_get3`，实现分块读取；
- **wait_until_dummy 切分**：`wait_until_dummy` 同步切分为四块：`tile_wait_until_dummy00`、`tile_wait_until_dummy01`、`tile_wait_until_dummy10`、`tile_wait_until_dummy11`；
- **依赖关联关系**：`tile_wait_until_dummy00/01` 对应等待阶段的 `tile_wait_until_dummy0`，`tile_wait_until_dummy10/11` 对应等待阶段的 `tile_wait_until_dummy1`；
- **细粒度调度**：`tile_shmem_wait_until0` 执行完成后，`tile_shmem_get0/1` 即可立即执行，无需等待所有分块等待操作完成。

## 5. 切块约束条件

合理切块是通信算子正常执行的前提。完整通信流程包含**写数据、写信号量、等信号量、读数据**四个阶段，各阶段支持独立配置切块策略。通信各阶段相互独立但又高度关联，切块配置必须保证整体通信流程符合语义，否则会引发信号量等待超时、计算精度异常等问题。

通信算子切块设置需遵循以下约束条件：

1. **切块数量不超过 1023 块**

由于底层哈希表和任务数组的性能限制，单个算子切分后的 tile_op 数量不得超过 1023 块。该限制来源于 `shmem_wait_until` 等通信算子底层实现中的固定大小数组 `SignalTileOp* hashTable[AICPU_TASK_ARRAY_SIZE]`，其中 `AICPU_TASK_ARRAY_SIZE` 定义为 1024（预留1个位置用于边界检查）。超出此限制会导致运行时错误：`taskCount >= AICPU_TASK_ARRAY_SIZE`。

在设计切块策略时，应确保根据 data tensor 的 shape 和 TileShape 计算得到的总切块数量 `n = tile_row_num × tile_col_num` 不超过 1023。

2. **切块维度**

TileShape 的维度数与 shmem tensor 的维度数一致。

3. **shmem_signal 写信号量时需确保对应的数据已经发送完成**

`shmem_signal` 通常在 `shmem_put` 之后写入信号量，用于通知 PE 对应的数据块已写入完成。二者的执行依赖关系，通过 `shmem_put` 输出的 `dummy` 进行控制。由于 dummy tensor 和 data tensor 切分策略不同，当 `shmem_signal` 与 `shmem_put` 这两个接口设置不同的切块时，需分析切块后的 `shmem_signal` 写入的信号量是否符合语义，即其对应的数据块是否已通过 `shmem_put` 写入完成，否则可能引起精度错误。

**示例分析：**

```python
# 示例 1：符合语义
input_tensor = pypto.tensor([64, 64], pypto.DT_INT32)
pypto.set_vec_tile_shapes(16, 64)
put_dummy = pypto.distributed.shmem_put(input_tensor, [0, 0], shmem_tensor, 0,
              put_op=pypto.AtomicType.ADD, pred=[input_tensor])
pypto.set_vec_tile_shapes(32, 64)
pypto.distributed.shmem_signal(shmem_tensor, 0, 1, shmem_shape,
              [0, 0], target_pe=0, sig_op=pypto.AtomicType.ADD, pred=[put_dummy])

# 示例 2：不符合语义
input_tensor = pypto.tensor([64, 64], pypto.DT_INT32)
pypto.set_vec_tile_shapes(16, 64)
put_dummy = pypto.distributed.shmem_put(input_tensor, [0, 0], shmem_tensor, 0,
              put_op=pypto.AtomicType.ADD, pred=[input_tensor])
pypto.set_vec_tile_shapes(33, 64)
pypto.distributed.shmem_signal(shmem_tensor, 0, 1, shmem_shape,
              [0, 0], target_pe=0, sig_op=pypto.AtomicType.ADD, pred=[put_dummy])
```

**示例 1** 中，输入数据的 shape 为 [64, 64]，`shmem_put` 与 `shmem_signal` 切块配置及执行逻辑如下：

- **shmem_put 切分**：设置 TileShape 为 [16, 64]，单个 `shmem_put` 被切分为 4 个 `tile_shmem_put`（`tile_shmem_put0`、`tile_shmem_put1`、`tile_shmem_put2`、`tile_shmem_put3`）；同时 `put_dummy` 切分为 4 个 `tile_put_dummy`（`tile_put_dummy0`、`tile_put_dummy1`、`tile_put_dummy2`、`tile_put_dummy3`）；
- **shmem_signal 切分**：设置 TileShape 为 [32, 64]，单个 `shmem_signal` 被切分为 2 个 `tile_shmem_signal`（`tile_shmem_signal0`、`tile_shmem_signal1`）；同时 `put_dummy` 同步切分为 2 个 `tile_put_dummy`，分别对应 `tile_put_dummy0`、`tile_put_dummy1` 和 `tile_put_dummy2`、`tile_put_dummy3`；
- **执行逻辑**：`tile_shmem_signal0` 需在 `tile_shmem_put0` 和 `tile_shmem_put1` 执行完成后触发，标识 [32, 64] 大小的数据已写入，数据写入与信号量对应关系正确，符合语义要求。

**示例 2**：`shmem_put` 与 `shmem_signal` 的切块数量，以及切块后 `tile_op` 之间的依赖关系与示例 1 保持一致：

- **切块配置**：`shmem_put` 与 `shmem_signal` 切块数量，以及依赖关系沿用示例 1 的对应规则；
- **执行逻辑**：`tile_shmem_signal0` 在 `tile_shmem_put0` 和 `tile_shmem_put1` 执行完成后触发，标识已写入 [33, 64] 大小的数据；
- **问题说明**：实际 `tile_shmem_put0` 和 `tile_shmem_put1` 共写入 [32, 64] 大小的数据，信号量标识的数据量与实际写入数据量不匹配，数据写入与信号量对应关系不正确，不符合语义，可能引起精度问题。

4. **shmem_wait_until 和 shmem_signal 需要设置相同的切块大小**

两者操作同一个 shared memory tensor，需要精确匹配。如果切块大小不同，signal 写入的 tile 块和 wait_until 等待的 tile 块不对应，会引起等信号量超时错误。

5. **shmem_get 读取数据前需确保数据已经写入完成**

`shmem_get` 通常在 `shmem_wait_until` 之后执行，以确保目标数据完成写入后读取。二者之间的执行依赖关系由 `shmem_wait_until` 输出的 dummy 张量进行控制。

由于 dummy tensor 和 data tensor 切分策略不同，当这两个接口配置不同的切块方式时，需要保证 `shmem_get` 读取的数据块语义正确，即其对应的数据块已完成写入，否则可能导致精度异常。

```python
# 示例 1：符合语义
input_tensor = pypto.tensor([64,64], pypto.DT_INT32)
pypto.set_vec_tile_shapes(16, 64)
put_dummy = pypto.distributed.shmem_put(input_tensor, [0, 0], shmem_tensor, 0,
            put_op=pypto.AtomicType.ADD, pred=[input_tensor])
pypto.distributed.shmem_signal(shmem_tensor, 0, 1, shmem_shape,
            [0, 0], target_pe=0, sig_op=pypto.AtomicType.ADD, pred=[put_dummy])
my_pe = pypto.distributed.my_symbolic_pe(group_name)
wait_until_dummy = pypto.distributed.shmem_wait_until(shmem_tensor, my_pe, world_size,
        shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[input_tensor])
pypto.set_vec_tile_shapes(32, 64)
all_reduce_out = pypto.distributed.shmem_get(
        shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_dummy], valid_shape=shmem_shape
    )
# 示例 2：不符合语义
input_tensor = pypto.tensor([64,64], pypto.DT_INT32)
pypto.set_vec_tile_shapes(32, 64)
put_dummy = pypto.distributed.shmem_put(input_tensor, [0, 0], shmem_tensor, 0,
            put_op=pypto.AtomicType.ADD, pred=[input_tensor])
pypto.distributed.shmem_signal(shmem_tensor, 0, 1, shmem_shape,
            [0, 0], target_pe=0, sig_op=pypto.AtomicType.ADD, pred=[put_dummy])
my_pe = pypto.distributed.my_symbolic_pe(group_name)
wait_until_dummy = pypto.distributed.shmem_wait_until(shmem_tensor, my_pe, world_size,
        shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[input_tensor])
pypto.set_vec_tile_shapes(33, 64)
all_reduce_out = pypto.distributed.shmem_get(
        shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_dummy], valid_shape=shmem_shape
    )
```

**示例 1** 中，输入数据的 shape 为 [64, 64]，`shmem_wait_until` 与 `shmem_get` 切块配置及执行逻辑如下：

- **shmem_wait_until 切分**：设置 TileShape 为 [16, 64]，单个 `shmem_wait_until` 被切分为 4 个 `tile_shmem_wait_until`（`tile_shmem_wait_until0`、`tile_shmem_wait_until1`、`tile_shmem_wait_until2`、`tile_shmem_wait_until3`）；同时 `wait_until_dummy` 切分为 4 个 `tile_wait_until_dummy`（`tile_wait_until_dummy0`、`tile_wait_until_dummy1`、`tile_wait_until_dummy2`、`tile_wait_until_dummy3`）；
- **shmem_get 切分**：设置 TileShape 为 [32, 64]，单个 `shmem_get` 被切分为 2 个 `tile_shmem_get`（`tile_shmem_get0`、`tile_shmem_get1`）；同时 `wait_until_dummy` 同步切分为 2 个 `tile_wait_until_dummy`，分别对应 `tile_wait_until_dummy0`、`tile_wait_until_dummy1` 和 `tile_wait_until_dummy2`、`tile_wait_until_dummy3`；
- **执行逻辑**：`tile_shmem_get0` 会在 `tile_shmem_wait_until0` 和 `tile_shmem_wait_until1` 执行完成后执行，用于读取 [32, 64] 大小的数据，与等待的信号量对应关系正确，符合语义要求。

**示例 2**：`shmem_wait_until` 与 `shmem_get` 的切块数量，以及切块后 `tile_op` 之间的依赖关系与示例 1 保持一致：

- **切块配置**：`shmem_wait_until` 与 `shmem_get` 切块数量，以及依赖关系沿用示例 1 的对应规则；
- **执行逻辑**：`tile_shmem_get0` 在 `tile_shmem_wait_until0` 和 `tile_shmem_wait_until1` 执行完成后执行，读取 [33, 64] 大小的数据；
- **问题说明**：实际 `tile_shmem_wait_until0` 和 `tile_shmem_wait_until1` 等待的信号量对应的数据区域大小为 [32, 64]，信号量标识的数据量与实际读取数据量不匹配，不符合语义，可能引起精度问题。
