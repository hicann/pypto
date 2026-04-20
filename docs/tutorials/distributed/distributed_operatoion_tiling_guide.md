# PyPTO 通信算子切块设置指南

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
| shmem_view-         | -        | -                        |
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

假设 data tensor 的 shape 大小为 $[x, y]$，TileShape 大小为 $[t_1, t_2]$。data tensor 会被切分为 $d_1, d_2, \ldots, d_n$ 共 n 个数据块。n 的计算方式如下所示：

$$
tile\_row\_num = \left\lceil \frac{x}{t_1} \right\rceil \\
tile\_col\_num = \left\lceil \frac{y}{t_2} \right\rceil \\
n = tile\_row\_num \times tile\_col\_num
$$

其中：
- `tile_row_num`：表示第一维切分的数量
- `tile_col_num`：表示第二维切分的数量
- `n`：表示总的数据块数量

切分之后的数据块 $d$ 相对于原数据的位置记为 `tile_index`（0 ≤ tile_index < n）。对于每个数据块 $d$，会更新其 shape 和 offset 信息，并将 $d$ 作为 tile_op 的输入或输出。shape 和 offset 信息更新如下：

$$
row\_index = \left\lfloor \frac{tile\_index}{tile\_col\_num} \right\rfloor \\
col\_index = tile\_index \bmod tile\_col\_num \\
tile\_offset_0 = row\_index \times t_1 \\
tile\_offset_1 = col\_index \times t_2 \\
tile\_shape_0 = \min(x - tile\_offset_0, t_1) \\
tile\_shape_1 = \min(y - tile\_offset_1, t_2)
$$

### 3.2 控制边类型的tensor切分策略

切块的主体对象是 data tensor。一个 op 会被切分为多少个 tile_op，由 data tensor 的 shape 和 TileShape 共同决定。

对 dummy tensor 切分时，遵循以下核心原则：将 dummy tensor 尽可能均匀地切分为 $m_1, m_2, \ldots, m_n$ 共 n 个子块，与切分后的数据块 $d$ 形成一一对应关系，并作为 tile_op 的输入或输出。

假设 dummy tensor 的 shape 为 $[p_1, p_2]$，根据 $p_1$ 与 $p_2$ 大小的不同，可分为以下三种切块方式：

- **$p_1 \geq tile\_row\_num$ 且 $p_2 \geq tile\_col\_num$**：将 dummy tensor 的行切为 tile_row_num 个，列切为 tile_col_num 个。切块采用"平均分配+尾块优先"策略：

**切分思路**：将 $p_1$ 行尽可能均匀地分配给 tile_row_num 个块，将 $p_2$ 列尽可能均匀地分配给 tile_col_num 个块。若不能整除，余数作为尾行/尾列分配给前面的块，使前几个块的 size 比后面的块多1。

**计算步骤**：

1. 计算基准大小和尾行/尾列数：
   - 行方向基准：$base\_row = p_1 / tile\_row\_num$（向下取整）
   - 行方向尾行数：$base\_tail\_row = p_1 \bmod tile\_row\_num$
   - 列方向基准：$base\_col = p_2 / tile\_col\_num$（向下取整）
   - 列方向尾列数：$base\_tail\_col = p_2 \bmod tile\_col\_num$

2. 计算每个块的坐标：
   - $row\_index = tile\_index / tile\_col\_num$（向下取整）
   - $col\_index = tile\_index \bmod tile\_col\_num$

3. 计算每个块的 shape：
   - 若 $row\_index < base\_tail\_row$（属于前 $base\_tail\_row$ 个块），则 $tile\_dummy\_shape_0 = base\_row + 1$
   - 若 $row\_index \geq base\_tail\_row$（属于后面的块），则 $tile\_dummy\_shape_0 = base\_row$
   - 列方向同理

4. 计算每个块的 offset（累加前面所有块的大小）：
   - 若 $row\_index < base\_tail\_row$，前 $row\_index$ 个块的 size 都是 $base\_row + 1$，因此 $offset = row\_index \times (base\_row + 1)$
   - 若 $row\_index \geq base\_tail\_row$，前 $base\_tail\_row$ 个块的 size 是 $base\_row + 1$，后面 $(row\_index - base\_tail\_row)$ 个块的 size 是 $base\_row$，因此 $offset = base\_tail\_row \times (base\_row + 1) + (row\_index - base\_tail\_row) \times base\_row$
   - 列方向同理

**示例**：假设 dummy tensor shape 为 $[p_1, p_2] = [10, 7]$，需要切分为 $tile\_row\_num = 3$ 行块、$tile\_col\_num = 2$ 列块。则：
- 行方向：$base\_row = 10 / 3 = 3$, $base\_tail\_row = 10 \bmod 3 = 1$（有1个尾行）
- 列方向：$base\_col = 7 / 2 = 3$, $base\_tail\_col = 7 \bmod 2 = 1$（有1个尾列）

切块矩阵（行按 4+3+3 分配，列按 4+3 分配）：

| tile_index | row_index | col_index | shape | offset |
|------------|-----------|-----------|-------|--------|
| 0 | 0 | 0 | [4, 4] | [0, 0] |
| 1 | 0 | 1 | [4, 3] | [0, 4] |
| 2 | 1 | 0 | [3, 4] | [4, 0] |
| 3 | 1 | 1 | [3, 3] | [4, 4] |
| 4 | 2 | 0 | [3, 4] | [7, 0] |
| 5 | 2 | 1 | [3, 3] | [7, 4] |

说明：前 base_tail_row=1 个行块多1行（shape_0=4），前 base_tail_col=1 个列块多1列（shape_1=4）。

- **$p_1 \times p_2 \geq n$**：dummy tensor 的元素总数足够但无法进行 2D 切分（即 $p_1 < tile\_row\_num$ 或 $p_2 < tile\_col\_num$），此时采用"逐元素分配+尾块兜底"策略：

**切分思路**：将 dummy tensor 按元素位置顺序切分，前 $n-1$ 块各取一个 [1, 1] 元素，最后一块（第 $n-1$ 块）取剩余所有元素，确保每个 tile_op 都有对应的 dummy tensor 块。

**计算步骤**：

1. 计算每个块的起始位置：
   - $tile\_row\_index = tile\_index / p_2$（向下取整）
   - $tile\_col\_index = tile\_index \bmod p_2$
   - offset 直接取该元素的位置：$tile\_dummy\_offset = [tile\_row\_index, tile\_col\_index]$

2. 计算每个块的 shape：
   - 若 $tile\_index \neq n-1$（前 $n-1$ 块）：shape = [1, 1]，只取一个元素
   - 若 $tile\_index = n-1$（最后一块）：取从当前位置到 dummy tensor 末尾的所有剩余元素，shape 分三种情况：
     - 尾块在最后一行同一行内：shape = [1, $tile\_col\_end - tile\_col\_index + 1$]
     - 尾块在列的开头（$tile\_col\_index = 0$）：shape = [$tile\_row\_end - tile\_row\_index + 1$, $p_2$]
     - 其他情况：shape = [1, 1]

其中 $tile\_end = p_1 \times p_2 - 1$，$tile\_row\_end = tile\_end / p_2$，$tile\_col\_end = tile\_end \bmod p_2$。

**示例**：假设 dummy tensor shape 为 $[p_1, p_2] = [2, 4]$（共8个元素），需要切分 $n=5$ 块（对应 data tensor 的5个切块）。则：

切块结果（前4块各取1元素，第4块取剩余4个元素）：

| tile_index | tile_row_index | tile_col_index | shape | offset | 说明 |
|------------|----------------|----------------|-------|--------|------|
| 0 | 0 | 0 | [1, 1] | [0, 0] | 第1个元素 |
| 1 | 0 | 1 | [1, 1] | [0, 1] | 第2个元素 |
| 2 | 0 | 2 | [1, 1] | [0, 2] | 第3个元素 |
| 3 | 0 | 3 | [1, 1] | [0, 3] | 第4个元素 |
| 4 | 1 | 0 | [2, 4] | [1, 0] | 尾块从[1,0]到末尾[1,3]，覆盖剩余4元素 |

说明：tile_end=7, tile_row_end=1, tile_col_end=3。尾块在 tile_col_index=0 处，按第三种情况取 [2, 4]。

- **$p_1 \times p_2 < n$**：因为 dummy tensor 中的元素个数小于 data tensor 的切块个数 $n$，因此不对 dummy tensor 进行切块，所有的 tile_op 共用 dummy tensor。在这种情况下，依赖该 op 的算子，需要等所有其展开的 tile_op 执行完成后方可执行。

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

1. **切块维度仅支持 2 维**

当前仅支持 2 维数据的拷贝，因此仅支持对 2 维数据进行切块。

2. **shmem_signal 写信号量时需确保对应的数据已经发送完成**

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

3. **shmem_wait_until 和 shmem_signal 需要设置相同的切块大小**

两者操作同一个 shared memory tensor，需要精确匹配。如果切块大小不同，signal 写入的 tile 块和 wait_until 等待的 tile 块不对应，会引起等信号量超时错误。

4. **shmem_get 读取数据前需确保数据已经写入完成**

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
