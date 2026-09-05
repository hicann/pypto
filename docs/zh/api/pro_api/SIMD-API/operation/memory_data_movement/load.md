# pypto_pro.language.load

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

把GM中的数据按绝对元素坐标搬入L1 Buffer或UB中的Tile，是Kernel获取输入数据的基础接口。支持在行、列维度上按任意元素数偏移，也支持从高维Tensor中选择指定维度搬运。

如果希望按“第几块Tile”来定位（自动乘以Tile形状），需要使用[pypto_pro.language.load_tile](load_tile.md)。

## 函数原型

```python
pypto_pro.language.load(
    dst_tile: Tile,
    src_tensor: Tensor,
    offsets: Offset,
    *,
    order: Optional[List[int]] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tile | 输出 | 目的操作数，Tile类型，存储空间为L1 Buffer或UB，首地址必须按32字节对齐。支持DT_FP4E2M1、DT_FP4E1M2、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32。可通过set_validshape设置尾块有效形状；涉及紧凑排列、分形转换或Cube计算时，须按数据路径设置compact。 |
| src_tensor | 输入 | 源操作数，Tensor类型，存储空间为GM，支持的数据类型与dst_tile一致，排布支持ND、DN和NZ。 |
| offsets | 输入 | 源Tensor的元素偏移，List[int或Scalar]类型，长度须与Tensor维数相同。各项为非负整数或运行时整数表达式，访问范围不得越过Tensor边界。 |
| order | 输入 | 维度映射，List[int]类型，可选。表示Tile两个维度分别对应源Tensor的哪两个维度；两个轴索引必须互不重复且位于Tensor维度范围内，升序表示不转置，降序表示转置。 |

## 约束说明

当src_tensor声明为pypto_pro.language.NZ时，其物理排布、分形轴和完整Tensor shape约束见[TensorLayout](../../basic_data_structures/TensorLayout.md#tensor布局)。load还需满足以下NZ搬运约束：

- 仅支持GM NZ到NZ Tile的同布局搬运，目标Tile位于L1 Buffer或UB；order省略或指定Tensor最后两轴的正序。
- Tile形状和有效M、N须满足M按16、N按Tensor数据类型对应的C0对齐，N方向偏移也须按C0对齐。
- 高维offset的前导项选择batch，最后两项为M、N方向的逻辑元素坐标。

当前DT_FP8E8M0 Tensor搬入fractal=32的ZZ或NN排布L1 Buffer Tile，仅支持作为matmul_mx或matmul_mx_acc的缩放因子搬运。普通E8M0数据不支持使用该目标组合；满足该组合的load会按MX缩放因子解释，并要求源Tensor的最后一轴是长度为2的物理phase轴。

开启auto_mutex时，若连续两次pypto_pro.language.load向同一个UB或L1 Buffer Tile地址搬运数据，并且前一次搬入的数据没有被读取，则必须在两次load之间调用pypto_pro.language.system.bar_mte2()，再复用该地址。

关于复用Tile地址的完整同步规则，请参考下文“Tile地址复用与流水同步”。

## 返回值说明

无。

## 调用示例

### 从GM搬入UB并完成逐元素加法

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

### 高维Tensor与转置搬运

```python
# 循环按行块载入（matmul 取左矩阵）
for i in pl.range(0, m_dim, 64):
    pl.load(tile_a, a, [i, 0])

# 列主序载入（DN 布局，flash attention 把 K 载入 L1）
pl.load(k_mat, k, [skv_off, 0], order=[1, 0])

# 高维Tensor + order：指定Tile对应Tensor的轴
pl.load(q_buf, q, [b_idx, 0, n_idx, 0], order=[1, 3])
```

### Tile地址复用与流水同步

一次pypto_pro.language.load会通过MTE2向目标Tile写入数据。开启auto_mutex并复用同一个UB或L1 Buffer Tile地址时，可按以下步骤判断是否需要手动同步。

#### 判断步骤

1. 先确认相邻两次load的目标是否为**同一个Tile（同一个物理地址）**。若目标地址不同，例如双缓冲或多缓冲轮转到不同Tile，则不适用本节规则。
2. 若目标地址相同，检查第二次load前是否已有操作读取前一次搬入的Tile：
   - **已读取**：例如UB上由Vector流水执行的计算，或通过MTE1将数据从L1 Buffer搬入L0A Buffer或L0B Buffer。数据依赖会建立MTE2到Vector或MTE2到MTE1的同步，无需额外调用pypto_pro.language.system.bar_mte2()。
   - **未读取**：在两次load之间调用pypto_pro.language.system.bar_mte2()，使前一次MTE2搬运完成后再复用该地址，避免两次写入同一地址的次序或覆盖风险（WAW，Write-After-Write）。与该Tile无关的计算或搬运不属于“读取”。

#### auto_mutex与bar_mte2的边界

- auto_mutex只能处理具有相同mutex ID的Tile在不同Pipeline之间的同步。同一个Tile被反复写入属于同一条Pipeline，框架不会自动补充同步，需要用户按上述步骤处理。
- pypto_pro.language.system.bar_mte2()只约束MTE2流水，不会保留随后被覆盖的旧数据。若后续仍需要前一次搬入的数据，必须在复用该地址前先读取或复制该数据。

#### 复用地址前未读取前一次结果

##### 未同步的错误写法

```python
# 同一Tile被连续写入，中间没有读取操作或bar_mte2
pl.load(in_tile, src_a, [0, 0])
pl.load(in_tile, src_b, [64, 0])    # 前一次MTE2写入可能尚未完成

# 部分迭代不读取in_tile，但下一轮仍会复用该地址
for x in pl.range(0, m_dim, 64):
    pl.load(in_tile, src, [x, 0])
    if x % 2 == 0:
        pl.add(out_tile, in_tile, in_tile)   # 偶数轮读取in_tile
    # 奇数轮没有读取in_tile，也没有bar_mte2
```

##### 使用bar_mte2同步地址复用

```python
# 复用同一Tile前先完成前一次MTE2搬运
pl.load(in_tile, src_a, [0, 0])
pl.system.bar_mte2()
pl.load(in_tile, src_b, [64, 0])

# 没有读取in_tile的分支显式约束MTE2
for x in pl.range(0, m_dim, 64):
    pl.load(in_tile, src, [x, 0])
    if x % 2 == 0:
        pl.add(out_tile, in_tile, in_tile)   # 数据依赖建立MTE2→V同步
    else:
        pl.system.bar_mte2()                  # 下一轮复用in_tile前完成MTE2
```

循环按照load、Vector计算、store的顺序执行时，每一轮都会在下一次load前读取该Tile，因此不需要调用bar_mte2。仅当某个分支跳过本轮读取且后续仍要复用同一Tile时，才需要在该分支中调用bar_mte2。

#### 复用地址前已读取前一次结果

##### 包含冗余同步的写法

```python
# 后续Vector计算读取in_tile，无需额外调用bar_mte2
pl.load(in_tile, src_a, [0, 0])
pl.system.bar_mte2()
pl.add(out_tile, in_tile, in_tile)
pl.load(in_tile, src_b, [64, 0])
```

##### 通过数据依赖建立同步

```python
# Vector计算读取in_tile，因此可在下一次搬运时复用该地址
pl.load(in_tile, src_a, [0, 0])
pl.add(out_tile, in_tile, in_tile)   # 数据依赖建立MTE2→V同步
pl.load(in_tile, src_b, [64, 0])

# 每一轮都会读取in_tile，下一轮可继续复用它
for x in pl.range(0, m_dim, 64):
    pl.load(in_tile, src, [x, 0])
    pl.add(out_tile, in_tile, in_tile)   # 读取in_tile
    pl.store(out, out_tile, [x, 0])
```

### GM Tensor多维场景

当Tensor多于二维时，需要通过order参数指定Tile的两个维度分别对应Tensor的哪两个维度。

pypto_pro.language.load和pypto_pro.language.load_tile的order参数，以及pypto_pro.language.store和pypto_pro.language.store_tile的order参数，采用相同的维度映射规则：

- offsets的长度必须与Tensor的维数相同，每个元素表示对应维度的偏移。
- order是一个长度为2的列表，列表元素为Tensor的绝对维度索引，依次指定Tile的两个维度对应的Tensor维度。order中的维度索引升序排列时不转置，反序排列时执行转置。未被order选中的维度由offsets中的对应值固定为一个下标。
- 不设置order时，Tile对应Tensor的最后两个维度，即order = [ndim - 2, ndim - 1]，不执行转置。

例如，对于形状为[B, N, Sq, D]的四维Tensor q和形状为[TS, TD]的Tile，可以使Tile覆盖Sq、D两个维度，并在B、N维分别固定一个下标。

#### 选择末尾相邻维度

```python
q: pl.Tensor[[B, N, Sq, D], pl.DT_FP16]
q_tile = pl.make_tile(pl.TileType(shape=[TS, TD], dtype=pl.DT_FP16,
                                  target_memory=pl.MemorySpace.Mat), addr=0x0, size=...)
# 最后两维恰好是 (Sq, D)，默认 order=[2,3]，可省略：
pl.load(q_tile, q, [b, n, sq_off, 0])
```

若要使Tile覆盖非相邻或非末尾的两个维度，例如形状为[B, S, N, D]的Tensor中的S、D维，需要显式设置order=[1, 3]。

#### 选择非相邻维度

```python
# Tile 的两维 = Tensor 的轴 1 与轴 3；轴 0(b_idx)、轴 2(n_idx) 被定死
pl.load_tile(k_mat_buf[buf_idx], k, [b_idx, ki, n_idx, 0], order=[1, 3])
```

order必须是编译期常量列表，不能使用运行时变量。

### 尾块需要padding场景

当GM Tensor的形状不能被Tile整除时，边界上会出现有效区域小于Tile物理形状的尾块。处理方式如下：

1. 将Tile的valid_shape声明为动态值[-1, -1]，并在load前调用pypto_pro.language.set_validshape设置当前尾块的有效行数和列数。load仅搬运有效区域，不会越界读取GM。
2. 若后续算子会读取有效区域之外的数据，例如归约或矩阵乘操作会读取整个Tile，则需要设置pad属性并调用pypto_pro.language.fillpad，将无效区域填充为对应计算的安全值。求和操作填充zero，row_max或softmax操作填充min，row_min操作填充max。

#### 为尾块填充安全值

```python
# Tile 物理 64x128，有效行/列运行时决定
tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
tile = pl.make_tile(tile_type, addr=0x0, size=64*128*2)
with pl.section_vector():
    # 每个尾块先写有效形状，再 load —— 只搬 rows x cols，不越界
    pl.set_validshape(tile, [rows, cols])
    pl.load(tile, a, [row_off, col_off])
    ...
    pl.store(out, tile, [row_off, col_off])   # 只写回有效区
```

valid_shape、pypto_pro.language.set_validshape、pad、pypto_pro.language.fillpad和pypto_pro.language.compact的配合方式，详见[尾块处理](../../../../../guide/programming_guide/pro/development/tile_based_python_programming/tail_block_handling.md)。

### Cube侧转置场景

matmul的左矩阵或右矩阵从L1 Buffer搬入L0A Buffer或L0B Buffer时，可能需要执行转置。相关硬件约束和接口写法如下。

#### 背景：硬件约束

- 左矩阵进入L0A Buffer，物理形态固定为shape=[M, K]、layout=pypto_pro.language.NZ；右矩阵进入L0B Buffer，固定为shape=[K, N]、layout=pypto_pro.language.ZN。
- 从L1 Buffer搬入L0A Buffer或L0B Buffer时，源Tile和目标Tile的物理[Rows, Cols]必须一致。转置通过NZ和ZN的分形差异实现，不能改变物理维度。
- 物理上，[N, K] NZ排布的L1 Buffer Tile与[K, N] ZN排布的L1 Buffer Tile表示同一段数据，只是排布标注不同。

因此，逻辑转置要求L1 Buffer Tile声明为与L0A Buffer或L0B Buffer一致的物理形态，并通过layout与order表达转置。

#### 写法：显式order

L1 Buffer Tile的shape与对应的L0A Buffer或L0B Buffer Tile保持一致，通过layout与order表达转置。

- **不转置**：左矩阵L1 Buffer Tile的layout与L0A Buffer Tile相同，使用pypto_pro.language.NZ；右矩阵L1 Buffer Tile使用pypto_pro.language.NZ。
- **转置**：左矩阵或右矩阵的L1 Buffer Tile使用pypto_pro.language.ZN，pypto_pro.language.load传入order=[1, 0]，框架会按DN排布搬运并交换stride。

| 矩阵 | 是否转置 | GM Tensor shape | L1 Buffer Tile声明 | load |
| --- | --- | --- | --- | --- |
| 左A[M,K] | 否 | [M, K] | shape=[M, K], layout=pypto_pro.language.NZ | 正常 |
| 左A[M,K] | 是 | [K, M] | shape=[M, K], layout=pypto_pro.language.ZN | order=[1, 0] |
| 右B[K,N] | 否 | [K, N] | shape=[K, N], layout=pypto_pro.language.NZ | 正常 |
| 右B[K,N] | 是 | [N, K] | shape=[K, N], layout=pypto_pro.language.ZN | order=[1, 0] |

#### 左矩阵不转置，从GM搬入L1 Buffer

```python
gm = pl.make_tensor(ptr, [M, K])
mat_type = pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm)                          # 正常 load
left = pl.make_tile(pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Left), addr=0x0, size=32768)
pl.move(left, mat_0)                        # L1 Buffer到L0A Buffer，同形，不转置
```

#### 左矩阵转置，从GM搬入L1 Buffer

```python
gm = pl.make_tensor(ptr, [K, M])            # 数据是 A^T=[K, M]
mat_type = pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm, order=[1, 0])       # 转置 load（order 反序）；框架标 DN 并对调 stride
left = pl.make_tile(pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Left), addr=0x0, size=32768)
pl.move(left, mat_0)                        # L1 Buffer到L0A Buffer，同形，通过分形转换实现转置
```

#### 右矩阵不转置，从GM搬入L1 Buffer

```python
gm = pl.make_tensor(ptr, [K, N])
mat_type = pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm)                          # 正常 load
right = pl.make_tile(pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Right), addr=0x0, size=32768)
pl.move(right, mat_0)                        # L1 Buffer到L0B Buffer，同形，不转置
```

#### 右矩阵转置，从GM搬入L1 Buffer

```python
gm = pl.make_tensor(ptr, [N, K])            # 数据是 B^T=[N, K]
mat_type = pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm, order=[1, 0])       # 转置 load（order 反序）；框架标 DN 并对调 stride
right = pl.make_tile(pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Right), addr=0x0, size=32768)
pl.move(right, mat_0)                        # L1 Buffer到L0B Buffer，同形，通过分形转换实现转置
```

#### 数据来自UB（片上产生）时的转置

当左矩阵或右矩阵不是从GM直接load，而是由Vector侧产生、经pypto_pro.language.move完成ND到NZ转换后再由pypto_pro.language.insert写入L1 Buffer时，L1 Buffer Tile的layout与shape同样需要与L0A Buffer或L0B Buffer Tile保持一致。

- **不转置**：L1 Buffer Tile使用pypto_pro.language.NZ，shape与L0A Buffer或L0B Buffer Tile一致。
- **转置**：L1 Buffer Tile使用pypto_pro.language.ZN，shape与L0A Buffer或L0B Buffer Tile一致；insert写入ZN物理排布，move时由分形差异完成转置。

源Tile的NZ [R, C]物理排布等价于ZN [C, R]。使用pypto_pro.language.insert写入ZN排布的L1 Buffer Tile时，须确保源Tile和目标Tile的物理形状匹配，并满足insert的边界约束。

| 矩阵 | 是否转置 | Vector产出数据 | L1 Buffer Tile声明 |
| --- | --- | --- | --- |
| 左A[M,K] | 否 | [M, K] | shape=[M, K], layout=pypto_pro.language.NZ |
| 左A[M,K] | 是 | [K, M] (Aᵀ) | shape=[M, K], layout=pypto_pro.language.ZN |
| 右B[K,N] | 否 | [K, N] | shape=[K, N], layout=pypto_pro.language.NZ |
| 右B[K,N] | 是 | [N, K] (Bᵀ) | shape=[K, N], layout=pypto_pro.language.ZN |

使用pypto_pro.language.insert时，源Tile和目标Tile的物理行数必须匹配；按列偏移分块插入时，还须保证每个分块不越过目标Tile边界。

### 重建Tensor视图并合并连续维度

pypto_pro.language.make_tensor可以在不搬运数据的情况下，通过shape和stride重建GM Tensor视图，用于调整维数、形状或步长，以及合并连续维度。新Tensor视图复用底层存储，其stride必须与实际内存排布一致。

通过pypto_pro.language.Ptr重建视图时，将Kernel参数声明为pypto_pro.language.Ptr[dtype]，并在函数体内调用pypto_pro.language.make_tensor(ptr, shape, stride)。shape中的各维长度可以来自运行时TilingData，但shape列表的长度必须在编译期确定。

#### 通过Ptr创建Tensor视图

```python
@pl.jit()
def k(q: pl.Ptr[pl.DT_FP16], tiling: OpTiling):
    # 行优先 => stride = [n*d, d, 1]；shape/stride 全部来自 tiling
    tensor_q = pl.make_tensor(q, [tiling.sq, tiling.n, tiling.d],
                              [tiling.n * tiling.d, tiling.d, 1])
    # 之后 tensor_q 的用法与 pl.Tensor 参数完全相同
```

pypto_pro.language.make_tensor的第一个参数也可以是已有的pypto_pro.language.Tensor。新视图复用其底层指针，并使用新指定的shape、stride和可选dtype。

合轴是将GM Tensor中相邻且连续的两个维度合并为一个维度。与通过order从多维Tensor中选择两个维度不同，合轴会降低Tensor视图的维数，使其与二维Tile对应。典型用途如下：

- 对于FlashAttention的TND或BSND排布，将批次维B和序列维S合并为总token维B × S，即TND中的T维，使一次load可以跨批次连续读取多行。
- 对于维数不同的Host输入，将前若干连续维度合并，在Kernel内构造固定的二维[M, N]视图。

#### 合并连续维度

仅当待合并的维度在内存中连续且不存在间隔时，才能合轴。对于行优先ND排布的[d0, d1, d2]，其stride为[d1 × d2, d2, 1]：

- 合并d0与d1后，新维度大小为d0 × d1，stride取内层维度d1的stride。合并条件为stride(d0) = d1 × stride(d1)。
- 如果d0与d1之间存在Padding，即stride(d0) > d1 × stride(d1)，则不能合并，否则Padding区域会被当作有效数据读取。

#### 将[B, S, D]合并为[B × S, D]

通过pypto_pro.language.make_tensor合并连续维度并构造低一维的Tensor视图，再调用load搬入Tile。

```python
# 原始 GM：q 是 [B, S, D]，行优先连续，stride = [S*D, D, 1]
q: pl.Tensor[[B, S, D], pl.DT_FP16]

# 合轴：B、S 合并成一维 B*S，stride 取里层的 D
q_merged = pl.make_tensor(q, [B * S, D], [D, 1])   # 复用 q 的指针，不搬数据

# 现在 q_merged 是二维 [B*S, D]，直接按 Tile 索引 load
tile = pl.make_tile(pl.TileType(shape=[TS, D], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Vec), addr=0x0, size=TS*D*2)
pl.load_tile(tile, q_merged, [t, 0])   # 第 t 块 = 合并轴上的第 [t*TS : (t+1)*TS] 行
```

若要读取原始Tensor中第b个批次的第s行，合并后的行号为b × S + s。

#### 使用运行时形状合并为[M, N]

Kernel可以接收指针和TilingData，将Host侧2～4维输入的形状合并为固定的二维Tensor视图。N为最内层维度，M为其余维度的乘积。

```python
@pl.jit(auto_mutex=True)
def add_dynrank_kernel(x: pl.Ptr[pl.DT_FP16], y: pl.Ptr[pl.DT_FP16],
                       z: pl.Ptr[pl.DT_FP16], tiling: AddTiling):
    N = tiling.shape[3]
    M = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]   # 前三维合轴成 M
    tensor_x = pl.make_tensor(x, [M, N], [N, 1])              # 折叠成二维 [M, N]
    tensor_y = pl.make_tensor(y, [M, N], [N, 1])
    tensor_z = pl.make_tensor(z, [M, N], [N, 1])
    ...
    pl.load_tile(tile_a, tensor_x, [i, j])                    # 之后就是普通二维 load
```

逐元素算子只依赖连续的元素顺序，因此可以将[2, 4, 256, 256]、[8, 256, 256]和[512, 512]等不同形状合并为[M, N]，由同一个Kernel处理。Kernel内通过pypto_pro.language.make_tensor构造的视图始终为二维。

#### 将BSND排布转换为TND视图

FlashAttention的TND排布将批次与序列合并为总token维T = ΣS_i。对于形状为[B, S, N, D]的BSND Tensor，如果各批次的S相同且数据在内存中连续，可以合并B、S维，构造TND视图。

```python
# BSND -> 把 B、S 合成 T = B*S（N、D 保留），stride 取里层
q_tnd = pl.make_tensor(q, [B * S, N, D], [N * D, D, 1])
# 再用 order 在 [T, N, D] 里挑 (T, D) 两维
pl.load_tile(q_tile, q_tnd, [t_off, n_idx, 0], order=[0, 2])
```

#### 小结

| 步骤 | 做法 |
|------|------|
| 1. 确认可合轴 | 被合并的两维在内存里连续（无padding）。 |
| 2. 重建视图   | pypto_pro.language.make_tensor(src, 合并后的shape, 合并后的stride)，stride取里层维的stride。 |
| 3. 正常load  | 合并后的Tensor维数更低，按普通二维 / 多维场景load即可。 |
| 4. 偏移换算   | 合并轴的行号 = 外层下标 * 里层大小 + 里层下标。 |

> **合轴约束**：被合并的维度必须连续，stride必须与实际内存排布一致。违反上述约束会读取错误位置或Padding区域的数据。排查精度问题时，可改用未合轴的逐维Tensor视图进行对比验证。
