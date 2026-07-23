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

把 GM 中一块数据按**绝对元素坐标**搬入 L1/UB Tile，是 kernel 取输入数据参与计算的基础接口。GM 数据支持在行（Row）、列（Col）维度上偏移任意元素个数，支持连续搬运和高维切分搬运两种模式。

如果希望按"第几块 tile"来定位（自动乘以 tile 形状），需要使用 [`pypto_pro.language.load_tile`](load_tile.md)。

## 函数原型

```python
pypto_pro.language.load(dst_tile, src_tensor, offsets, *, order=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 只能是 L1、UB Tile，搬入目的地 |
| `src_tensor` | 输入 | Tensor 类型，来自 GM 的源数据 |
| `offsets` | 输入 | GM 地址偏移，标记在每根轴上的初始偏移，单位元素个数 |
| `order` | 输入 | 可选，Tile 维度在 GlobalTensor 维度中对应哪几根轴；元素为 Tensor 绝对轴索引，升序表示不转置，反序表示转置；省略时默认 `[ndim-2, ndim-1]`（不转置） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 数据类型：b8、b16、b32、b64<br>尾块处理：<br>• 可通过 set_validshape 设置尾块大小，Tile shape 需要 32 字节对齐，不对齐报错<br>• valid_shape 可以不对齐<br>• set_validshape 需要 compact=1，compact 不等于 1 且 validshape 不等于 shape 时需要报错<br>• 支持设定 padding 值<br>地址配置：<br>• Tile 的类型只能是 L1、UB，Cube 侧非 L1 报错<br>• Vector 侧非 UB 报错<br>• L1、UB buffer 首地址必须 32 字节对齐，不对齐编译报错 |
| `src_tensor` | 输入 | 数据类型：b8、b16、b32、b64<br>layout：支持 `ND`、`DN`、`NZ`<br>stride：支持配置 Stride，stride 维度需要等于 tensor 维度数，默认不配置时是尾轴 stride=1 的连续场景 |
| `offsets` | 输入 | 单位元素个数，大小不超过对应维度的 shape，不支持负数索引 |
| `order` | 输入 | 只支持配置 tensor 维度范围内的 dim，只支持二维数组配置，其余配置报错<br>用于高维 tensor 中指定 tile 对应哪几个维度；order 中轴索引的顺序决定是否转置：升序不转置（ND 行主序），反序转置（DN 列主序），需要配合 Tensor 的 layout 以及 Tile 的 shape 和 stride 填写<br>省略时默认取 tensor 的最后两维 `[ndim-2, ndim-1]`（不转置） |

## 流水类型

MTE2（GM → L1/UB 的搬入流水）。

## 调用示例

下面是一个完整 kernel：从 GM 载入两个 64×64 的输入到 UB，相加后写回 GM。`pypto_pro.language.load` 负责把 GM 数据搬入 UB tile。vector kernel 开 `auto_mutex`，同步由 `make_tile_group` 自动管理。

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

其他典型用法（节选）：

```python
# 循环按行块载入（matmul 取左矩阵）
for i in pl.range(0, m_dim, 64):
    pl.load(tile_a, a, [i, 0])

# 列主序载入（DN 布局，flash attention 把 K 载入 L1）
pl.load(k_mat, k, [skv_off, 0], order=[1, 0])

# 高维 tensor + order：指定 tile 对应 tensor 的哪几根轴
pl.load(q_buf, q, [b_idx, 0, n_idx, 0], order=[1, 3])
```

## 高级用法

以下介绍 `pypto_pro.language.load` 在复杂数据布局和边界处理场景中的用法。

### GM Tensor多维场景

Tensor的维度可能有多维，而Tile只有两维，那么Tensor和Tile之间的拷贝需要指定Tile的两维是Tensor中的哪两维。

这由 `pl.load` / `pl.load_tile` 的 `order` 参数（以及 `store` / `store_tile` 的 `tile_dims` 参数）控制：

- **`offsets` 的长度 == Tensor 的维数**：每个 Tensor 轴给一个偏移。
- **`order`**：一个长度为 2 的列表，元素为 Tensor 绝对轴索引，指出 Tile 的两维分别落在 Tensor 的哪两个轴上。order 中轴索引的顺序决定是否转置：升序不转置，反序转置。
  未被 `order` 选中的轴，用 `offsets` 里对应的值**定死为一个下标**（相当于在那一维上
  切一刀）。
- **默认值**：不填 `order` 时，取 Tensor 的**最后两维**，即
  `order = [ndim-2, ndim-1]`（不转置）（见 `block_ops.py:_ir_load_tile`）。

例：一个 4 维 Tensor `q: [B, N, Sq, D]`，Tile 是 `[TS, TD]`，想让 Tile 覆盖
`(Sq, D)` 这两维、并在 `b`、`n` 上各定死一个下标：

```python
q: pl.Tensor[[B, N, Sq, D], pl.DT_FP16]
q_tile = pl.make_tile(pl.TileType(shape=[TS, TD], dtype=pl.DT_FP16,
                                  target_memory=pl.MemorySpace.Mat), addr=0x0, size=...)
# 最后两维恰好是 (Sq, D)，默认 order=[2,3]，可省略：
pl.load(q_tile, q, [b, n, sq_off, 0])
```

若想让 Tile 覆盖**非相邻**或**非末尾**的两维（例如 `(N, D)`，即轴 1 和轴 3），显式给
`order`（真实例子来自 `fa/test_fa_bsnd_dn.py`，Tensor `[B, S, N, D]`）：

```python
# Tile 的两维 = Tensor 的轴 1 与轴 3；轴 0(b_idx)、轴 2(n_idx) 被定死
pl.load_tile(k_mat_buf[buf_idx], k, [b_idx, ki, n_idx, 0], order=[1, 3])
```

> `order` 是**编译期常量列表**（它驱动 codegen 的 tensor-view stride），不能是运行时
> 变量。

### Tensor维度变化

有时 kernel 拿到的 GM 数据布局和你想用的 Tile 布局对不上（维数不同、或想换一种 shape /
stride 去看同一块内存）。这时**不改动底层数据**，只重建一个新的 Tensor 视图。

- **通过 `pl.Ptr` 重新生成**：kernel 参数声明成裸指针 `pl.Ptr[dtype]`，在函数体里用
  `pl.make_tensor(ptr, shape, stride)` 按需要的 shape/stride 构造视图。这是"动态 rank"
  kernel 的基础 —— 维数、shape 都来自运行时的 tiling 数据，而非签名。

```python
@pl.jit()
def k(q: pl.Ptr[pl.DT_FP16], tiling: OpTiling):
    # 行优先 => stride = [n*d, d, 1]；shape/stride 全部来自 tiling
    tensor_q = pl.make_tensor(q, [tiling.sq, tiling.n, tiling.d],
                              [tiling.n * tiling.d, tiling.d, 1])
    # 之后 tensor_q 的用法与 pl.Tensor 参数完全相同
```

- **也可以从已有的 `pl.Tensor` 重建**：`pl.make_tensor` 的第一个参数可以是一个已存在的
  `pl.Tensor`，此时新视图复用它的底层指针，只换 shape/stride（可选换 dtype）。这正是下面
  "合轴场景"用到的能力。

> 重建视图只改变"怎么看这块内存"，不搬运、不拷贝数据。给出的 stride 必须与底层内存的真实
> 排布一致，否则读到的就是错位的数据。

### 尾块需要padding场景

当 GM Tensor 的 shape 不能被 Tile 整除时，边界上会出现比 Tile 小的**尾块**。Tile 的
物理大小固定，但每个尾块的**有效区域**不同。处理方式：

1. Tile 的 `valid_shape` 声明成动态（`[-1, -1]`），load 前用 `pl.set_validshape` 写入这一
   块真实的有效行/列 —— load 会**只搬有效区**，不会越界读 GM，也不会把 padding 写回。
2. 若后续算子会读到有效区**之外**（归约 / matmul 会整块读入），再配合 `pad` +
   `pl.fillpad` 把无效区填成安全值（求和填 `zero`、`row_max`/softmax 填 `min`、`row_min`
   填 `max`）。

```python
# tile 物理 64x128，有效行/列运行时决定
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

尾块的 `valid_shape` / `set_validshape` / `pad` / `fillpad` / `compact` 如何协同，详见
尾块 Tile 文档。

### Cube侧转置场景

**总结：**

- L0A固定为`shape=[M, K]`、`layout=pl.NZ`；
- L0B固定为`shape=[K, N]`、`layout=pl.ZN`；
- L1（`Mat`）的shape与L0A或L0B一致；
- 如果需要转置，L1（`Mat`）上固定为`pl.ZN`
  - 如果是 GM->L1，load上固定设置`order=[1, 0]`（反序转置）；
  - 如果是 UB->L1，无需设置`order`（默认升序不转置）；
- 如果不需要转置，L1（`Mat`）上固定为`pl.NZ`
  - 如果是 GM->L1，load上默认`order`升序不转置（可省略）；
  - 如果是 UB->L1，无需设置`order`（默认升序不转置）；

matmul 的左/右矩阵在从 L1（`Mat`）搬到 L0（`Left`/`Right`）时可能需要转置。下面先说明硬件约束，再给出前端的写法。

#### 背景：硬件约束

- 左矩阵进入 L0A（`Left`），物理形态固定为 `shape=[M, K]`、`layout=pl.NZ`；右矩阵进入 L0B（`Right`），固定为 `shape=[K, N]`、`layout=pl.ZN`。
- `Mat` 搬到 L0（`pl.move`，底层是 CANN 的 TMOV）**要求源和目的的物理 `[Rows, Cols]` 完全一致**；转置只能借助 `NZ`/`ZN` 的 fractal 差异实现，不能改变维度本身。
- 物理上 `Mat [N, K] NZ` 与 `Mat [K, N] ZN` 是**同一份数据（同一段 bytes）**，只是标注方式不同。

因此"逻辑转置"本质上要求 `Mat` 声明成与 `Left`/`Right` 一致的物理形态，并通过 `layout` 与 `order` 表达转置。

#### 写法：显式 `order`

`Mat` 的 `shape` 与对应的 `Left`/`Right` **保持一致**，靠 `layout` 与 `order` 表达转置。

- **不转置**：左矩阵 `Mat` 的 `layout` 与 `Left` 相同（`pl.NZ`）；右矩阵 `Mat` 的 `layout` 与 `Right` 相反（`pl.NZ`）；`pl.load` 正常调用。
- **转置**：左矩阵 `Mat` 的 `layout` 与 `Left` 相反（`pl.ZN`）；右矩阵 `Mat` 的 `layout` 与 `Right` 相同（`pl.ZN`）；`pl.load` 增加 `order=[1, 0]`（反序转置），框架会把对应的 `GlobalTensor` 标成 `DN` 并对调 stride。

| 矩阵 | 是否转置 | GM tensor shape | Mat 声明 | load |
| --- | --- | --- | --- | --- |
| 左 A[M,K] | 否 | `[M, K]` | `shape=[M, K], layout=pl.NZ` | 正常 |
| 左 A[M,K] | 是 | `[K, M]` | `shape=[M, K], layout=pl.ZN` | `order=[1, 0]` |
| 右 B[K,N] | 否 | `[K, N]` | `shape=[K, N], layout=pl.NZ` | 正常 |
| 右 B[K,N] | 是 | `[N, K]` | `shape=[K, N], layout=pl.ZN` | `order=[1, 0]` |

- **左矩阵不转置，从 GM 拷贝到 L1**

```python
gm = pl.make_tensor(ptr, [M, K])
mat_type = pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm)                          # 正常 load
left = pl.make_tile(pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Left), addr=0x0, size=32768)
pl.move(left, mat_0)                        # Mat[M,K] -> Left[M,K]，同形，不转置
```

- **左矩阵转置，从 GM 拷贝到 L1**

```python
gm = pl.make_tensor(ptr, [K, M])            # 数据是 A^T=[K, M]
mat_type = pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm, order=[1, 0])       # 转置 load（order 反序）；框架标 DN 并对调 stride
left = pl.make_tile(pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Left), addr=0x0, size=32768)
pl.move(left, mat_0)                        # Mat[M,K] -> Left[M,K]，同形，move 时 fractal 翻转实现转置
```

- **右矩阵不转置，从 GM 拷贝到 L1**

```python
gm = pl.make_tensor(ptr, [K, N])
mat_type = pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm)                          # 正常 load
right = pl.make_tile(pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Right), addr=0x0, size=32768)
pl.move(right, mat_0)                        # Mat[K,N] -> Right[K,N]，同形，不转置
```

- **右矩阵转置，从 GM 拷贝到 L1**

```python
gm = pl.make_tensor(ptr, [N, K])            # 数据是 B^T=[N, K]
mat_type = pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm, order=[1, 0])       # 转置 load（order 反序）；框架标 DN 并对调 stride
right = pl.make_tile(pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Right), addr=0x0, size=32768)
pl.move(right, mat_0)                        # Mat[K,N] -> Right[K,N]，同形，move 时 fractal 翻转实现转置
```

#### 数据来自 UB（片上产生）时的转置

当左/右矩阵不是从 GM 直接 load，而是由 Vector 侧产生、经 `pl.move`(ND→NZ) + `pl.insert` 写入 L1 时，`Mat` 的 `layout` 与 `shape` 同样需要与 `Left`/`Right` 保持一致：

- **不转置**：`Mat` 的 `layout=pl.NZ`，`shape` 与 `Left`/`Right` 一致；`insert` 后 `move` 同形不转置。
- **转置**：`Mat` 的 `layout=pl.ZN`，`shape` 与 `Left`/`Right` 一致；`insert` 写入 ZN 物理布局（与源 NZ tile 物理等价），`move` 时 SFractal 不匹配触发硬件转置。

> 注意：`pl.insert` 写入 `ZN` Mat 时，源 tile 的 NZ `[R, C]` 物理布局等价于 `ZN [C, R]`，因此源 tile shape 与 Mat shape 互为转置时物理数据恰好匹配。需确保 `insert` 的逻辑边界检查通过（`indexCol + validCol ≤ Mat.Cols`）。

| 矩阵 | 是否转置 | Vector 产出数据 | Mat 声明 | 对应用例 |
| --- | --- | --- | --- | --- |
| 左 A[M,K] | 否 | `[M, K]` | `shape=[M, K], layout=pl.NZ` | `test_insert_left.py` |
| 左 A[M,K] | 是 | `[K, M]` (Aᵀ) | `shape=[M, K], layout=pl.ZN` | `test_insert_transpose_left.py` |
| 右 B[K,N] | 否 | `[K, N]` | `shape=[K, N], layout=pl.NZ` | `test_insert_right.py` |
| 右 B[K,N] | 是 | `[N, K]` (Bᵀ) | `shape=[K, N], layout=pl.ZN` | `test_insert_transpose_right.py` |

> 约束：`pl.insert` 要求 `tile.dim0 == mat.dim0`（物理行匹配）；分 subcore 带列偏移插入时，被转置的矩阵通常需为方阵（见 FA 场景）。

### GM Tensor合轴场景

**"合轴"指把 GM Tensor 的两个维度当成一维来处理**（two dimensions treated as one）。

与前面"多维场景"用 `order` 在多维里挑两维不同，合轴是把**相邻的两维合并**，让
Tensor 的有效维数降下来，从而更自然地对上二维 Tile。典型用途：

- FA 的 **TND / BSND** 布局：把 batch 轴 `B` 和序列轴 `S` 合成一个"总 token"轴 `B*S`
  （即 TND 里的 `T`），一次 load 就能跨 batch 连续取若干行。
- 动态 rank kernel：把 rank 2..4 的前若干维乘到一起，折叠成统一的二维 `[M, N]`。

#### 合轴的前提：两维在内存里必须连续

只有当被合并的两维在内存中**首尾相接、没有间隔**时才能合轴。对行优先（ND）的
`[d0, d1, d2]`，其 stride 是 `[d1*d2, d2, 1]`：

- 合并 `d0` 与 `d1` → 新轴大小 `d0*d1`，stride 取里层的 `d2`。成立条件：
  `stride(d0) == d1 * stride(d1)`，即 `d1*d2 == d1 * d2` ✔。
- 如果 `d0` 与 `d1` 之间有 padding（`stride(d0) > d1 * stride(d1)`），**不能**合轴 ——
  合并后会把那段 padding 也当成数据读进来。

#### 怎么表达：用 `pl.make_tensor` 重建一个"降维"视图

合轴不是 `load` 的某个开关，而是先用 `pl.make_tensor` 把两维乘到一起、重建一个低一维的
Tensor 视图（复用同一块底层内存，见上文"Tensor维度变化"），再正常 load。

**例 1：把 `[B, S, D]` 合成 `[B*S, D]`（静态 shape）**

```python
# 原始 GM：q 是 [B, S, D]，行优先连续，stride = [S*D, D, 1]
q: pl.Tensor[[B, S, D], pl.DT_FP16]

# 合轴：B、S 合并成一维 B*S，stride 取里层的 D
q_merged = pl.make_tensor(q, [B * S, D], [D, 1])   # 复用 q 的指针，不搬数据

# 现在 q_merged 是二维 [B*S, D]，直接按 tile 索引 load
tile = pl.make_tile(pl.TileType(shape=[TS, D], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Vec), addr=0x0, size=TS*D*2)
pl.load_tile(tile, q_merged, [t, 0])   # 第 t 块 = 合并轴上的第 [t*TS : (t+1)*TS] 行
```

合并轴上的偏移换算：要读原始的第 `b` 个 batch、第 `s` 行，合并后的行号是
`b * S + s`（行优先的自然展开）。

**例 2：动态 rank —— 把前若干维乘到一起折叠成 `[M, N]`**

来自 `element_wise/test_eltwise_dynamic_rank.py`：kernel 收裸指针 + tiling，把 rank 2..4 的
shape 折叠成二维再处理。`N` 是最内维，`M` 是其余维的乘积（合轴）：

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

因为逐元素算子只关心"扁平的元素顺序"，把 `[2, 4, 256, 256]`、`[8, 256, 256]`、
`[512, 512]` 统统合轴成 `[M, N]` 后，**同一份 kernel** 就能处理任意 rank。

**例 3：TND / BSND 布局的合轴**

FlashAttention 的 TND 布局本身就是把 batch 与序列合成一个"总 token"轴 `T = ΣS_i`。若拿到
的是 BSND（`[B, S, N, D]`）且各 batch 的 `S` 相同、内存连续，可临时合轴成 TND 视角：

```python
# BSND -> 把 B、S 合成 T = B*S（N、D 保留），stride 取里层
q_tnd = pl.make_tensor(q, [B * S, N, D], [N * D, D, 1])
# 再用 order 在 [T, N, D] 里挑 (T, D) 两维
pl.load_tile(q_tile, q_tnd, [t_off, n_idx, 0], order=[0, 2])
```

#### 小结

| 步骤 | 做法 |
|------|------|
| 1. 确认可合轴 | 被合并的两维在内存里连续（无 padding）。 |
| 2. 重建视图   | `pl.make_tensor(src, 合并后的shape, 合并后的stride)`，stride 取里层维的 stride。 |
| 3. 正常 load  | 合并后的 Tensor 维数更低，按普通二维 / 多维场景 load 即可。 |
| 4. 偏移换算   | 合并轴的行号 = `外层下标 * 里层大小 + 里层下标`。 |

> **精度自查**：合轴是把两维当一维搬运的优化，一旦被合并的两维实际不连续、或 stride 写
> 错，就会读到错位 / padding 数据。怀疑精度问题时，可先按逐维不合轴的写法对比验证（参见
> `docs/zh/trouble_shooting/machine.md` 中"关闭合轴特性"的排查建议）。
