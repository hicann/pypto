# pypto_pro.language.TensorLayout

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

数据布局枚举，用于描述 GM Tensor 的存储形式和 Tile 的数据排列形式。

- **GM Tensor** 支持 `ND`、`NZ`，默认 `ND`。
- **Tile** 支持 `ND`、`DN`、`NZ`、`ZN`、`NN`、`ZZ`，默认值由内存空间和芯片架构决定。

## 取值

| 取值 | 数据排列 | 适用对象 | 典型用途 |
|---|---|---|---|
| `pl.ND` | 非分形行主序，最后一维连续 | Tensor / Tile | 普通 GM Tensor（默认）；UB Tile（默认）；Scaling buffer |
| `pl.DN` | 非分形列主序 | Tile | UB 上 `[ROWS, 1]` 列向量（归约结果、histogram 索引） |
| `pl.NZ` | NZ 分形排列 | Tensor / Tile | NZ GM Tensor（输出）；L1 Mat（默认）；A5 的 L0A（默认）；L0C Acc（默认） |
| `pl.ZN` | ZN 分形排列 | Tile | L0B Right（默认）；转置搬入时的 L1 Mat |
| `pl.ZZ` | ZZ 分形排列 | Tile | A3 的 L0A（默认） |
| `pl.NN` | NN 分形排列 | — | 当前版本已定义但无公开 API 场景实际使用 |

以上短名称分别等价于 `pypto_pro.language.TensorLayout.ND`、`DN`、`NZ`、`ZN`、`NN`、`ZZ`。

---

## 布局声明

### Tensor 布局

GM Tensor 统一使用 `ND`（行主序），`layout` 可省略（默认 `ND`）：

```python
x: pl.Tensor[[64, 128], pl.DT_FP16]                   # 默认 ND
x_nz: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]         # NZ 分形布局（用于输出）
```

> [!IMPORTANT] 重要
> 转置搬运由 [`load`](../operation/memory_data_movement/load.md)/[`load_tile`](../operation/memory_data_movement/load_tile.md) 的 `order` 参数决定（`order=[1,0]` 即 `is_transpose=True`），需与 L1 Tile 布局 `ZN` 配合。详见下文[转置搬入](#转置搬入)。

### Tile 布局

Tile 通过 [`pl.TileType`](TileType.md) 的 `layout` 参数指定。不指定时，默认值由内存空间和芯片架构决定：

| 内存空间 | A3 默认 | A5 默认 | 额外允许 |
|---|---|---|---|
| `Vec`（UB） | 无默认值 | 无默认值 | `ND`；`DN`（仅特定 API 要求的列主序场景） |
| `Mat`（L1） | `NZ` | `NZ` | `ZN`（转置搬入）；`UINT64`/`INT64` 还允许 `ND` |
| `Left`（L0A） | `ZZ` | `NZ` | `ZZ`、`NZ` |
| `Right`（L0B） | `ZN` | `ZN` | — |
| `Acc`（L0C） | `NZ` | `NZ` | — |
| `Scaling` | `ND` | `ND` | — |

---

## 调用示例

### 转置搬入

GM Tensor 统一使用 `ND` 布局。Cube 场景下，是否需要转置搬入由 Tensor 的 shape 决定：当传入的 shape 与 L1 Tile 的 shape 轴序一致时不需要转置；当轴序相反时需要转置，此时 `load` 设置 `order=[1, 0]`（`is_transpose=True`），L1 Tile 布局配 `ZN`。`order` 参数由框架内部解析为 `is_transpose` 标志（`order[0] > order[1]` 即为转置），供 codegen 生成对应的 TLOAD 指令。

以 `C[M, N] = A[M, K] @ B[K, N]` 为例：

| 操作数 | Tensor shape | 是否转置 | `load` 的 `order` | L1 Mat Tile layout |
|---|---|---|---|---|
| 左矩阵 A | `[M, K]` | 否 | `[0, 1]`（默认） | `NZ`（默认） |
| 左矩阵 A | `[K, M]` | 是 | `[1, 0]` | `ZN` |
| 右矩阵 B | `[K, N]` | 否 | `[0, 1]`（默认） | `NZ`（默认） |
| 右矩阵 B | `[N, K]` | 是 | `[1, 0]` | `ZN` |

#### 左矩阵转置搬入

Tensor shape 为 `[K, M]`（与 L1 Tile 的 `[M, K]` 轴序相反），`load` 设置 `order=[1, 0]`，L1 Tile 配 `ZN`：

```python
@pl.jit(auto_mutex=True)
def kernel_left_transpose(
    a: pl.Tensor[[K, M], pl.DT_FP16],               # [K, M]，需转置
    b: pl.Tensor[[K, N], pl.DT_FP16],               # [K, N]，不转置
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),  # ZN
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat),                # NZ（默认）
        addrs=0x10000, mutex_ids=[1])
    ...
    with pl.section_cube():
        cur_a = a_l1.current()
        pl.load(cur_a, a, [0, 0], order=[1, 0])    # 转置搬入
        cur_b = b_l1.current()
        pl.load(cur_b, b, [0, 0])                   # 不转置
        ...
```

#### 右矩阵转置搬入

Tensor shape 为 `[N, K]`（与 L1 Tile 的 `[K, N]` 轴序相反），`load` 设置 `order=[1, 0]`，L1 Tile 配 `ZN`：

```python
@pl.jit(auto_mutex=True)
def kernel_right_transpose(
    a: pl.Tensor[[M, K], pl.DT_FP16],               # [M, K]，不转置
    b: pl.Tensor[[N, K], pl.DT_FP16],               # [N, K]，需转置
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat),                # NZ（默认）
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),  # ZN
        addrs=0x10000, mutex_ids=[1])
    ...
    with pl.section_cube():
        cur_a = a_l1.current()
        pl.load(cur_a, a, [0, 0])                   # 不转置
        cur_b = b_l1.current()
        pl.load(cur_b, b, [0, 0], order=[1, 0])    # 转置搬入
        ...
```

### UB Tile 的 ND 与 DN

UB Tile 大部分情况使用 `ND`（行主序）。`DN`（列主序）仅在特定 API 要求时使用，典型场景是归约操作产生 `[ROWS, 1]` 列向量：

```python
# 普通数据 Tile：ND（行主序）
tile_src = pl.TileType(shape=[32, 128], dtype=pl.DT_UINT16,
                       target_memory=pl.MemorySpace.Vec, layout=pl.ND)

# 归约结果列向量：DN（列主序）
tile_red = pl.TileType(shape=[TILE_ROWS, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec, layout=pl.DN)
```

### Cube 分形布局

Cube 计算的 L1/L0A/L0B/L0C 各级 Buffer 使用分形布局，默认值由内存空间决定：

```python
# L1 Mat：默认 NZ
mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)

# L0B Right：默认 ZN
right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN)

# L0C Acc：默认 NZ，fp32 需指定 fractal=1024
acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024)
```

### A3 架构 L0A 的 ZZ 布局

A3 架构下 L0A（左矩阵）默认 `ZZ`；A5 架构下默认 `NZ`。以下为显式指定 `ZZ` 的用法：

```python
tile_type = pl.TileType(
    shape=[128, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Left, layout=pl.ZZ,
)
```

### NZ Tensor 输出

输出 Tensor 标注为 `NZ` 时，`pl.store` 会将 Acc 的计算结果按 NZ 分形直接写入 GM Tensor，无需额外格式转换：

```python
@pl.jit()
def store_nz_cce_kernel(
    ...
    nz_out: pl.Tensor[[64, 64], pl.DT_FP32, pl.NZ],   # NZ 输出 Tensor
    ...
):
    ...
    pl.store(nz_out, acc, [0, 0])    # 按 NZ 分形写入 GM
```

### NN 布局

`NN` 分形在当前版本中已定义（`TensorLayout.NN`），但尚无公开 API 场景实际使用。开发者无需显式指定该布局。
