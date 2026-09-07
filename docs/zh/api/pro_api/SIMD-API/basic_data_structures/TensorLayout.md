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

数据布局枚举，用于描述GM Tensor的存储形式和Tile的数据排列形式。

- **GM Tensor**支持ND、NZ，默认ND。
- **Tile**支持ND、DN、NZ、ZN、NN、ZZ，默认值由内存空间决定。

## 原型定义

```python
PYPTO_DECLARE_ENUM(
    TensorLayout,
    ND,
    DN,
    NZ,
    ZN,
    NN,
    ZZ
)
```

## 参数说明

| 参数值 | 说明 |
|---|---|
| ND | 非分形行主序，最后一维连续。支持用于Tensor和Tile；GM Tensor默认使用ND，典型用于普通GM Tensor、UB Tile和Fixpipe Buffer。 |
| DN | 非分形列主序。仅支持用于Tile，典型用于UB上的[ROWS, 1]列向量，如归约结果和histogram索引。 |
| NZ | NZ分形排列。支持用于Tensor和Tile，典型用于GM Tensor、L1 Buffer、L0A Buffer和L0C Buffer。 |
| ZN | ZN分形排列。仅支持用于Tile，典型用于L0B Buffer和转置搬入时的L1 Buffer。 |
| ZZ | ZZ分形排列。仅支持用于Tile，用于MX矩阵乘中左量化系数矩阵在L1 Buffer和L0A_MX Buffer中的布局。 |
| NN | NN分形排列。仅支持用于Tile，用于MX矩阵乘中右量化系数矩阵在L1 Buffer和L0B_MX Buffer中的布局。 |

## 约束说明

GM Tensor仅支持ND（行主序，默认）和NZ（分形布局）。

NZ只声明GM内存的布局，不会把普通ND buffer自动转换成NZ。调用Kernel前，输入buffer必须已经按NZ物理顺序完成packing；NZ输出也必须使用按NZ格式分配的buffer。高维NZ Tensor的最后两轴固定解释为[M, N]，所有前导轴均作为batch轴，不支持在layout中指定任意分形轴。

NZ将逻辑[..., M, N]存储为[..., ceil(N/C0), ceil(M/16), 16, C0]。M/N无需分形对齐，但实际存储空间必须按align(M, 16) × align(N, C0)的容量分配并使用上述NZ物理排布；仅分配M × N元素的紧凑buffer不受支持。补齐区不属于逻辑Tensor内容，框架不会为传入的buffer自动扩容或完成packing。INT8、FP8E4M3FN、FP8E5M2、FP8E8M0和HF8的C0为32，FP4E2M1和FP4E1M2为64，FP16和BF16为16，FP32和INT32为8。FP4的M/N同样按逻辑元素计数，不使用packed字节数作为Tensor shape。

MX矩阵计算使用的E8M0分组缩放因子在GM中仍声明为普通ND Tensor；物理shape和搬运约束见[matmul_mx](../matrix_computation/matmul_mx.md)和[load](../memory_data_movement/load.md)。

普通ND GM Tensor的转置搬运由[load](../memory_data_movement/load.md)/[load_tile](../memory_data_movement/load_tile.md)的order参数决定（order=[1,0]即is_transpose=True），需与L1 Buffer中的Tile布局ZN配合。GM NZ只支持与NZ Tile同布局正序搬运，不支持通过order转置。详见下文[转置搬入](#转置搬入)。

Tile的布局约束请参见[pypto_pro.language.TileType](TileType.md)。

## 调用示例

### GM Tensor布局声明

```python
import pypto_pro.language as pl

x: pl.Tensor[[64, 128], pl.DT_FP16]                   # 默认ND
x_nz: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]         # NZ分形布局
```

### 转置搬入

GM ND Tensor搬入L1 Buffer中的Tile时，两轴顺序一致可省略order；需要交换两轴时，load设置order=[1, 0]，目标Tile布局使用ZN。框架根据order执行对应的转置搬运。

以C[M, N] = A[M, K] @ B[K, N]为例：

| 操作数 | Tensor shape | 是否转置 | load的order | L1 Buffer中的Tile layout |
|---|---|---|---|---|
| 左矩阵A | [M, K] | 否 | [0, 1]（默认） | NZ（默认） |
| 左矩阵A | [K, M] | 是 | [1, 0] | ZN |
| 右矩阵B | [K, N] | 否 | [0, 1]（默认） | NZ（默认） |
| 右矩阵B | [N, K] | 是 | [1, 0] | ZN |

#### 左矩阵转置搬入

Tensor shape为[K, M]（与L1 Buffer中Tile的[M, K]轴序相反），load设置order=[1, 0]，L1 Buffer中的Tile配ZN：

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

Tensor shape为[N, K]（与L1 Buffer中Tile的[K, N]轴序相反），load设置order=[1, 0]，L1 Buffer中的Tile配ZN：

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

### UB Tile的ND与DN

UB Tile大部分情况使用ND（行主序）。DN（列主序）仅在特定API要求时使用，典型场景是归约操作产生[ROWS, 1]列向量：

```python
# 普通数据 Tile：ND（行主序）
tile_src = pl.TileType(shape=[32, 128], dtype=pl.DT_UINT16,
                       target_memory=pl.MemorySpace.Vec, layout=pl.ND)

# 归约结果列向量：DN（列主序）
tile_red = pl.TileType(shape=[TILE_ROWS, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec, layout=pl.DN)
```

### Cube分形布局

Cube计算的L1 Buffer、L0A Buffer、L0B Buffer和L0C Buffer使用分形布局，默认值由内存空间决定：

```python
# L1 Buffer：默认 NZ
mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)

# L0B Buffer：默认 ZN
right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN)

# L0C Buffer：默认 NZ，fp32 需指定 fractal=1024
acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024)
```

### NZ Tensor输入输出

Tensor标注为NZ时，pypto_pro.language.load/pypto_pro.language.store按NZ物理布局访问GM，目标/源Tile也必须使用NZ。以下示例展示二维GM NZ在UB中的原始搬入和写回：

```python
@pl.jit()
def copy_nz_kernel(
    nz_in: pl.Tensor[[64, 64], pl.DT_FP16, pl.NZ],
    nz_out: pl.Tensor[[64, 64], pl.DT_FP16, pl.NZ],
):
    tile_type = pl.TileType(
        shape=[64, 64], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
    )
    tile = pl.make_tile(tile_type, addr=0x0000)
    with pl.section_vector():
        pl.load(tile, nz_in, [0, 0])
        pl.store(nz_out, tile, [0, 0])
```

高维NZ沿用相同写法，例如pypto_pro.language.Tensor[[B, H, M, N], dtype, NZ]固定以最后两轴M/N作为分形轴，B/H为batch轴。GM NZ不支持通过order=[1, 0]转置，也不支持直接搬入ND/ZN Tile。完整分形、offset以及L0C Buffer直接写回限制见[load](../memory_data_movement/load.md)和[store](../memory_data_movement/store.md)。

### MX矩阵乘量化系数的ZZ与NN

ZZ和NN分别用于MX矩阵乘中左量化系数矩阵和右量化系数矩阵。L1 Buffer中的Tile默认使用NZ，因此左量化系数矩阵需要显式指定ZZ，右量化系数矩阵需要显式指定NN；L0A_MX Buffer中的Tile仅支持ZZ，L0B_MX Buffer中的Tile仅支持NN。

```python
# 左、右量化系数矩阵的逻辑shape分别为[M,G]和[G,N]，其中G=K/32。
M, G, N = 64, 4, 64

# L1 Buffer的默认布局是NZ，因此需要显式指定ZZ或NN。
scale_a_l1_type = pl.TileType(
    shape=[M, G],
    dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat,
    layout=pl.ZZ,
)
scale_b_l1_type = pl.TileType(
    shape=[G, N],
    dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat,
    layout=pl.NN,
)

# L0A_MX Buffer/L0B_MX Buffer分别仅支持ZZ/NN；未指定layout时自动采用对应布局。
scale_a_type = pl.TileType(
    shape=[M, G],
    dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleLeft,
)
scale_b_type = pl.TileType(
    shape=[G, N],
    dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleRight,
)
```
