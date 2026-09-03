# pypto_pro.language.store

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

把UB或L0C Buffer中的Tile按绝对元素坐标写回GM，是与[pypto_pro.language.load](load.md)对应的写回接口。写回过程中可以融合ReLU、量化或原子累加。

源Tile支持位于UB或L0C Buffer，不支持从L1 Buffer直接写回GM。

如果希望按“第几块Tile”定位写出位置，需要使用[pypto_pro.language.store_tile](store_tile.md)。

## 函数原型

```python
pypto_pro.language.store(
    dst_tensor: Tensor,
    src_tile: Tile,
    offsets: Offset,
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    order: Optional[List[int]] = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: Optional[STPhase] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst_tensor | 输出 | 目标GM Tensor。源Tile位于UB时支持ND、DN和NZ布局；源Tile位于L0C Buffer时支持ND和NZ布局。写入起始位置和有效写入区域不能超过各维shape。 |
| src_tile | 输入 | 待写回GM的源Tile，须位于UB或L0C Buffer。UB Tile的首地址须按32字节对齐，L0C Buffer Tile的首地址须按64字节对齐。不支持从L1 Buffer直接写回GM。 |
| offsets | 输入 | 目标Tensor各维度的绝对元素偏移，长度与dst_tensor的维数相同，不支持负数。 |
| relu_pre_mode | 输入 | 可选，仅用于L0C Buffer写回GM时在写回前执行ReLU。支持pypto_pro.language.ReluPreMode.NormalRelu。不能与Tile类型的scale同时使用。 |
| scale | 输入 | 可选，仅用于L0C Buffer写回GM时设置量化比例。支持float、Scalar或Tile类型。Scalar支持DT_FP32、DT_INT32和DT_INT64。Tile用于按列分别设置比例，须位于MemorySpace.Scaling对应的Fixpipe Buffer，数据类型为DT_INT64，shape为[1, N]，其中N是16的倍数且不大于512。不支持Tensor类型和[N, 1]形式。 |
| order | 输入 | 可选，指定源Tile各维度对应的目标Tensor维度。各维度编号必须在目标Tensor的维度范围内、不能重复并按升序排列；省略时对应目标Tensor的最后两个维度。 |
| atomic | 输入 | 可选，原子写模式，[pypto_pro.language.AtomicType](../../basic_data_structures/AtomicType.md)类型。支持AtomicNone和AtomicAdd。 |
| phase | 输入 | 可选，L0C Buffer中多步矩阵计算结果写回GM时所处的阶段，[pypto_pro.language.STPhase](../../basic_data_structures/STPhase.md)类型。不能与Tile类型的scale同时使用。 |

## 约束说明

### 数据类型

| 数据通路 | scale | 源数据类型 → 目标数据类型 |
|---|---|---|
| UB → GM | 不支持 | 源Tile和目标Tensor的数据类型必须相同，支持DT_FP4E2M1、DT_FP4E1M2、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32。 |
| L0C Buffer → GM | 不配置 | 支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32。 |
| L0C Buffer → GM | 配置float、Scalar或Tile类型的scale | 支持DT_FP32 → DT_INT8/DT_HF8/DT_FP8E4M3FN/DT_FP16/DT_FP32，以及DT_INT32 → DT_INT8/DT_FP16。 |

### NZ布局

当dst_tensor声明为pypto_pro.language.NZ时，其物理排布、分形轴和完整Tensor shape约束见[TensorLayout](../../basic_data_structures/TensorLayout.md#tensor布局)。store还需满足以下NZ搬运约束：

- 仅支持NZ Tile到GM NZ的同布局搬运，源Tile位于UB或L0C Buffer；order省略或指定Tensor最后两轴的正序。
- Tile shape和valid M/N须满足M按16、N按目标Tensor dtype对应的C0对齐，N方向offset也须按C0对齐。
- 高维offset的前导项选择batch，最后两项为M、N方向的逻辑元素坐标。

L0C Buffer中的NZ Tile直接写回GM NZ时，仅支持以下场景之一：有效M等于源Tile的M维大小向上对齐至16的倍数，或有效N不大于C0。不满足时，需先搬到UB，再从UB写回GM。

### 原子累加

- AtomicAdd仅适用于从UB或L0C Buffer写回GM的store操作，load和move等接口不支持原子累加。
- 源Tile位于UB时，源Tile与目标Tensor的数据类型必须相同，支持DT_INT8、DT_INT16、DT_FP16、DT_BF16、DT_INT32和DT_FP32。
- 源Tile位于L0C Buffer且不配置scale时，支持DT_FP32 → DT_FP32/DT_FP16/DT_BF16，以及DT_INT32 → DT_INT32。
- 源Tile位于L0C Buffer且配置float或Scalar类型的scale时，支持DT_FP32 → DT_INT8/DT_FP16/DT_FP32，以及DT_INT32 → DT_INT8/DT_FP16。源、目的数据类型不同时，先转换为目标数据类型，再执行原子累加。
- 接口不会自动清零目标Tensor。首次累加前，调用方必须将目标区域初始化为零或预期的累加初值。
- 多核同时累加同一目标地址时，每次更新具有原子性。由于浮点加法不满足结合律，更新顺序不同时结果可能存在微小差异。
- AtomicAdd不能与Tile类型的scale同时使用。

## 返回值说明

无。

## 调用示例

### UB写回GM

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

### 写回前融合ReLU

```python
pl.store(relu_out, acc, [0, 0], relu_pre_mode=pl.ReluPreMode.NormalRelu)
```

### 原子累加

```python
# 多核或多步累加到同一GM位置
pl.store(dk_out, fp32_row_tile, [b_id, g_id, single_indice, 0], atomic=pl.AtomicType.AtomicAdd)

# 将L0C Buffer中的矩阵计算结果原子累加到GM
pl.store(out, acc.current(), [0, 0], atomic=pl.AtomicType.AtomicAdd)
```

### 分阶段矩阵计算结果写回

```python
pl.matmul(acc, left, right, phase=pl.AccPhase.Partial)
pl.matmul_acc(acc, acc, next_left, next_right, phase=pl.AccPhase.Final)
pl.store(out_tensor, acc, [i, j], phase=pl.STPhase.Final)
```
