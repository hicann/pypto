# pypto_pro.language.simt.cast

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

将源操作数转换为指定的目的数据类型。

## 函数原型

```python
pypto_pro.language.simt.cast(
    value: Scalar,
    dtype: DType,
    *,
    mode: RoundMode = pl.RoundMode.CAST_NONE,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型，支持DT_FP16、DT_BF16、DT_FP32、DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32和DT_UINT64。Tensor或Tile元素需通过下标访问后传入。 |
| dtype | 输入 | 目的数据类型，DataType类型，必须使用编译期pl.DT_*常量。支持的转换组合见下表。 |
| mode | 输入 | 可选，舍入模式，RoundMode类型，必须使用编译期pl.RoundMode.*常量，默认值为pl.RoundMode.CAST_NONE。支持的取值见下表。 |

支持的数据类型转换如下，未列出的组合不支持：

| 源数据类型 | 目的数据类型 | 支持的mode |
|---|---|---|
| DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64 | DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64 | CAST_NONE |
| DT_FP16、DT_BF16 | DT_FP32 | CAST_NONE |
| DT_FP32 | DT_FP16 | CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC、CAST_ODD |
| DT_FP32 | DT_BF16 | CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC |
| DT_FP32 | DT_INT32、DT_UINT32、DT_INT64、DT_UINT64 | CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC |
| DT_INT32、DT_UINT32、DT_INT64、DT_UINT64 | DT_FP32 | CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC |

舍入模式说明如下。除特别说明外，示例均为DT_FP32转DT_INT32：

| mode | 说明 | 示例 |
|---|---|---|
| CAST_NONE | 不显式指定舍入规则。相同数据类型直接返回原值；FP32转32位或64位整数时向零截断；FP32转FP16或BF16时向最近偶数舍入；其余支持路径按普通数值转换处理。 | 1.6 → 1，-1.6 → -1 |
| CAST_RINT | 舍入到最近值，中间值取偶数 | 2.5 → 2，3.5 → 4 |
| CAST_ROUND | 舍入到最近值，中间值远离零 | 2.5 → 3，-2.5 → -3 |
| CAST_FLOOR | 向负无穷方向舍入 | 1.6 → 1，-1.6 → -2 |
| CAST_CEIL | 向正无穷方向舍入 | 1.6 → 2，-1.6 → -1 |
| CAST_TRUNC | 向零方向舍入 | 1.6 → 1，-1.6 → -1 |
| CAST_ODD | 发生精度丢失时，将结果的最低有效位设为1 | DT_FP32转DT_FP16：1.0001 → 1.0009765625 |

## 约束说明

只能在由@pl.simt.function定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回转换后的Scalar，数据类型由dtype指定。

## 调用示例

```python
import pypto_pro.language as pl

@pl.simt.function(max_threads=256)
def cast_floor_fp32_to_int32(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.cast(
        source[0, tid],
        pl.DT_INT32,
        mode=pl.RoundMode.CAST_FLOOR,
    )


@pl.jit()
def simt_cast_floor_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_INT32],
):
    with pl.section_vector():
        pl.simt.launch(cast_floor_fp32_to_int32, threads=256, args=(source, output))
```
