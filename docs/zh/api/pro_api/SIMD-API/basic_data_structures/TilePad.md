# pypto_pro.language.TilePad

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

Tile边界不足时的填充方式枚举，用于尾块/非满块场景。

当Tile的有效数据区域小于其shape时（如动态维度的尾块），超出有效区域的部分需要按指定模式填充。

## 原型定义

```python
PYPTO_DECLARE_ENUM(
    TilePad,
    null,
    zero,
    max,
    min
)
```

## 参数说明

| 参数值 | 说明 |
|---|---|
| null | 不填充，默认值，适用于不需要处理无效区域的场景。 |
| zero | 补0，典型用于卷积padding和零初始化。 |
| max | 补对应数据类型的最大值，典型用于取最小值操作的无效区域。 |
| min | 补对应数据类型的最小值，典型用于flash attention掩码，使无效行在max/softmax计算中被忽略。 |

## 约束说明

- 在Flash Attention掩码场景中，当KV长度不是Tile大小的整数倍时，最后一块的无效行需要补FP32最小值，使其在后续的row_max和exp操作中被忽略。

- 在卷积padding场景中，边界区域需要填充零值。

## 调用示例

### Tile填充模式

```python
import pypto_pro.language as pl
# 不填充（默认）
tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Vec)

# 补零（卷积 padding）
tt_pad = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)

# 补最小值（flash attention 掩码）
tt_mask = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                      target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.min)
```
