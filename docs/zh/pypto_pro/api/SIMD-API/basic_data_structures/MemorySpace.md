# pypto_pro.language.MemorySpace

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

内存空间枚举，用于标记 Tile 所在的物理存储位置，是 [`pypto_pro.language.TileType`](TileType.md) 的关键属性。

不同内存空间对应昇腾芯片上不同的物理存储区域，决定了 Tile 能参与哪些计算、需要哪条流水搬运。

## 取值

| 取值 | 物理位置 | 说明 | 典型用途 |
|---|---|---|---|
| `pypto_pro.language.MemorySpace.DDR` | 片外 DDR | 全局内存，Tensor 所在 | GM 张量存储 |
| `pypto_pro.language.MemorySpace.Vec` | 片上 UB | 向量/统一缓冲区 | 向量计算（element-wise、reduce 等）的输入输出 |
| `pypto_pro.language.MemorySpace.Mat` | 片上 L1 | 矩阵缓冲区 | matmul 的 L1 暂存（GM→L1→L0A/L0B 两跳的中间站） |
| `pypto_pro.language.MemorySpace.Left` | 片上 L0A | 左操作数缓冲区 | matmul 左矩阵输入 |
| `pypto_pro.language.MemorySpace.Right` | 片上 L0B | 右操作数缓冲区 | matmul 右矩阵输入 |
| `pypto_pro.language.MemorySpace.Acc` | 片上 L0C | 累加器缓冲区 | matmul 累加器输出 |
| `pypto_pro.language.MemorySpace.Scaling` | 片上 | 缩放/量化参数缓冲区 | quantization/反量化参数 |
| `pypto_pro.language.MemorySpace.Bias` | Bias Buffer | 底层偏置缓冲区标识 | 当前 CCE Tile codegen 未实现该内存空间映射，不能用于 `make_tile` 或 `make_tile_group` 创建 Tile |

## 补充说明

不同内存空间的 tile 在构造 [`pypto_pro.language.TileType`](TileType.md) 时有不同的默认 `layout`：

| 内存空间 | A3 默认 `layout` | A5 默认 `layout` | 额外允许 |
|---|---|---|---|
| `Vec` | 无约束 | 无约束 | — |
| `Mat` | `pl.NZ` | `pl.NZ` | `pl.ZN` 转置分形布局；UINT64/INT64 还允许 `pl.ND` |
| `Left` | `pl.ZZ` | `pl.NZ` | 同时允许 `pl.ZZ` 和 `pl.NZ` |
| `Right` | `pl.ZN` | `pl.ZN` | — |
| `Acc` | `pl.NZ` | `pl.NZ` | FP32/INT32 自动 `fractal=1024` |
| `Scaling` | `pl.ND` | `pl.ND` | — |

## 调用示例

```python
import pypto_pro.language as pl
# UB tile（向量计算）
tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)

# L1 tile（matmul 中间暂存）
tt_l1 = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)

# L0A tile（matmul 左矩阵）
tt_left = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                       layout=pl.NZ)

# Acc tile（matmul 累加器）
tt_acc = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
```
