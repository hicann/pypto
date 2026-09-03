# pypto_pro.language.dequant

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

把INT8或INT16源Tile反量化为FP32。scale[i,0]和offset[i,0]按行广播：

$$
out_{i,j}=\left(\operatorname{FP32}(src_{i,j})-offset_{i,0}\right)\times scale_{i,0}
$$

计算顺序为：整数源数据扩展并转换为FP32、减去offset、乘以scale。若与[quant](quant.md)配套使用，量化阶段传入的乘子通常为本接口scale的倒数。

## 函数原型

```python
pypto_pro.language.dequant(
    out: Tile,
    src: Tile,
    scale: Tile,
    offset: Tile,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | UB、RowMajor、DT_FP32 Tile；逻辑shape和valid_shape须与src一致。 |
| src | 输入 | UB、RowMajor Tile，dtype为DT_INT8或DT_INT16；逻辑shape和valid_shape须与out一致。 |
| scale | 输入 | UB、RowMajor、DT_FP32 Tile。若out.valid_shape=[M,N]，则scale.valid_shape须为[M,1]，物理shape的行数不得小于M且列数必须为1；第i行的scale[i,0]广播到输出第i行全部有效列。 |
| offset | 输入 | UB、RowMajor、DT_FP32 Tile，物理shape和valid_shape均须与scale一致；第i行的offset[i,0]广播到对应数据行。对称量化数据须传入全零Tile，本参数不能省略。 |

## 返回值说明

无返回值。反量化结果写入out。

## 约束说明

1. out、src、scale和offset应使用互不重叠的UB区域；本接口不保证地址重叠时的结果。
2. scale和offset必须覆盖src的每个有效行，不能依赖参数Tile最后一行或最后一列隐式扩展。
3. 计算使用FP32。scale或offset中的NaN、Inf按照FP32运算传播；超出FP32范围的结果按目标硬件浮点规则处理。
4. 接口只定义src.valid_shape有效区域内的输出；有效区域外的内容未定义。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dequant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_INT8],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    offset: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x0000, mutex_ids=[0])
    tile_scale = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                    addrs=0x4000, mutex_ids=[1])
    tile_offset = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                     addrs=0x5000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x6000, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_offset = tile_offset.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.load(cur_offset, offset, [0, 0])
        pl.dequant(cur_out, cur_src, cur_scale, cur_offset)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:dequant:start -->
```bash
输入数据src：[[-32 -31 -30 -29 -28 -27 -26 -25 ...], [-32 -31 -30 -29 -28 -27 -26 -25 ...], [-32 -31 -30 -29 -28 -27 -26 -25 ...], [-32 -31 -30 -29 -28 -27 -26 -25 ...], ...]
输入数据scale：[[0.25], [0.25], [0.25], [0.25], ...]
输入数据offset：[[0], [0], [0], [0], ...]
输出数据out：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], ...]
```
<!-- pypto-doc-output:dequant:end -->
