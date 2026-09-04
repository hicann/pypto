# pypto_pro.language.quant

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

把FP32源Tile量化为INT8或UINT8。scale是量化乘子（常见量化定义中真实scale的倒数），scale[i, 0]和offset[i, 0]按行广播：

$$
out_{i,j}=\begin{cases}
\operatorname{clamp}_{[-128,127]}\left(\operatorname{roundToEven}\left(src_{i,j}\times scale_{i,0}\right)\right), & mode=\mathrm{SYM} \\
\operatorname{clamp}_{[0,255]}\left(\operatorname{roundToEven}\left(src_{i,j}\times scale_{i,0}+offset_{i,0}\right)\right), & mode=\mathrm{ASYM}
\end{cases}
$$

非对称模式先在FP32中完成乘法和加法，再对合并后的结果执行一次舍入；不是先舍入乘积再添加offset。

## 函数原型

```python
pypto_pro.language.quant(
    out: Tile,
    src: Tile,
    scale: Tile,
    *,
    mode: QuantMode = pypto_pro.language.QuantMode.SYM,
    offset: Optional[Tile] = None,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | UB、RowMajor Tile。mode=SYM时必须为DT_INT8；mode=ASYM时必须为DT_UINT8。逻辑shape和valid_shape须与src一致。 |
| src | 输入 | UB、RowMajor、DT_FP32 Tile。逻辑shape和valid_shape须与out一致。 |
| scale | 输入 | UB、RowMajor、DT_FP32 Tile。若src.valid_shape=[M,N]，则scale.valid_shape须为[M,1]，物理shape的行数不得小于M且列数必须为1；第i行的scale[i,0]广播到src第i行全部有效列。该值直接与src相乘，因此若使用常见公式$q=round(x/s)$，此处应传入$1/s$。 |
| mode | 输入 | 可选，编译期[pypto_pro.language.QuantMode](../../basic_data_structures/QuantMode.md)枚举值，可取SYM（默认）或ASYM。模式同时决定输出dtype和是否需要offset。 |
| offset | 输入 | 可选，mode=ASYM时必填，须为UB、RowMajor、DT_FP32 Tile，物理shape和valid_shape均须与scale一致；第i行的offset[i,0]广播到对应数据行。mode=SYM时不参与后端调用，建议省略而不是传入无效占位Tile。 |

## 返回值说明

无返回值。量化结果写入out。

## 约束说明

1. mode必须在编译期确定，不能使用运行时Scalar或Tensor动态选择。
2. 舍入规则固定为舍入到最近值，中间值取偶数，接口不提供RoundMode参数。有限输入超出目标整数范围时分别饱和到[-128,127]或[0,255]。
3. scale通常应为有限正数；接口不检查其数值范围。scale=0、负数、NaN或Inf会按底层浮点和类型转换规则执行，不应作为可移植量化用法。
4. ASYM模式的offset在FP32域中参与舍入，因此允许FP32存储，但规范用法应传入可表示零点的有限数值。
5. out、src、scale和offset应使用互不重叠的UB区域。由于源、目的位宽不同，本接口不保证地址重叠时的结果。
6. 接口只定义src.valid_shape有效区域内的输出；有效区域外的内容未定义。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def quant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP32],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_INT8],
):
    tile_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x0000, mutex_ids=[0])
    tile_scale = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                    addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
                                  addrs=0xA000, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.quant(cur_out, cur_src, cur_scale, mode=pl.QuantMode.SYM)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:quant:start -->
```bash
输入数据src：[[-16 -15.75 -15.5 -15.25 -15 -14.75 -14.5 -14.25 ...], [16 16.25 16.5 16.75 17 17.25 17.5 17.75 ...], [48 48.25 48.5 48.75 49 49.25 49.5 49.75 ...], [80 80.25 80.5 80.75 81 81.25 81.5 81.75 ...], ...]
输入数据scale：[[4], [4], [4], [4], ...]
输出数据out：[[-64 -63 -62 -61 -60 -59 -58 -57 ...], [64 65 66 67 68 69 70 71 ...], [127 127 127 127 127 127 127 127 ...], [127 127 127 127 127 127 127 127 ...], ...]
```
<!-- pypto-doc-output:quant:end -->
