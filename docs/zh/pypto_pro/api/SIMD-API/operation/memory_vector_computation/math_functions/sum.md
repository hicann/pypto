# pypto_pro.language.sum

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

沿指定维度对 `src` tile 求和。`dim=0` 沿最后一维对每行求和，输出 `[行数, 1]`；`dim=1` 沿第一维对每列求和，输出 `[1, 列数]`。

## 函数原型

```python
pypto_pro.language.sum(out, src, tmp, *, dim=0)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 规约结果 tile |
| `src` | 输入 | 源 tile |
| `tmp` | 输入 | 临时 tile（中间计算用） |
| `dim` | 输入 | 归约方向 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：与 `src` 一致<br>`dim=0` 时 shape 为 `[行数, 1]`，须设 `layout=pl.DN`；`dim=1` 时 shape 为 `[1, 列数]` |
| `src` | 输入 | 数据类型：b16、b32<br>shape：`[行数, 列数]` |
| `tmp` | 输入 | 数据类型：与 `src` 一致<br>shape：与 `src` 一致<br>硬件中间计算用，不可与 `out`/`src` 重叠 |
| `dim` | 输入 | `0`：沿最后一维做行向归约；`1`：沿第一维做列向归约。默认值为 `0` |

## 流水类型

V（向量计算流水）。

## 调用示例

### dim=0

下面是一个完整 kernel：对 64×128 FP32 源 tile 做行向求和，输出 `[64, 1]`。注意输出 tile 需设 `layout=pl.DN`。vector kernel 开 `auto_mutex`，同步由 `make_tile_group` 自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def row_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[64, 1], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                         layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sum(cur_out, cur_a, cur_tmp)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:row_sum:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[1.008000e+03], [5.104000e+03], [9.200000e+03], [1.329600e+04], ...]
```
<!-- pypto-doc-output:row_sum:end -->

### dim=1

下面的 kernel 对 64×128 FP32 源 tile 做列向求和，输出 `[1, 128]`。

```python
@pl.jit(auto_mutex=True)
def col_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[1, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:col_sum:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[6.400000e+04 6.401600e+04 6.403200e+04 6.404800e+04 6.406400e+04 6.408000e+04 6.409600e+04 6.411200e+04 ...]]
```
<!-- pypto-doc-output:col_sum:end -->

### FP16 精度说明

FP16 归约受有限精度和硬件累加顺序影响。输入规模较大或数值较大时，实测结果可能与按实数精度手算的结果不同。AscendC 的 `ReduceSum` 同样保留设备输出，并说明同一 repeat 内采用二叉树累加、不同 repeat 按顺序或二叉树累加。

以下为同一 `sum(..., dim=1)` 接口使用 FP16 输入时的实测结果：

<!-- pypto-doc-output:col_reduce_sum:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据z：[[6.390400e+04 6.390400e+04 6.390400e+04 6.416000e+04 6.416000e+04 6.416000e+04 6.419200e+04 6.419200e+04 ...]]
```
<!-- pypto-doc-output:col_reduce_sum:end -->
