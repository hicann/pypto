# pypto_pro.language.simt.log1p

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

计算源操作数加1后的自然对数，计算公式如下：

$$result = \ln(1 + value)$$

## 函数原型

```python
pypto_pro.language.simt.log1p(
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型，仅支持DT_FP32，实数定义域为value > -1。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pl.simt.function定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回源操作数加1后的自然对数，数据类型为DT_FP32。当前实现的数值精度等同于先执行一次DT_FP32加法，再对加法结果计算自然对数。

## 调用示例

```python
import pypto_pro.language as pl

@pl.simt.function(max_threads=64)
def log1p_fp32(
    source: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.log1p(source[0, tid])


@pl.jit()
def simt_log1p_kernel(
    source: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        pl.simt.launch(log1p_fp32, threads=64, args=(source, output))
```
