# pypto_pro.language.simt.launch

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

在JIT Kernel的Vector执行域中启动由@pl.simt.function定义的SIMT入口函数。

每个执行到launch的Vector Block启动一个SIMT线程块，threads配置改线程块的尺寸。

## 函数原型

```python
pypto_pro.language.simt.launch(
    callee: Callable,
    *,
    threads: Union[int, Tuple[int, ...]],
    args: Tuple,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| callee | 输入 | 待启动的SIMT入口函数，Callable类型。必须直接传入使用@pl.simt.function(max_threads=...)定义的函数名；不支持未配置max_threads的SIMT辅助函数或动态可调用对象。 |
| threads | 输入 | 单个SIMT线程块的线程尺寸，必须通过关键字参数传入。支持一至三个维度：int表示一维配置，Tuple表示一至三维配置，所有维度均须为[1, 2048]范围内的编译期整数。各维乘积不得超过callee的max_threads，max_threads须位于[1, 2048]范围内。 |
| args | 输入/输出 | 传递给callee的实参元组，必须通过关键字参数传入。实参数量、顺序和类型必须与callee形参一致。支持以下实参：<br>- Scalar：按值传递。<br>- Tensor：必须使用ND Layout，元素数据类型位宽不得小于8 bit。<br>- Tile：必须位于MemorySpace.Vec，使用ND Layout，元素数据类型位宽不得小于8 bit。<br>Tensor和Tile必须以完整变量传入，不支持元素下标访问表达式、Slice或Tile subview。 |

`threads`参数与[`block_dim()`](block_dim.md)返回的线程块尺寸对应关系如下：

| `threads`参数 | `block_dim()`返回值 |
|---|---|
| `x`或`(x,)` | `(x, 1, 1)` |
| `(x, y)` | `(x, y, 1)` |
| `(x, y, z)` | `(x, y, z)` |

## 约束说明

- 只能在pl.section_vector作用域内调用；
- 不支持在SIMT函数内嵌套调用launch；

## 返回值说明

无。

## 调用示例

```python
@pl.simt.function(max_threads=256)
def auto_mutex_add(data: pl.Tile[[1, 256], pl.DT_FP32], delta: pl.DT_FP32):
    tid = pl.simt.linear_thread_idx()
    data[0, tid] = data[0, tid] + delta


@pl.jit()
def simt_auto_mutex_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(
        shape=[1, 256],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    data = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        pl.load(data.current(), x, [0, 0])
        pl.simt.launch(auto_mutex_add, threads=256, args=(data.current(), delta))
        pl.store(out, data.current(), [0, 0])
```
