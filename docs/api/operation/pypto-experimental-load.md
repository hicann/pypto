# pypto.experimental.load

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

该接口为定制接口，约束较多。不保证稳定性。

从 GM（Global Memory）中根据源数据以及在源数据中的偏移列表将所需数据离散的加载到  UB（Unified Buffer） 中。

## 函数原型

```python
load(src: Tensor, offsets: Tensor):
```

参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src     | 输入      | 源操作数。 <br> 支持的数据类型为：DT_INT32、DT_FP32、DT_FP16等。 <br> 不支持空Tensor，支持任意维度。 |
| offsets | 输入      | 待加载数据在源操作数一维视图下的偏移。 <br> 支持的数据类型为DT_INT32、DT_INT64。 <br> 不支持空Tensor，支持两维或三维。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和src相同，Shape和offsets相同。

## 调用示例

```python
src = pypto.tensor([3, 2], pypto.DT_FP32)
offsets = pypto.tensor([2, 3], pypto.DT_INT32)
dst = pypto.load(src, offsets)
```

结果示例如下：

```python
输入数据src: [[2.2, 3],
              [7.3, 1],
              [-1.5, 100.2]]
输入数据src的一维视图：[2.2, 3, 7.3, 1, -1.5, 100.2]

输入数据offsets: [[2, 5, 3],
                  [1, 2, 0]]
输出数据dst: [[7.3, 100.2, 1],
              [3, 7.3, 2.2]]
```

