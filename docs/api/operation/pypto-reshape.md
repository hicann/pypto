# pypto.reshape

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

改变Tensor形状， 改变valid\_shape部分的形状\(Shape\)

## 函数原型

```python
reshape(input: Tensor,shape: List[int],*,valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, inplace: bool = False) -> Tensor
```

## 参数说明


| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| input       | 输入      | 源操作数。 <br> 支持的数据类型为：PyPTO支持的数据类型 <br> 不支持空Tensor，Shape Size不大于INT32_MAX。 |
| shape       | 输入      | 目标Shape。 <br> Shape Size不大于INT32_MAX；某个维度为-1时，支持自动推导。 |
| valid_shape | 输入      | 输出Tensor的有效数据的Shape，且valid_shape Size不大于INT32_MAX。 |
| inplace     | 输入      | 是否为inplace；参数为True时，不会为输出申请新地址； |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，形状\(Shape\)为输入参数指定的shape。

## 约束说明

inplace为True时，需要保证输入输出分别是当前loop的输入输出；输出不可作为整个Function的输出

## 调用示例

示例1：

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.reshape(x, [4, 1], [2, 1])
z = pypto.add(y, 1.0)
```

结果示例如下：

```python
输入数据x: [[1, 2],
            [3, 4]]
输出数据y: [[1],
            [2],
            [3],
            [4]
输出数据z: [[2],
            [3],
            [3],
            [4]]
```

示例2：

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
for _ in pypto.loop(1, name="reshape_inplace", idx_name="tmp_loop"):
    x_1 = x.reshape(x, [4], inplace=True)
for _ in pypto.loop(1, name="loop", idx_name="loop"):
    y = pypto.add(x_1, 1.0)
```

结果示例如下：

```python
输入数据x: [[1, 2],
            [3, 4]]
输出数据y: [2, 3, 4, 5]
```

