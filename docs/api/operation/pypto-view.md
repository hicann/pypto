# pypto.view

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

从输入Tensor中取出部分视图，用于后续计算。

## 函数原型

```python
view(input: Tensor, shape: List[int] = None, offsets: List[Union[int, SymbolicScalar]] = None, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, dtype: DataType = None,
) -> Tensor:
```

## 参数说明


| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| input       | 输入      | 源操作数。<br> 支持的数据类型为：PyPto支持的数据类型<br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| shape       | 输入      | 获取出视图的大小。<br> Shape Size不大于2147483647（即INT32_MAX） |
| offsets     | 输入      | 获取视图时每个维度相对于input的偏移。<br> 需要保证offsets小于input的Shape |
| valid_shape | 输入      | 取出示意图块的有效数据大小。<br> 需要保证valid_shape小于input的Shape |
| dtype       | 输入      | 返回值的数据类型，允许将输入数据解读为不同数据类型 |

## 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape为参数shape指定大小，若指定了valid\_shape，则真实大小为valid\_shape。若指定dtype，则会将输入按照dtype进行读取。

## 调用示例

-   基本使用方法

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [0, 4]
    y = pypto.view(x, shape, offsets)
    ```

    结果示例如下：

    ```python
    输入数据x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4]]
    输出数据y: [[3 3 4 4],
                [3 3 4 4],
                [3 3 4 4],
                [3 3 4 4]]
    ```

-   增加valid\_shape

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [2, 4]
    valid_shape = [2, 4]
    y = pypto.view(x, shape, offsets, valid_shape)
    ```

    结果示例如下：

    ```python
    输入数据x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 5 5 6 6],
                [1 1 2 2 5 5 6 6]]
    输出数据y: [[5 5 6 6],
                [5 5 6 6],
                [0 0 0 0],
                [0 0 0 0]]
    ```

-   指定dtype

    ```python
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.view(x, dtype=pypto.DT_INT8)
    ```

    结果如下：

    ```python
    输入数据x: 
    [[0.9405094  0.20237109],
     [0.99819463 0.13246714]]
    
    输出数据y: 
    [[  57  -59  112   63   94   58   79   62],
     [ -81 -119  127   63  119  -91    7   62]]
    
    ```

