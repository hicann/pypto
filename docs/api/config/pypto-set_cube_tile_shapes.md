# pypto.set\_cube\_tile\_shapes

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

在调用`pypto.matmul`前必须调用本接口设置矩阵运算的切分大小，具体切分配置可参考[Matmul高性能编程](https://gitcode.com/cann/pypto/blob/master/docs/tutorials/debug/matmul_performance_guide.md)。


## 函数原型

```python
set_cube_tile_shapes(m: List[int], k: List[int], n: List[int], enable_multi_data_load: bool = False, enable_split_k: bool = False) -> None
```

## 参数说明


| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| m                      | 输入      | m维度在L0和L1上的TileShape（切片形状）的切分大小，分别对应mL0和mL1的切分大小 |
| k                      | 输入      | k维度在L0和L1上的TileShape（切片形状）的切分大小，分别对应kL0和kL1的切分大小 |
| n                      | 输入      | n维度在L0和L1上的TileShape（切片形状）的切分大小，分别对应nL0和nL1的切分大小 |
| enable_multi_data_load | 输入      | 设置True表示使能matmul的L1大包搬运功能，False表示未使能L1的大包搬运，默认为False |
| enable_split_k         | 输入      | 设置True表示使能matmul的多核切K功能，False表示未使能多核切K，默认为False |

## 返回值说明

void

## 约束说明

TileShape需要满足以下约束条件：

-   对齐约束：
    -   要求kL0、kL1、nL0、nL1均满足32字节对齐（DT\_FP32输入场景要求满足16元素对齐）。例如：输入矩阵的数据类型为DT\_FP16时，kL0 \* sizeof\(DT\_FP16\) % 32 == 0。
    -   A矩阵在format为ND且转置场景时（即数据排布为\(K, M\)），要求mL0满足32字节对齐。
    -   A、B矩阵在format为NZ场景时，要求外轴切分大小满足16元素对齐，内轴切分大小满足32字节对齐。例如，在A矩阵非转置场景，外轴为M、内轴为K，要求mL0、mL1满足16元素对齐，kL0、kL1满足32字节对齐。
    -   0 < mL0 <= mL1 且 mL1 % mL0 == 0
    -   0 < kL0 <= kL1 且 kL1 % kL0 == 0
    -   0 < nL0 <= nL1 且 nL1 % nL0 == 0

-   buffer空间约束：
    -   L0A、L0B、L0C空间约束：
        -   当输入dtype为DT\_FP16、DT\_BF16或DT\_FP32

            CeilAlign\(mL0,16\)\* CeilAlign\(kL0,16\) \* sizeof\(aDtype\) <= L0A\_size

            CeilAlign\(nL0,16\) \* CeilAlign\(kL0,16\)\* sizeof\(bDtype\) <= L0B\_size

            CeilAlign\(mL0,16\)\* CeilAlign\(nL0,16\)\* sizeof\(cDtype\) <= L0C\_size

            其中aDtype、bDtype为输入dtype，cDtype为DT\_FP32

        -   当输入dtype为DT\_INT8

            CeilAlign\(mL0,32\)\* CeilAlign\(kL0,32\) \* sizeof\(aDtype\) <= L0A\_size

            CeilAlign\(nL0,32\) \* CeilAlign\(kL0,32\)\* sizeof\(bDtype\) <= L0B\_size

            CeilAlign\(mL0,32\)\* CeilAlign\(nL0,32\)\* sizeof\(cDtype\) <= L0C\_size

            其中aDtype、bDtype为输入dtype，cDtype为DT\_INT32

    -   L1空间约束：

        -   输入dtype为DT\_FP16或DT\_BF16或DT\_FP32

            CeilAlign\(mL1,16\)\* CeilAlign\(kL1,16\) \* sizeof\(aDtype\) + CeilAlign\(nL1,16\) \* CeilAlign\(kL1,16\) \* sizeof\(bDtype\) <= L1\_size

            其中aDtype、bDtype为输入dtype

        -   输入dtype为DT\_INT8

            CeilAlign\(mL1,32\)\* CeilAlign\(kL1,32\) \* sizeof\(aDtype\) + CeilAlign\(nL1,32\) \* CeilAlign\(kL1,32\) \* sizeof\(bDtype\) <= L1\_size

            其中aDtype、bDtype为输入dtype

        其中，CeilAlign（元素对齐）基本实现为：

        ```
        def ceil_align(value, align) {  return ((value + align - 1) // align) * align;}
        ```

Bias场景约束条件：

-   Bias空间约束：
    -   BTBuffer空间大小为1kb，且bias数据到达BTBuffer全部转为fp32，需满足以下约束:

        nL0 \* 4 <= BTBuffer\_size

FixPipe场景约束条件：

-   FixBuffer空间约束：
    -   FixBuffer空间大小为2kb，且scaleTensor数据为uint64\_t，需满足以下约束:

        nL0 \* 8 <= FixBuffer\_size

Output需要满足以下约束条件：

-   格式约束：当输出为NZ格式时，需要满足内轴（N轴）32字节对齐

当输入矩阵维度为3维或4维时，set\_l1\_tile、enable\_split\_k参数仅支持False，即不支持使能L1大包搬运及多核切K功能。

多核切K场景支持数据类型：

-   输入矩阵数据类型为DT\_FP16时，out\_dtype可选DT\_FP32。
-   输入矩阵数据类型为DT\_BF16时，out\_dtype可选DT\_FP32。
-   输入矩阵数据类型为DT\_INT8时，out\_dtype可选DT\_INT32。
-   输入矩阵数据类型为DT\_FP32时，out\_dtype可选DT\_FP32。

## 调用示例

```python
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], False, True)
```

