# pypto.cast

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

根据源操作数和目的操作数Tensor的数据类型进行精度转换，如果目的操作数是整型且源操作数的数值超过整型的数据表示范围进行精度转换结果为目的操作数的最大值或者最小值。

在了解精度转换规则之前，需要先了解浮点数的表示方式和二进制的舍入规则：

-   浮点数的表示方式
    -   DT\_FP16共16bit，包括1bit符号位（S），5bit指数位（E）和10bit尾数位（M）。

        当E不全为0或不全为1时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>E - 15</sup>  \* \(1 + M\)

        当E全为0时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>-14</sup>  \* M

        当E全为1时，若M全为0，表示的结果为±inf（取决于符号位）；若M不全为0，表示的结果为nan。

        ![](../figures/pypto.cast.png)

        上图中S=0，E=15，M = 2<sup>-1</sup>  + 2<sup>-2</sup>，表示的结果为1.75。

    -   DT\_FP32共32bit，包括1bit符号位（S），8bit指数位（E）和23bit尾数位（M）。

        当E不全为0或不全为1时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>E - 127</sup>  \* \(1 + M\)

        当E全为0时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>-126</sup>  \* M

        当E全为1时，若M全为0，表示的结果为±inf（取决于符号位）；若M不全为0，表示的结果为nan。

        ![](../figures/pypto.cast-0.png)

        上图中S = 0，E = 127，M = 2<sup>-1</sup>  + 2<sup>-2</sup>，最终表示的结果为1.75 。

    -   DT\_BF16共16bit，包括1bit符号位（S），8bit指数位（E）和7bit尾数位（M）。

        当E不全为0或不全为1时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>E - 127</sup>  \* \(1 + M\)

        当E全为0时，表示的结果为：

        \(-1\)<sup>S</sup>  \* 2<sup>-126</sup>  \* M

        当E全为1时，若M全为0，表示的结果为±inf（取决于符号位）；若M不全为0，表示的结果为nan。

        ![](../figures/pypto.cast-1.png)

        上图中S = 0，E = 127，M = 2<sup>-1</sup>  + 2<sup>-2</sup>，最终表示的结果为1.75。

-   二进制的舍入规则和十进制类似，具体如下：

    ![](../figures/pypto.cast-2.png)

    -   CAST\_RINT模式下，若待舍入部分的第一位为0，则不进位；若第一位为1且后续位不全为0，则进位；若第一位为1且后续位全为0，当M的最后一位为0则不进位，当M的最后一位为1则进位。

    -   CAST\_FLOOR模式下，若S为0，则不进位；若S为1，当待舍入部分全为0则不进位，否则，进位。
    -   CAST\_CEIL模式下，若S为1，则不进位；若S为0，当待舍入部分全为0则不进位；否则，进位。
    -   CAST\_ROUND模式下，若待舍入部分的第一位为0，则不进位；否则，进位。
    -   CAST\_TRUNC模式下，总是不进位。
    -   CAST\_ODD模式下，若待舍入部分全为0，则不进位；若待舍入部分不全为0，当M的最后一位为1则不进位，当M的最后一位为0则进位。

## 函数原型

```python
cast(input: Tensor, dtype: DataType, mode: CastMode = CastMode.CAST_NONE, 
     satmode: SaturationMode = SaturationMode.OFF) -> Tensor
```

## 参数说明


| 参数名     | 输入/输出 | 说明                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| input      | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_UINT8，DT_INT16，DT_INT32，DT_INT64。 <br> 不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dtype      | 输入      | 精度转换后的数据类型。 <br> 支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_UINT8，DT_INT16，DT_INT32，DT_INT64。 |
| CastMode   | 输入      | 源操作数枚举类型，用以控制精度转换处理模式，具体定义为：[CastMode](../datatype/CastMode.md) 。<br> 默认为 CAST_NONE，常见类型之间的转换，框架会自动转换，与torch对齐，详见约束说明。 |
| SaturationMode    | 输入      | 饱和模式枚举类型，用以控制浮点数转整数时的溢出处理方式，具体定义为：[SaturationMode](../datatype/SaturationMode.md) 。<br> 默认为 OFF（截断模式），当设置为 ON 时，超出目标类型范围的数值会被截断到最大值或最小值（饱和截断），详见约束说明。 |

## 约束说明

1.  目的操作数是整型且源操作数的数值超过整型的数据表示范围进行精度转换结果为目的操作数的最大值或者最小值。例如DT\_FP16转DT\_INT8时，若输入是130.0，将会输出127（DT\_INT8的上界）
2.  支持以下转化：
    1.  DT\_FP16 到 DT\_FP32\\DT\_INT32\\DT\_INT16\\DT\_INT8\\DT\_UINT8 转化
    2.  DT\_BF16 到 DT\_FP32\\DT\_INT32 转化
    3.  DT\_INT32 到 DT\_FP32\\DT\_INT16\\DT\_INT64 转化
    4.  DT\_FP32 到 DT\_BF16\\DT\_FP16\\DT\_INT16\\DT\_INT32\\DT\_INT64  转化
    5.  DT\_UINT8 到 DT\_FP16 转化
    6.  DT\_INT8 到 DT\_FP16 转化
    7.  DT\_INT16 到 DT\_FP32\\DT\_FP16 转化
    8.  DT\_INT64 到 DT\_FP32\\DT\_INT32 转化

3.  支持精度转换处理模式CastMode，默认处理模式如下：
    1.  DT\_FP32 -\> DT\_FP16\\DT\_BF16 : CAST\_RINT，与 Torch 对齐。
    2.  DT\_FP16\\DT\_BF16\\DT\_INT32 -\> DT\_FP32 : 与 Torch 对齐。
    3.  DT\_FP32 -\> DT\_INT32 :  CAST\_TRUNC，见约束1。
    4.  DT\_FP16 -\> DT\_TNT8:  CAST\_TRUNC，见约束1。
    5.  DT\_INT32-\> DT\_FP16 : 与 Torch 对齐。

4.  当 cast 前后类型相同的时候，某些场景下会产生空操作，不保证精度。

5.  饱和模式默认值：
    1.  FP16→UINT8、FP16→INT8、FP32→INT16、FP16→INT16、INT64→INT32、INT32→INT16 转换默认使用 OFF模式（截断/回绕），以保持PyTorch兼容性;当cast用于量化场景或有其他需求时，上述6种转换场景用户可以根据需要设置为ON模式（饱和）。
    2.  其他转换默认使用 ON模式（饱和），OFF模式不支持。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如输入intput shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1, n1分别用于切分m, n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2], pypto.DT_FP32)
y = pypto.cast(x, pypto.DT_FP16)
```

结果示例如下：

```python
输入数据x: [2.0, 3.0] # x.dtype: pypto.DT_FP32

输出数据y: [2.0, 3.0] # y.dtype: pypto.DT_FP16
```

#### 使用饱和模式（推荐用于浮点数转整数）

```python
# 示例 1：FP16 转 INT8，使用饱和模式防止溢出
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)
# 输出：[127, -128, 50] 

# 示例 2：FP16 转 INT8，使用饱和模式防止溢出
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.OFF)
# 输出：[44, -44, 50] 
```