# pypto.matmul

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

实现input 、mat2矩阵的矩阵乘运算，计算公式为：out = input @ mat2

-   input 、mat2为源操作数，input 为左矩阵；mat2为右矩阵
-   out 为目的操作数，存放矩阵乘结果的矩阵

## 函数原型

```python
matmul(input, mat2, out_dtype, *, a_trans = False, b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## 参数说明


| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| input             | 输入      | 表示输入左矩阵。不支持输入空Tensor。 <br> 支持的数据类型为：DT_INT8, DT_FP16, DT_BF16，DT_FP32，且左右矩阵数据类型需保持一致。 <br> 支持的矩阵维度：2维、3维、4维，且左右矩阵维度需保持一致。 <br> 输入矩阵支持的Format为：TILEOP_ND, TILEOP_NZ（DT_FP32输入不支持TILEOP_NZ格式）。 <br> 当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。 <br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐（当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐。 <br> 内轴外轴：当输入矩阵input非转置时，对应数据排布为[M, K]，此时外轴为M，内轴为K；当输入矩阵input转置时，对应数据排布为[K, M]，此时外轴为K，内轴为M； <br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐（当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐 <br> 当矩阵维度为3维或者4维时，不支持pypto.view场景。 |
| mat2              | 输入      | 表示输入右矩阵。不支持输入空Tensor。 <br> 支持的数据类型为：DT_INT8, DT_FP16, DT_BF16，DT_FP32，且左右矩阵数据类型需保持一致。 <br> 支持的矩阵维度：2维、3维、4维，且左右矩阵维度需保持一致。 <br> 输入矩阵支持的Format为：TILEOP_ND, TILEOP_NZ（DT_FP32输入不支持TILEOP_NZ格式）。 <br> 当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。 <br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐（当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐。 <br> 内轴外轴：当输入矩阵mat2非转置时，对应数据排布为[K, N]，此时外轴为K，内轴为N；当输入矩阵mat2转置时，对应数据排布为[N, K]，此时外轴为N，内轴为K； <br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐（其中当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐。 <br> 当矩阵维度为3维或者4维时，不支持pypto.view场景。 |
| out_dtype         | 输出      | 表示输出矩阵数据类型，支持DT_FP32，DT_FP16，DT_BF16, DT_INT32。 <br> - 输入矩阵数据类型为DT_FP16时，out_dtype可选DT_FP32, DT_FP16。 <br> - 输入矩阵数据类型为DT_BF16时，out_dtype可选DT_FP32, DT_BF16。 <br> - 输入矩阵数据类型为DT_INT8时，out_dtype可选DT_INT32。 <br> - 输入矩阵数据类型为DT_FP32时，out_dtype可选DT_FP32。 |
| a_trans           | 输入      | 参数a_trans表示输入左矩阵是否转置，默认为False。 |
| b_trans           | 输入      | 参数b_trans表示输入右矩阵是否转置，默认为False。 |
| c_matrix_nz       | 输入      | 参数c_matrix_nz表示输出矩阵的Format是否采用NZ格式，默认为False，当前仅支持设置False，即输出矩阵仅支持ND格式。 |
| extend_params     | 输入      | 支持bias及fixpipe的反量化功能，数据类型为字典格式。 <br> 参数：bias_tensor <br> 类型：Tensor <br> 功能说明： <br> - 输入AB矩阵数据类型为DT_FP16时，Bias矩阵数据类型可选DT_FP16和DT_FP32。 <br> - 输入AB矩阵数据类型为DT_FP32时，Bias矩阵数据类型只能为DT_FP32。 <br> - 输入AB矩阵数据类型为DT_INT8时，Bias矩阵数据类型只能为DT_INT32。 <br> - bias_tensor只支持ND格式。 <br> - bias_tensor的第一维度应置1，且N维度需要与mat2矩阵的N维度相等。 <br> - Bias不支持多核切K功能。 <br> - 仅支持矩阵维度为2维场景。 <br> 参数：scale <br> 类型：float <br> 功能说明： <br> - 输入为float类型，取1为符号位 + 8位指数位 + 10位尾数位参与运算。 <br> 参数：relu_type <br> 类型：ReLuType <br> 功能说明： <br> - 支持RELU和NO_RELU两种模式。 <br> - 仅支持矩阵维度为2维场景。 <br> 参数：scale_tensor <br> 类型：Tensor <br> 功能说明： <br> - scale_tensor输入固定为uint64_t 的Tensor。计算时会转换uint64_t为float类型的低32位bit后，取1为符号位 + 8位指数位 + 10位尾数位参与运算。 <br> - scale_tensor的第一维度必须置1，且N维度需要与mat2矩阵的N维度相等。 <br> - scale_tensor只支持ND格式。 <br> - 仅支持矩阵维度为2维场景。 |

## 返回值说明

返回值为out 矩阵（Tensor）。

## 约束说明

-   调用matmul接口前需要通过pypto.set\_cube\_tile\_shapes设置M、N、K轴上的切分大小
-   当矩阵维度为3维或者4维时，需要调用pypto.set\_vec\_tile\_shapes接口设置vector的TileShape切分，如未设置，接口内部会设置2维的vec\_tile\_shape，其值为128，128。
-   调用matmul接口的输入为调用pypto.reshape后的NZ格式时，需要调用pypto.set\_matrix\_size接口设置pypto.reshape前的输入到matmul的原始Shape的m,k,n值。
-   调用matmul接口的输入矩阵维度为3维/4维并且数据格式为NZ格式时，需要调用pypto.set\_matrix\_size接口设置输入到matmul的原始Shape的m,k,n值。

## 调用示例

```python
a1 = pypto.tensor([16, 32], pypto.DT_BF16, "tensor_a")
b1 = pypto.tensor([32, 64], pypto.DT_BF16, "tensor_b")
out1 = pypto.matmul(a1, b1, pypto.DT_BF16)

a2 = pypto.tensor((2, 16, 32), pypto.DT_FP16, "tensor_a")
b2 = pypto.tensor((2, 32, 16), pypto.DT_FP16, "tensor_b")
out2 = pypto.matmul(a2, b2, pypto.DT_FP16)

a3 = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b3 = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
out3 = pypto.matmul(a3, b3, pypto.DT_FP32)

a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
extend_params = {'bias_tensor': bias}
pypto.matmul(a, b, pypto.DT_FP32, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)
   
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
extend_params = {'scale': 0.2}
pypto.matmul(a, b, pypto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)

 a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
 b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
 extend_params = {'scale': 0.2, 'relu_type': pypto.ReLuType.RELU}
 pypto.matmul(a, b, pypto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)

 a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
 b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
 scale_tensor = pypto.tensor((1, 64), pypto.DT_UINT64, "tensor_scale")
 extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
 pypto.matmul(a, b, pypto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)
```

