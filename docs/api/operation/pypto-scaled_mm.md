# pypto.scaled\_mm

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |

## 功能说明

实现mat_a 、mat_b矩阵的mx量化矩阵乘运算，计算公式为：out = (mat_a * scale_a) @ (mat_b * scale_b)

-   mat_a 、mat_b 、scale_a 、scale_b为源操作数，mat_a 为左矩阵；mat_b为右矩阵；scale_a为左矩阵量化参数；scale_b为右矩阵量化参数
-   out 为目的操作数，存放矩阵乘结果的矩阵

## 函数原型

```python
scaled_mm(mat_a, mat_b, out_dtype, scale_a, scale_b, *, a_trans = False, b_trans = False, scale_a_trans = False, scale_b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## 参数说明


| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| mat_a             | 输入      | 表示输入左矩阵。不支持输入空Tensor。 <br> 支持的数据类型为：DT_FP8E5M2, DT_FP8E4M3，且左右矩阵数据类型需保持一致。 <br> 支持的矩阵维度：2维。 <br> 输入矩阵支持的Format为：TILEOP_ND, TILEOP_NZ。 <br> 当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。 <br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐，外轴16元素对齐。 <br> 在满足Format约束的基础上，其Shape维度需满足K轴64元素对齐。 <br> 内轴外轴：当输入矩阵mat_a非转置时，对应数据排布为[M, K]，此时外轴为M，内轴为K；当输入矩阵mat_a转置时，对应数据排布为[K, M]，此时外轴为K，内轴为M。 <br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐，外轴16元素对齐。 |
| mat_b              | 输入      | 表示输入右矩阵。不支持输入空Tensor。 <br> 支持的数据类型为：DT_FP8E5M2, DT_FP8E4M3，且左右矩阵数据类型需保持一致。 <br> 支持的矩阵维度：2维。 <br> 输入矩阵支持的Format为：TILEOP_ND, TILEOP_NZ。 <br> 当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。 <br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐，外轴16元素对齐。 <br> 在满足Format约束的基础上，其Shape维度需满足K轴64元素对齐。 <br> 内轴外轴：当输入矩阵mat_b非转置时，对应数据排布为[K, N]，此时外轴为K，内轴为N；当输入矩阵mat_b转置时，对应数据排布为[N, K]，此时外轴为N，内轴为K。 <br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐，外轴16元素对齐。 |
| out_dtype         | 输出      | 表示输出矩阵数据类型，支持DT_FP32，DT_FP16，DT_BF16。 |
| scale_a              | 输入      | 表示输入左矩阵量化参数。不支持输入空Tensor。 <br> 支持的数据类型为：DT_FP8E8M0。 <br> 支持的量化参数维度：3维。 <br> 输入量化参数shape为：当输入量化参数非转置时，对应输入shape为[M, K/64, 2]；当输入量化参数转置时，对应输入shape为[K/64, M, 2]。其中M和K值等于输入矩阵mat_a的M、K值。<br> 输入量化参数支持的Format为：TILEOP_ND。|
| scale_b              | 输入      | 表示输入右矩阵量化参数。不支持输入空Tensor。 <br> 支持的数据类型为：DT_FP8E8M0。 <br> 支持的量化参数维度：3维。 <br> 输入量化参数shape为：当输入量化参数非转置时，对应输入shape为[K/64, N, 2]；当输入量化参数转置时，对应输入shape为[N, K/64, 2]。其中M和K值等于输入矩阵mat_a的M、K值。<br> 输入量化参数支持的Format为：TILEOP_ND。|
| a_trans           | 输入      | 参数a_trans表示输入左矩阵是否转置，默认为False。 |
| b_trans           | 输入      | 参数b_trans表示输入右矩阵是否转置，默认为False。 |
| scale_a_trans           | 输入      | 参数scale_a_trans表示输入左矩阵量化参数是否转置，默认为False。 |
| scale_b_trans           | 输入      | 参数scale_b_trans表示输入右矩阵量化参数是否转置，默认为False。 |
| c_matrix_nz       | 输入      | 参数c_matrix_nz表示输出矩阵的Format是否采用NZ格式，默认为False，当前仅支持设置False，即输出矩阵仅支持ND格式。 |
| extend_params     | 输入      | 支持bias及fixpipe的反量化功能，数据类型为字典格式。默认为None，当前仅支持设置None，即不支持bias及fixpipe的反量化功能。 |

## 返回值说明

返回值为out 矩阵（Tensor）。

## 约束说明

-   调用matmul接口前需要通过pypto.set\_cube\_tile\_shapes设置M、N、K轴上的切分大小。
-   调用matmul接口的输入为调用pypto.reshape后的NZ格式时，需要调用pypto.set\_matrix\_size接口设置pypto.reshape前的输入到matmul的原始Shape的m,k,n值。

## 调用示例

```python
mat_a = pypto.tensor([64, 128], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([128, 32], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([64, 2, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([2, 32, 2], pypto.DT_FP8E8M0, "scale_b")
out1 = pypto.scaled_mm(a1, b1, pypto.DT_BF16, scale_a, scale_b)

mat_a = pypto.tensor([64, 128], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([128, 32], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 64, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([32, 2, 2], pypto.DT_FP8E8M0, "scale_b")
out1 = pypto.scaled_mm(a1, b1, pypto.DT_BF16, scale_a, scale_b, scale_a_trans=True, scale_b_trans=True)
```

