# pypto.conv

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    ×     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    ×     |

## 功能说明

实现输入input_conv、weight完成卷积运算，支持bias参数，计算公式为：out = input_conv @ weight + bias (@表示为卷积处理)

-   input_conv 、weight、bias为源操作数；input_conv 为输入矩阵，weight为权重矩阵，bias为输入的偏置
-   out 为目的操作数，存放卷积处理结果的矩阵
-   当前暂不支持量化场景
-   当前暂不支持Relu功能

## 函数原型

```python
conv(input_conv, weight, out_dtype, strides, paddings, dilations, *, groups=1, transposed=False, output_paddings=[], extend_params=None) -> Tensor
```

## 参数说明

| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| input_conv       | 输入      | 输入特征图 Tensor。<br>不支持空 Tensor。<br>支持维度：3D（1D conv）、4D（2D conv）、5D（3D conv）。<br>支持格式：NCL、NCHW、NCDHW。<br>支持数据类型：DT_FP16、DT_BF16、DT_FP32。<br>shape 约束：各维度取值范围 [1, 1000000]。 input_conv：cin = weight：cin * groups|
| weight            | 输入      | 卷积核 Tensor。<br>维度必须与 input_conv 一致（3D/4D/5D）。<br>数据类型必须与 input_conv 一致。<br>shape 约束：各维度取值范围 [1, 1000000]。 |
| out_dtype         | 输入      | 输出 Tensor 数据类型。<br>支持：DT_FP16、DT_BF16、DT_FP32。<br>必须与 input_conv 一致；fixpipe 量化场景可单独指定。 |
| strides           | 输入      | 卷积步长，单向参数。<br>- 1D（1D conv）<br>- 2D（2D conv）<br>- 3D（3D conv）<br>取值范围：[1, 63]。 |
| paddings          | 输入      | 卷积填充，双向参数。<br>- 2D（1D conv）<br>- 4D（2D conv）<br>- 6D（3D conv）<br>取值范围：[0, 255]，且每维填充值 < 对应卷积核大小。 |
| dilations         | 输入      | 空洞卷积膨胀率，单向参数。<br>- 1D（1D conv）<br>- 2D（2D conv）<br>- 3D（3D conv）<br>取值范围：[1, 63]。 |
| groups            | 输入      | 分组卷积组数，默认 1。<br>取值范围：[1, 65535]。<br>Cin、Cout 必须可被 groups 整除。 |
| transposed        | 输入      | 是否为转置卷积（反卷积），默认 False。<br>当前暂不支持 True。 |
| output_paddings   | 输入      | 转置卷积输出端填充，仅 transposed=True 时使用。<br>当前暂不支持。 |
| extend_params     | 输入      | 扩展参数字典，支持 bias、scale、relu、scale_tensor：<br>- bias_tensor：可选的偏置张量，形状为 (C_out,)，仅支持 ND 格式，bias的数据类型必须与 input_conv 一致。<br>- scale：浮点型，per-tensor 缩放因子。<br>- scale_tensor：uint64 类型 per-channel 缩放 Tensor，shape [1, Cout]，仅 ND 格式。<br>- relu_type：激活类型，支持 RELU/NO_RELU 等。 |

## 返回值说明

返回卷积运算后的输出 Tensor：
- 1D 卷积输出 shape：(Batch, Cout, Wout)
- 2D 卷积输出 shape：(Batch, Cout, Hout, Wout)
- 3D 卷积输出 shape：(Batch, Cout, Dout, Hout, Wout)

输出 shape 各维度范围：[1, 1000000]。

## 约束说明

### 1. Shape 合法性约束
- 输入特征图（input_conv）：Batch、Cin、Hin、Win、Din 维度必须在 [1, 1000000] 范围内；
- 卷积核（weight）：Cout、Kh、Kw、Kd 维度必须在 [1, 1000000] 范围内；
- 偏置（bias_tensor）：shape 必须等于 [Cout]，否则校验失败；
- 输出特征图：H_out、W_out、D_out 维度必须在 [1, 1000000] 范围内。

### 2. 属性参数合法性约束
- 基础维度匹配约束：
  - strides 维度数必须与卷积维度匹配（2D conv 长度=2，3D conv 长度=3）；
  - dilations 维度数必须与卷积维度匹配（2D conv 长度=2，3D conv 长度=3）；
  - paddings 维度数必须为2×卷积维度（2D conv 长度=4，3D conv 长度=6）；
- 数值范围约束：
  - strides 取值范围 [1, 63]；
  - dilations 取值范围 [1, 63]；
  - paddings 取值范围 [0, 255]，且每维填充值 < 对应卷积核维度大小（如 padding_h < Kh、padding_w < Kw）；
  - groups 取值范围 [1, 65535]；
- 卷积核约束：
  - Kh ≤ 255、Kw ≤ 255；
  - Kh × Kw × 32bytes/dtype ≤ 65535；dtype为input_conv的数据类型所占bit数，如FP16是16，FP32是32等
- 通道数约束：
  - Cin（输入通道数）必须能被 groups 整除；
  - Cout（输出通道数）必须能被 groups 整除；
  - CinFmap = CinWeight × groups。

### 3. 缓存空间约束
- 调用 conv 接口前，必须通过 pypto.set_conv_tile_shapes 接口设置 L1/L0 层级的卷积 TileShape 切分大小。

### 4. 功能支持约束
- transposed=True（转置卷积）暂不支持，调用会抛出 RuntimeError；
- input_conv/weight 仅支持 DT_FP16、DT_BF16、DT_FP32 数据类型，其他类型会抛出 ValueError；
- input_conv 与 weight 的维度必须一致（如 input_conv 为 4D 则 weight 也需为 4D），否则抛出 RuntimeError。

## 调用示例

```python
# 2D 卷积基础示例（当前暂不支持动态轴切分调用）
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")

# 暂时不支持子图合并特性，需要手动关闭
pypto.set_pass_options(
    cube_l1_reuse_setting={-1: 1},
    cube_nbuffer_setting={-1: 1},
)

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1])

# 2D 卷积带 bias 和 ReLu （当前暂不支持ReLu）
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")
bias = pypto.tensor((32,), pypto.DT_FP16, "bias")
extend_params = {'bias_tensor': bias, 'relu_type': pypto.ReLuType.RELU}

# 暂时不支持子图合并特性，需要手动关闭
pypto.set_pass_options(
    cube_l1_reuse_setting={-1: 1},
    cube_nbuffer_setting={-1: 1},
)

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1],
                   extend_params=extend_params)

# 3D 卷积示例（当前暂不支持动态轴切分调用）
input_conv = pypto.tensor((1, 96, 2, 16, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 96, 1, 1, 1), pypto.DT_FP16, "weight")

# 暂时不支持子图合并特性，需要手动关闭
pypto.set_pass_options(
    cube_l1_reuse_setting={-1: 1},
    cube_nbuffer_setting={-1: 1},
)

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1, 1],
                   paddings=[0, 0, 0, 0, 0, 0],
                   dilations=[1, 1, 1])
```