# 中级样例 (Intermediate Examples)

本目录包含 PyPTO 的中级开发样例，主要展示了如何构建常见的神经网络组件、自定义复杂算子以及利用 PyPTO 运行时（Runtime）的高级特性。

## 样例说明

中级样例分为以下三个主要类别：

### 1. 神经网络组件 ([nn](nn/))
- **Layer Normalization ([layer_normalization](nn/layer_normalization/))**:
    - 展示标准 LayerNorm 和 RMSNorm 的实现。
    - 涉及均值和方差的计算。
- **FFN Module ([ffn](nn/ffn/))**:
    - 实现完整的 Feed-Forward Network（前馈网络）。
    - 支持多种激活函数（ReLU, GELU, SwiGLU）。
    - 结合了矩阵乘法、逐元素加法和激活函数。

### 2. 自定义算子 ([operators](operators/))
- **Custom Activation ([activation](operators/activation/))**:
    - 展示如何组合基础算子来构建复杂的激活函数，如 SiLU (Swish), GELU, SwiGLU 和 GeGLU。
- **Softmax ([softmax](operators/softmax/))**:
    - 深入展示 Softmax 算子的手动分步实现。
    - 涉及 `exp` 计算和跨维度的 `sum` 规约。

### 3. 运行时特性 ([controflow](controflow/))
- **Dynamic Shapes ([dynamic.py](controflow/others/dynamic.py))**:
    - 展示如何处理动态 Batch Size 或序列长度。
    - 使用 `dynamic_axis` 参数进行标记。
- **Condition & Loop**:
    - `condition/condition.py`: 展示算子内部的条件分支逻辑。
    - `loop/`: 展示复杂的循环控制逻辑。
- **Kernel Input Order ([kernel_input.py](controflow/others/kernel_input.py))**:
    - 展示 JIT 内核对输入顺序的灵活性。

## 核心特性

在中级样例中，您将学习到：
- **复杂算子组合**: 如何将多个基础算子封装成具有特定功能的模块。
- **控制流**: 在 `@pypto.jit` 内核中使用 `pypto.loop` 和条件判断。
- **内存优化**: 理解如何通过 `view` 和 `inplace` 操作减少显存占用。

## 运行方法

在运行任何样例之前，请确保已配置 CANN 环境并设置了设备 ID：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export TILE_FWK_DEVICE_ID=0
```

进入对应子目录运行脚本即可。

## 学习建议

1. 首先学习 `operators/activation`，了解如何通过基础算子组合出新算子。
2. 学习 `nn/layer_normalization`，掌握涉及规约运算的规范化层实现。
3. 深入 `controflow` 目录，理解 PyPTO 在处理真实世界复杂逻辑时的强大能力。
