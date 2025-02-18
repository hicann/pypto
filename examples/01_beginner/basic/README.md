# 基础操作样例 (Basic Operations)

本样例展示了 PyPTO 的基础算子操作和典型的编程模式，非常适合初学者快速上手。

## 总览介绍

本样例涵盖了 PyPTO 编程中的核心概念，包括：
- 张量的创建与属性访问。
- 基础的逐元素算术运算（加法、乘法等）。
- 矩阵乘法（Matmul）操作。
- 激活函数（如 Sigmoid）的使用。
- 用于分块（Tiling）的 View 操作。
- 多个算子的组合使用。

## 样例代码特性

本样例突出了 PyPTO 的以下特性：
- **JIT 编译**: 使用 `@pypto.jit` 装饰器定义可以在 NPU 上执行的内核函数。
- **PyTorch 集成**: 能够直接接受 PyTorch NPU 张量作为输入输出，无缝衔接现有深度学习工作流。
- **显式 Tiling 控制**: 通过 `set_vec_tile_shapes` 和 `set_cube_tile_shapes` 手动优化硬件执行效率。
- **符号化张量**: 即使在内核函数之外，也可以定义张量的形状和类型。

## 代码结构

- **`basic_ops.py`**: 包含所有示例的主脚本。
  - `test_tensor_creation()`: 示例 1 - 张量创建。
  - `test_element_wise_operations()`: 示例 2 - 逐元素运算。
  - `test_matrix_multiplication()`: 示例 3 - 矩阵乘法。
  - `test_activation_functions()`: 示例 4 - 激活函数。
  - `test_view_operations()`: 示例 5 - View 与分块操作。
  - `test_combined_operations()`: 示例 6 - 组合运算（构建简单的线性层）。
- **`tensor_creation.py`**: 张量创建操作示例，包含 Arange、Full、数据类型等创建方法。
- **`symbolic_scalar.py`**: 符号标量（Symbolic Scalar）的使用示例。

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行脚本

```bash
# 运行所有示例
python3 basic_ops.py

# 运行特定示例（例如示例 2：逐元素运算）
python3 basic_ops.py element_wise_operations::test_element_wise_operations

# 查看所有可用示例列表
python3 basic_ops.py --list
```

## 关键代码段解析

### 1. JIT 函数定义与 Tiling 配置

```python
@pypto.jit
def element_wise_ops_kernel(a: pypto.Tensor, b: pypto.Tensor, result: pypto.Tensor) -> None:
    # 设置向量计算的分块形状
    pypto.set_vec_tile_shapes(8, 8)
    # 算子组合
    add_result = pypto.add(a, b)
    mul_result = pypto.mul(add_result, 2.0)
    result[:] = mul_result
```

### 2. 与 PyTorch 张量的转换

```python
# 将 PyTorch 张量转换为 PyPTO 张量
a_pto = pypto.from_torch(a_torch)
# 执行 JIT 函数
element_wise_ops_kernel(a_pto, b_pto, result_pto)
```

## 注意事项

- **数据类型**: 昇腾 NPU 对 FP16 和 BF16 有原生硬件加速，建议在算子开发中优先考虑这些类型。
- **分块大小**: Tiling 形状的选择会显著影响算子性能，通常应根据 NPU 架构的向量/矩阵计算单元大小来设定。
- **环境**: 确保 `torch_npu` 已正确安装并能识别到昇腾显卡。
