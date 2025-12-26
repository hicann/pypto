# pypto.frontend.dynamic

## 产品支持情况

| AI处理器类型 | 是否支持 |
|------------|:--------:|
| Ascend 910C | √ |
| Ascend 910B | √ |
| Ascend 310B | ☓ |
| Ascend 310P | ☓ |
| Ascend 910 | ☓ |

## 功能说明

`pypto.frontend.dynamic` 用于定义动态维度（Dynamic Dimension），允许张量的某些维度在运行时变化。这对于处理可变的 batch size、序列长度等场景非常有用。动态维度通常在模块级别定义，然后在 JIT 编译的内核函数的类型注解中使用。

主要应用场景：
- **动态 Batch Size**: 推理时 batch size 可能随请求数量变化
- **动态序列长度**: NLP 任务中文本序列长度不固定
- **动态图结构**: 图神经网络中节点数量可变
- **条件计算**: 根据输入形状决定计算流程

## 函数原型

```python
pypto.frontend.dynamic(name: str) -> SymbolicScalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|----------|------|
| name | 输入 | 字符串类型，动态维度的名称标识，建议使用有意义的名称如 "B" (batch), "N" (sequence length) 等 |

## 返回值说明

返回一个 SymbolicScalar ，可在张量类型注解中使用。

## 约束说明

1. 动态维度必须在模块级别定义，不能在函数内部定义
2. 定义后的动态维度变量应在 JIT 函数的类型注解中使用
3. 同一个动态维度变量可以在多个张量和多个内核函数中重复使用
4. 运行时所有使用相同动态维度变量的位置必须具有相同的实际值

## 调用示例

### 示例1: 基础用法 - 动态 Batch Size

```python
import pypto

# 模块级别定义动态维度
B = pypto.frontend.dynamic("B")
HIDDEN_SIZE = 128

@pypto.frontend.jit()
def add_bias(
    x: pypto.Tensor((B, HIDDEN_SIZE), pypto.DT_FP32),
    bias: pypto.Tensor((HIDDEN_SIZE,), pypto.DT_FP32)
) -> pypto.Tensor((B, HIDDEN_SIZE), pypto.DT_FP32):
    # 实现 add 逻辑
    # B 是动态的
    ...
    return output

# 可以用不同的 batch size 调用
x1 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
result1 = add_bias(x1, bias)  # batch=2

x2 = torch.randn(8, 128, dtype=torch.float32, device='npu:0')
result2 = add_bias(x2, bias)  # batch=8
```

### 示例2: 多个动态维度

```python
# 定义多个动态维度
B = pypto.frontend.dynamic("B")
SEQ_LEN = pypto.frontend.dynamic("SEQ_LEN")
HIDDEN = 768

@pypto.frontend.jit()
def attention_kernel(
    q: pypto.Tensor((B, SEQ_LEN, HIDDEN), pypto.DT_FP32),
    k: pypto.Tensor((B, SEQ_LEN, HIDDEN), pypto.DT_FP32),
    v: pypto.Tensor((B, SEQ_LEN, HIDDEN), pypto.DT_FP32)
) -> pypto.Tensor((B, SEQ_LEN, HIDDEN), pypto.DT_FP32):
    # 实现 attention 逻辑
    # B 和 SEQ_LEN 都是动态的
    ...
    return output

# 可以处理不同的 batch 和序列长度
output = attention_kernel(q_4_128, k_4_128, v_4_128)  # B=4, SEQ=128
output = attention_kernel(q_2_256, k_2_256, v_2_256)  # B=2, SEQ=256
```

## 最佳实践

1. **命名规范**: 使用清晰的名称如 "B" (batch), "N" (sequence), "H" (height), "W" (width)
2. **定义位置**: 始终在模块级别定义，避免在函数内部定义
3. **文档说明**: 在代码注释中说明哪些维度是动态的及其含义
4. **测试覆盖**: 测试不同的动态维度取值，确保代码的正确性
