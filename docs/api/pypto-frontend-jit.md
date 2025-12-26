# pypto.frontend.jit

## 产品支持情况

| AI处理器类型 | 是否支持 |
|------------|:--------:|
| Ascend 910C | √ |
| Ascend 910B | √ |
| Ascend 310B | ☓ |
| Ascend 310P | ☓ |
| Ascend 910 | ☓ |

## 功能说明

`pypto.frontend.jit` 是新前端架构中的核心装饰器，用于将 Python 函数即时编译（JIT）为高效的计算图并在 NPU 上执行。相比旧版 `pypto.jit`，新前端提供了更现代化的函数式编程范式，支持函数返回值而非 in-place 修改，并且可以直接接受 torch 张量作为输入，无需显式转换。

主要特性：
- **函数式风格**: 内核函数通过返回值传递计算结果，而不是修改输入张量
- **类型注解**: 在函数签名中明确指定张量的形状和数据类型
- **直接调用**: 测试时可直接传入 torch 张量，无需 `pypto.from_torch()` 转换
- **动态形状支持**: 配合 `pypto.frontend.dynamic()` 支持运行时变化的维度
- **多运行模式**: 支持 NPU 和 SIM（模拟器）两种运行模式

## 函数原型

```python
@pypto.frontend.jit(
    host_options=None,
    runtime_options=None,
    codegen_options=None,
    pass_options=None
)
def kernel_function(...) -> pypto.Tensor:
    ...
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|----------|------|
| func | 输入 | frontend.jit 修饰的函数，kernel 入口，描述计算过程，用于构建计算图。 |
| codegen_options | 输入 | 类型为 `dict[str, any]`，用于设置 codegen 配置项，配置项参数见[参数说明](pypto-set_codegen_options.md)  |
| host_options | 输入 | 类型为 `dict[str, any]`，用于设置 host 配置项，配置项参数见[参数说明](pypto-set_host_options.md) |
| pass_options | 输入 | 类型为 `dict[str, any]`，用于设置 Pass 配置项，配置项参数见[参数说明](pypto-set_pass_options.md)  |
| runtime_options | 输入 | 类型为 `dict[str, any]`，用于设置 runtime 配置项，配置项参数见[参数说明](pypto-set_runtime_options.md) |

## 返回值说明

返回装饰后的函数，该函数可被直接调用执行。

## 约束说明

1. 函数参数必须使用类型注解指定为 `pypto.Tensor` 类型，并明确指定形状和数据类型
2. 动态维度必须使用 `pypto.frontend.dynamic()` 在模块级别定义

## 调用示例

### 示例1: 基础使用

```python
@pypto.frontend.jit
def add_kernel(
    a: pypto.Tensor((3,), pypto.DT_FP32),
    b: pypto.Tensor((3,), pypto.DT_FP32)
) -> pypto.Tensor((3,), pypto.DT_FP32):
    pypto.set_vec_tile_shapes(2, 8)
    out = pypto.add(a, b)
    return out

# 直接传入 torch 张量调用
x = torch.randn(3, dtype=torch.float32, device='npu:0')
y = torch.randn(3, dtype=torch.float32, device='npu:0')
result = add_kernel(x, y)
```

### 示例2: 指定运行模式

```python
# NPU 模式
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel_npu(x: pypto.Tensor) -> pypto.Tensor:
    ...

# Cost Model 模式
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def kernel_sim(x: pypto.Tensor) -> pypto.Tensor:
    ...
```
