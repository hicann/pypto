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

`pypto.frontend.jit` 是新前端架构中的核心装饰器，用于将 Python 函数即时编译（JIT）为高效的计算图并在 NPU 上执行。新前端不支持返回值，仅支持 in-place 修改；支持传入 torch 张量及其他类型的变量。

主要特性：
- **In-place 修改**: 内核函数通过 in-place 修改输出张量传递计算结果，不支持返回值
- **类型注解**: 在函数签名中明确指定张量的形状和数据类型
- **直接调用**: 测试时可直接传入 torch 张量及其他类型的变量，无需显式转换
- **动态形状支持**: 配合 `pypto.DYNAMIC` 支持运行时变化的维度
- **多运行模式**: 支持 NPU 和 SIM（模拟器）两种运行模式

## 函数原型

```python
@pypto.frontend.jit(
    host_options=None,
    runtime_options=None,
    codegen_options=None,
    pass_options=None
)
def kernel_function(...):
    ...
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|----------|------|
| func | 输入 | frontend.jit 修饰的函数，kernel 入口，描述计算过程，用于构建计算图。 |
| codegen_options | 输入 | 类型为 `dict[str, any]`，用于设置 codegen 配置项，配置项参数见[参数说明](./config/pypto-set_codegen_options.md)  |
| host_options | 输入 | 类型为 `dict[str, any]`，用于设置 host 配置项，配置项参数见[参数说明](./config/pypto-set_host_options.md) |
| pass_options | 输入 | 类型为 `dict[str, any]`，用于设置 Pass 配置项，配置项参数见[参数说明](./config/pypto-set_pass_options.md)  |
| runtime_options | 输入 | 类型为 `dict[str, any]`，用于设置 runtime 配置项，配置项参数见[runtime_options 参数说明](./config/pypto-jit.md#runtime_options_detail) |

## 返回值说明

返回装饰后的函数，该函数可被直接调用执行。

## 约束说明

1. 张量参数，必须使用类型注解指定为 `pypto.Tensor` 类型
2. 动态维度必须使用 `pypto.DYNAMIC` 或 `pypto.DYN` 在参数注解中标记
3. tensor format用format标记，format支持非显式标记(参考示例1中的a), 默认为pypto.TileOpFormat.TILEOP_ND;
   format显式标记时, 性能更优, 要求传入的torch tensor与pypto.Tensor声明的format一致，能获得更优的性能;
4. 张量参数在前，非张量参数（如 `scalar`、`tiling`）在后
5. 非张量参数支持 keyword 传参、位置参数、使用默认值

**pypto.Tensor[...]说明**：
- kernel函数里申明推荐使用 `pypto.Tensor[[shape], dtype]` 方括号语法，符合 Python 类型注解规范
- 也兼容旧的小括号语法 `pypto.Tensor([shape], dtype)`
- 方括号内不支持 `key=value` 形式的关键字参数（Python 语法限制），只能按位置传递或使用字典
- `pypto.Tensor[]`（空参数）不支持

## 调用示例

### 示例1: 基础使用

```python
@pypto.frontend.jit
def add_kernel(
    a: pypto.Tensor([3], pypto.DT_FP32),
    b: pypto.Tensor([3], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_NZ),
    out: pypto.Tensor([3], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


# 直接传入 torch 张量调用
x = torch.randn(3, dtype=torch.float32, device='npu:0')
y = torch.randn(3, dtype=torch.float32, device='npu:0')
result = add_kernel(x, y)
```

### 示例2: 指定运行模式

```python
# NPU 模式
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel_npu(x: pypto.Tensor):
    ...

# Cost Model 模式
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def kernel_sim(x: pypto.Tensor):
    ...
```
