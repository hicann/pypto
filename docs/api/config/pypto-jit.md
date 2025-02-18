# pypto.jit

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

jit 函数是一个用于装饰和优化动态函数的工具，它通过即时编译技术将 Python 函数转换为高效的计算图。jit 函数可以接受一个函数以及多个可选的配置参数，包括 codegen\_options、host\_options、pass\_options 和 runtime\_options。当 jit 被调用时，它会自动管理函数的编译和执行过程。在首次运行时，jit 会编译函数并生成执行计划，后续调用会复用编译结果以提高性能。此外，jit 还支持缓存机制，根据输入输出Tensor的形状判断是否需要重新编译，从而优化计算效率。

## 函数原型

```python
def jit(dyn_func=None,
        *,
        codegen_options=None,
        host_options=None,
        pass_options=None,
        runtime_options=None)
```

## 参数说明


| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| dyn_func          | 输入      | jit修饰的函数，需要传入pypto.Tensor作为平铺参数，用于后续构建计算图。 |
| codegen_options   | 输入      | 类型为dict[str, any]，用于设置codegen配置项，配置项参数见[参数说明](pypto-set_codegen_options.md) |
| host_options      | 输入      | 类型为dict[str, any]，用于设置host配置项，配置项参数见[参数说明](pypto-set_host_options.md) |
| pass_options      | 输入      | 类型为dict[str, any]，用于设置Pass配置项，配置项参数见[参数说明](pypto-set_pass_options.md) |
| runtime_options   | 输入      | 类型为dict[str, any]，用于设置runtime配置项，配置项参数见[参数说明](pypto-set_runtime_options.md) |

## 返回值说明

无

## 约束说明

修饰的函数传入的计算参数需为pypto.Tensor类型。

## 调用示例

无参数装饰

```python
@pypto.jit
def func(tensor1, tensor2, tensor3):
...
```

带配置装饰

```python
@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_aligned": True}
)
def func(tensor1, tensor2, tensor3):
...
```

