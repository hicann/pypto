# pypto.experimental.set\_operation\_config

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

该接口是编译框架提供的运行时动态配置管理功能的核心部分，它将原本静态地写在配置文件tile\_fwk\_config.json中的参数转变为动态、可编程的指令。

## 函数原型

```python
set_operation_config(*, force_combine_axis: Optional[bool] = None,
                      combine_axis: Optional[bool] = None)
```

## 参数说明


| 参数名               | 输入/输出 | 说明                                                                 |
|----------------------|-----------|----------------------------------------------------------------------|
| combine_axis         | 输入      | **含义**：在代码生成阶段进行合轴优化。 <br> **说明**：例如 Reduce 输出为 (32, 1) 这类场景，为实现数据连续搬运，将其合轴为 (1, 32)，以支撑后续 ElementWise 算子随路 Broadcast 计算的优化。 <br> **类型**：bool <br> **取值范围**：{True, False} <br> **默认值**：False |
| force_combine_axis   | 输入      | **含义**：同combine_axis，不推荐使用。 |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

-   类型安全：必须确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
-   作用范围：参数设置是全局性的，会影响后续所有的编译过程。

## 调用示例

```python
pypto.set_operation_config(combine_axis=True)
pypto.set_operation_config(force_combine_axis=True)
```

