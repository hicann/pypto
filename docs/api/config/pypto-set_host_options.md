# pypto.set\_host\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

该接口是编译框架提供的运行时动态配置管理功能的核心部分，它将原本静态配置在tile\_fwk\_config.json中的参数转变为动态、可编程的指令，主要功能是控制上板流程的执行。

## 函数原型

```python
set_host_options(*, compile_stage: CompStage = pypto.CompStage.CODEGEN) -> None
```

## 参数说明


| 参数名          | 输入/输出 | 说明                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| compile_stage    | 输入      | 含义：控制编译执行的阶段 <br> 说明：<br> CODEGEN: Codegen执行GenCode后，忽略静态的上板流程，终止后续执行; <br> HOST: 仅完成Host编译，不执行Codegen编译，且终止后续执行; <br> FUNCTION: 生成TensorGraph后，终止后续执行；<br> 类型: Enum <br> 取值范围: CompStage (CODEGEN/HOST/FUNCTION) <br> 默认值: CODEGEN |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

-   类型安全：需要确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
-   作用范围：参数设置是全局性的，会影响后续所有的编译过程。

## 调用示例

```python
pypto.set_host_options(compile_stage=pypto.CompStage.CODEGEN)
```

