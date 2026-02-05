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
set_host_options(*, compile_stage: Optional[CompStage] = None) -> None
```

## 参数说明


| 参数名          | 输入/输出 | 说明                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| compile_stage    | 输入      | 含义：控制编译执行的阶段 <br> 说明：<br> ALL_COMPLETE: 无影响，正常编译与运行; <br> TENSOR_GRAPH: 编译阶段，生成最终张量图后停止; <br> TILE_GRAPH: 编译阶段，生成最终分片图后终止；<br> EXECUTE_GRAPH: 编译阶段，生成最终执行图后终止；<br> CODEGEN_INSTRUCTION: 编译阶段，生成指令代码后终止；<br> CODEGEN_BINARY: 编译生成代码二进制后终止, 编译阶段结束。 <br> 取值范围: CompStage (ALL_COMPLETE/TENSOR_GRAPH/TILE_GRAPH/EXECUTE_GRAPH/CODEGEN_INSTRUCTION/CODEGEN_BINARY) <br> 默认值: ALL_COMPLETE |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

-   类型安全：需要确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
-   作用范围：参数设置是全局性的，会影响后续所有的编译过程。

## 调用示例

```python
pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)
```

