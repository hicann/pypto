# pypto.set\_debug\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

设置debug的选项。

## 函数原型

```python
set_debug_options(*,
                  compile_debug_mode: Optional[int] = None,
                  runtime_debug_mode: Optional[int] = None,
                  ) -> None
```

## 参数说明


| 参数名               | 输入/输出 | 说明                                                                 |
|----------------------|-----------|----------------------------------------------------------------------|
| compile_debug_mode   | 输入      | 含义：设置编译阶段调试模式 <br> 说明：0：代表默认不使能编译阶段调试模式； <br> 1：代表图使能编译阶段调试模式，一键开启图编译相关配置，当前进包括计算图； <br> 类型：int <br> 取值范围：0 或 1 <br> 默认值：0 <br> 影响Pass范围：NA |
| runtime_debug_mode   | 输入      | 含义：设置执行阶段调试模式 <br> 说明：0：代表默认不使能执行阶段调试模式； <br> 1：代表使能执行阶段调试模式，一键开启图执行相关配置，当前仅包括泳道图； <br> 类型：int <br> 取值范围：0 或 1 <br> 默认值：0 <br> 影响Pass范围：NA |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

无。

## 调用示例

```python
pypto.set_debug_options(compile_debug_mode=1)
pypto.set_debug_options(runtime_debug_mode=1)
```

