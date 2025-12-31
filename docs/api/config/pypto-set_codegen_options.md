# pypto.set\_codegen\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

设置codegen的选项。

## 函数原型

```python
set_codegen_options(*, support_dynamic_aligned: bool = None) -> None
```

## 参数说明


| 参数名                      | 输入/输出 | 说明                                                                 |
|-----------------------------|-----------|----------------------------------------------------------------------|
| support_dynamic_aligned     | 输入      | 含义：是否支持动态Shape。 <br> 说明： <br> 当值为True，算子生成的设备侧二进制可支持动态Shape对齐场景。 <br> 当值为False，算子生成的设备侧二进制仅支持处理动态Shape非对齐场景。 <br> 类型：bool <br> 取值范围：{True, False} <br> 默认值：False（当算子确认动态Shape，且Shape尾轴均为对齐时，可尝试打开确认是否有性能收益） <br> 影响Pass范围：无，仅影响CodeGen模块生成设备侧目标代码 |


## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

support\_dynamic\_aligned选项效果后续会通过Pass推导机制进行优化，无需用户手工设置并日落，建议用户谨慎使用。

## 调用示例

```python
pypto.set_codegen_options(support_dynamic_aligned=True)
```

