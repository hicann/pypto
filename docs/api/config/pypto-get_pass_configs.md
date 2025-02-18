# pypto.get\_pass\_configs

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取指定Pass的全部配置信息。

## 函数原型

```python
get_pass_configs(strategy: str, identifier: str) -> PassConfigs
```

## 参数说明


| 参数名     | 输入/输出 | 说明                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| strategy   | 输入      | Pass 策略名称，如"PVC2_OOO"。 |
| identifier | 输入      | Pass 名称，如"ExpandFunction"。 |

## 返回值说明

PassConfigs 对象，该对象含有以下属性，属性只读：


| 属性               | 说明                                                                 |
|--------------------|----------------------------------------------------------------------|
| printGraph         | dump计算图ir。 |
| dumpGraph          | dump 计算图。 |
| dumpPassTimeCost   | dump Pass耗时。 |
| preCheck           | Pass执行前进行校验。 |
| postCheck          | Pass执行后进行校验。 |
| disablePass        | 不执行当前Pass。 |
| healthCheck        | 执行健康检查并生成报告。 |

## 约束说明

无

## 调用示例

```python
pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
```

