# pypto_pro.language.AtomicType

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

原子写模式枚举，在指定GM地址上进行原子加操作。

## 原型定义

```python
PYPTO_DECLARE_ENUM(AtomicType,
    AtomicNone,   # 普通写，GM上的原有数据被源Tile数据覆盖，不作累加。
    AtomicAdd     # 原子累加写，将指定数据累加到GM中，调用前须将GM目的区域初始化。
)
```

支持的数据类型及使用约束，请参见[pypto_pro.language.store](../operation/memory_data_movement/store.md#原子累加)。
