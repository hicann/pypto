# pypto_pro.language.SyncAllMode

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

SyncAllMode是指定sync_all同步实现模式的枚举。

## 原型定义

```python
PYPTO_DECLARE_ENUM(SyncAllMode,
    HARD,  # 使用FFTS硬件同步，不需要workspace
    SOFT   # 使用GM共享状态同步，需要workspace
)
```
