# API 文档 dtype 表格模板

本文件展示 pypto operation API 文档中 dtype 表格的两种格式。

## 格式 1：架构区分型（使用 npu 标签）

适用于不同产品线支持不同 dtype 集合的 operation。

```markdown
## 约束说明

1. input和other都为Tensor时，数据类型应该相同。
2. other为scalar时，若input为浮点类型，则scalar支持整型（自动转为浮点）；若input为整型，则scalar不支持浮点类型（会报错）。
3. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR/Ascend 950DT：DT_INT32，DT_FP32，DT_INT16，DT_FP16，DT_BF16，DT_UINT8，DT_INT8，DT_INT64，DT_UINT64。
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 训练系列产品/Atlas A3 推理系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 训练系列产品/Atlas A2 推理系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。
   <!-- end id6 -->
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。
```

### 产品线与架构对照

| npu 标签值 | 产品线 | 对应架构 | pypto 集合 |
|-----------|--------|---------|-----------|
| `950` | Ascend 950PR/950DT | A5 | `{OP}_A5_TYPES` |
| `A3` | Atlas A3 训练/推理系列 | A2/A3 | `{OP}_A2A3_TYPES` |
| `910b` | Atlas A2 训练/推理系列 | A2/A3 | `{OP}_A2A3_TYPES` |

### 修改规则

- 在 `950` 标签下的 dtype 列表中追加 A5 新增的 dtype
- 在 `A3` 和 `910b` 标签下的 dtype 列表中追加 A2/A3 新增的 dtype
- 保持标签格式 `<!-- npu="910b" id1 -->` ... `<!-- end id1 -->` 不变
- 不要修改 id 编号

## 格式 2：架构区分型（文本内联）

部分 operation 文档不使用 npu 标签，而是在文本中直接描述架构差异：

```markdown
## 约束说明

Tensor支持的数据类型为：Atlas A2系列产品/Atlas A3系列产品：DT_FP16，DT_FP32，DT_BF16；Atlas A5系列产品：DT_FP16，DT_FP32，DT_BF16，DT_INT16。
```

### 修改规则

- 在 `Atlas A2系列产品/Atlas A3系列产品：` 后的 dtype 列表中追加 A2/A3 新增的 dtype
- 在 `Atlas A5系列产品：` 后的 dtype 列表中追加 A5 新增的 dtype
- 使用中文逗号 `，` 分隔 dtype

## 格式 3：统一型

适用于所有架构统一支持相同 dtype 集合的 operation：

```markdown
## 约束说明

Tensor支持的数据类型为：DT_FP16，DT_FP32，DT_BF16，DT_INT8，DT_UINT8。
```

### 修改规则

- 直接在 dtype 列表末尾追加新 dtype，使用中文逗号分隔

## dtype 列表顺序建议

为了保持一致性，建议按以下顺序排列 dtype：

1. DT_INT32
2. DT_FP32
3. DT_INT16
4. DT_FP16
5. DT_BF16
6. DT_UINT8
7. DT_INT8
8. DT_INT64
9. DT_UINT64
10. DT_BOOL
11. DT_UINT16
12. DT_UINT32
13. DT_FP8E4M3
14. DT_FP8E5M2
15. DT_FP8E8M0

> 实际顺序应与源码中 `supportedTypes` 集合的顺序保持一致。如果现有文档已有特定顺序，保持一致即可。
