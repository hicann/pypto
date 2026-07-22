---
name: pypto-precision-overall
description: PyPTO 算子精度问题调试技能。提供场景路由式排查：根据用户问题自动选择排查路径，支持前端校验、Pass校验、特定问题排查、上板二分的自由组合。当需要调试 PyPTO 算子精度、定位精度差异来源时使用此技能。
---

# PyPTO 算子精度调试技能

提供场景路由式排查流程，根据用户问题自动选择排查路径，支持各阶段的自由组合。

## 场景路由

### 路由匹配规则

1. 用户描述中**明确包含**某条路由的触发关键词 → 直接按该路由执行
2. 用户描述**无法匹配**或**匹配到多条**路由 → 列出以下场景让用户选择：

```
请选择排查场景：
1. 全自动排查 — 从前端到上板逐层排查
2. 只怀疑前端问题 — 只做 tensor_graph 校验
3. 验证VF/同步/合轴 — 直接验证开关配置问题
4. 前端校验后排查特定问题 — tensor_graph 校验后直接查VF/同步/合轴
5. 只跑Pass对比 — 只做 Pass 层校验
6. 直接上板二分 — 直接添加检查点二分定位
```

### 路由表

| 场景 | 触发关键词 | 路由链 |
|------|-----------|--------|
| 全自动排查 | 精度问题、精度调试、精度不对 | tensor-graph → pass校验 → 特定排查 → 二分 |
| 只怀疑前端问题 | 前端校验、tensor_graph、构图问题 | tensor-graph → 结束 |
| 验证VF/同步/合轴 | VF融合、同步问题、合轴、开关验证 | 特定问题排查 → 结束 |
| 前端校验后排查特定问题 | 先查前端再用配置定界、先查构图再查开关、tensor_graph后查特定问题 | tensor-graph → 特定问题排查 |
| 只跑Pass对比 | pass校验、pass_compare、Pass层 | pass校验 → 结束 |
| 直接上板二分 | 二分、上板二分、检查点tensor | 二分 → 结束 |

## 路由链详情

### 全自动排查

```
[precision-tensor-graph]  算子写法校验 + tensor_graph 校验 + 中间 tensor 对比
       │
       ├── 定位到前端构图错误 / 首个计算结果不同的 op ──→ 结束
       │
       │ tensor_graph PASS
       ▼
[precision-pass]  PreCheck/PostCheck 全链路校验 + pass_compare 逐 Op 对比
       │
       ├── 定位到引入偏差的 Pass 和具体 Op ──→ 结束
       │
       │ 所有 Pass 校验通过
       ▼
[precision-pass]  特定问题排查（同步/VF融合/合轴等开关配置验证）
       │
       ├── 通过开关配置定位到问题 ──→ 结束
       │
       │ 仍未定位
       ▼
[precision-binary-search]  添加检查点 tensor + 二分对比上板真实数据
       │
       └── 定位到上板执行阶段首个出错 op ──→ 结束
```

执行步骤：
1. 进入 [precision-tensor-graph/SKILL.md](precision-tensor-graph/SKILL.md) 执行完整流程
2. 若 tensor_graph FAIL → 阶段二定位问题 op → 结束
3. 若 tensor_graph PASS → 进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"一、Pass 校验"
4. 若 Pass 校验定位到问题 Op → 结束
5. 若 Pass 校验全部通过 → 进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"二、特定问题排查"
6. 若特定问题定位到 → 结束
7. 若仍未定位 → 进入 [precision-binary-search/SKILL.md](precision-binary-search/SKILL.md)

### 只怀疑前端问题

```
[precision-tensor-graph]  算子写法校验 + tensor_graph 校验 + 中间 tensor 对比
       │
       ├── FAIL → 阶段二定位首个计算结果不同的 op ──→ 结束
       │
       └── PASS → 前端构图正确 ──→ 结束
```

执行步骤：
1. 进入 [precision-tensor-graph/SKILL.md](precision-tensor-graph/SKILL.md) 执行完整流程
2. 无论结果如何，流程结束

### 验证VF/同步/合轴

```
[precision-pass]  特定问题排查（直接进入，跳过 Pass 校验）
       │
       ├── 通过开关配置定位到问题 ──→ 结束
       │
       └── 未定位 ──→ 结束
```

执行步骤：
1. 直接进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"二、特定问题排查"章节
2. 跳过"一、Pass 校验"流程

### 前端校验后排查特定问题

```
[precision-tensor-graph]  算子写法校验 + tensor_graph 校验 + 中间 tensor 对比
       │
       ├── FAIL → 阶段二定位问题 op ──→ 结束
       │
       │ PASS（跳过 Pass 校验）
       ▼
[precision-pass]  特定问题排查
       │
       ├── 定位到 ──→ 结束
       │
       └── 未定位 ──→ 结束
```

执行步骤：
1. 进入 [precision-tensor-graph/SKILL.md](precision-tensor-graph/SKILL.md) 执行完整流程
2. 若 tensor_graph FAIL → 阶段二定位问题 op → 结束
3. 若 tensor_graph PASS → 跳过 Pass 校验，直接进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"二、特定问题排查"

### 只跑Pass对比

```
[precision-pass]  PreCheck/PostCheck 全链路校验 + pass_compare 逐 Op 对比
       │
       ├── 定位到问题 Op ──→ 结束
       │
       └── 未定位 ──→ 结束
```

执行步骤：
1. 直接进入 [precision-pass/SKILL.md](precision-pass/SKILL.md) 的"一、Pass 校验"章节
2. Pass 校验完成后结束，不进入特定问题排查

### 直接上板二分

```
[precision-binary-search]  添加检查点 tensor + 二分对比上板真实数据
       │
       └── 定位到上板执行阶段首个出错 op ──→ 结束
```

执行步骤：
1. 直接进入 [precision-binary-search/SKILL.md](precision-binary-search/SKILL.md) 执行完整流程

## 方法对比

| 特性 | precision-tensor-graph | precision-pass | precision-binary-search |
|------|------------------------|----------------|-------------------------|
| **定位目标** | 前端构图错误 / 首个计算不同的 op | 引入偏差的 Pass 和 Op / 开关配置问题 | 上板执行阶段首个出错 op |
| **实现方式** | `pass_verify_save()` + `torch.save()` 对比 | PreCheck/PostCheck + `pass_compare` + 开关验证 | 检查点 tensor 作为输入参数对比 |
| **代码修改** | 添加 `pass_verify_save()` 调用 | 配置 `verify_options` + `tile_fwk_config.json` | 修改 kernel 函数签名，添加检查点参数 |
| **使用难度** | 简单 | 中等 | 较复杂 |

## 参考资料

- [PyPTO API 文档](../../../docs/zh/api/)
- [pass_verify_save API](../../../docs/zh/api/others/pypto-pass_verify_save.md)
