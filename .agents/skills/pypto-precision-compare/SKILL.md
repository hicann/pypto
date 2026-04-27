---
name: pypto-precision-compare
description: PyPTO 算子精度问题调试技能。提供两种精度对比方法：文件保存方法（使用 pypto.pass_verify_save 和 torch.save）和二分对比方法（使用检查点 tensor）。当需要调试 PyPTO 算子精度、定位精度差异来源、进行中间结果对比时使用此技能。
---

# PyPTO 算子精度对比技能

提供两种精度对比方法，用于快速定位 PyPTO 算子中导致精度问题的具体 op。

## 执行逻辑

当用户调用本技能时，首先检查用户是否指定了 `mode` 参数：

### 情况一：关键词触发模式（手动干预）

如果用户描述中包含以下**关键词**，则**跳过所有判断**，直接执行对应的子技能：

| 关键词 | 触发模式 | 执行子技能 |
|--------|---------|-----------|
| **精度工具**、**精度对比**、**文件保存**、`pass_verify_save` | `mode="verify"` | `precision-verify` |
| **二分**、**上板二分**、**检查点tensor** | `mode="binary"` | `precision-binary-search` |

**示例触发语句**：
- "用精度工具方式调试" → 直接调用 `precision-verify`
- "用二分方式定位" → 直接调用 `precision-binary-search`
- "用文件保存方法对比" → 直接调用 `precision-verify`

> 注意：此时假设用户已经明确知道问题所在，不需要再执行 tensor_graph 校验流程。

### 情况二：未指定模式（全自动）

如果用户未指定 `mode`（或 `mode="auto"`），则执行**标准决策树流程**：

**步骤 1：开启 tensor_graph 校验**

**编译安装**（启用精度工具必须用 `--no-build-isolation` 参数）：

```bash
python3 -m pip install . --verbose --no-build-isolation
```

在 PyPTO 算子实现文件中配置 `verify_options`，开启 tensor_graph 校验：

```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,  # 保存中间tensor数据用于分析
    "pass_verify_pass_filter": []     # tensor_graph 校验中跳过pass校验
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(...) 
```

设置 Golden 数据（**必须严格按照计算 golden → 设置 golden → 执行算子顺序**）：

```python
# 计算 golden
torch_output = torch.add(input_data0, input_data1)

# 设置 golden（必须放在 pypto 算子运行前设置！）
pypto.set_verify_golden_data(goldens=[None, None, torch_output])

# 执行 pypto 算子
pypto_output = your_kernel(input_data0, input_data1)
```

**步骤 2：查看 tensor_graph 校验结果**

运行测试后，查看校验结果：

```bash
ls -lt output/output_*/verify_*/verify_graph_result_brief.log | head -n 1
```

打开 `verify_graph_result_brief.log` 文件，查看校验状态。

**步骤 3：根据校验结果选择调试路线**

```
查看 Tensor Graph 校验结果
    │
    ├── FAIL → 使用精度工具对比法保存中间结果比对，找出首个失败的 op
    │           参考 → precision-verify/SKILL.md
    │
    └── PASS → 上板二分定位，找到首个出错的 op
                参考 → precision-binary-search/SKILL.md
```

#### 情况 A：tensor_graph Verify FAIL → 前端代码问题

| 项目         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **日志特征** | `[VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED` |
| **错误码**   | `0xB4001U`（VERIFY_RESULT_MISMATCH）                         |
| **原因**     | 前端代码书写问题，构图阶段已出错                             |
| **处理**     | 使用 **精度工具对比法** 定位首个失败的 op                    |

**操作步骤**：

1. 阅读 [precision-verify/SKILL.md](precision-verify/SKILL.md) 了解详细步骤
2. 在 kernel 中添加 `pypto.pass_verify_save()` 调用
3. 在 golden 中添加 `torch.save()` 调用
4. 运行测试生成数据
5. 使用对比工具分析结果，找出首个不匹配的检查点

#### 情况 B：tensor_graph Verify PASS → 上板二分定位

| 项目         | 说明                                                  |
| ------------ | ----------------------------------------------------- |
| **日志特征** | tensor_graph 验证 PASS，但上板结果与 golden 不一致    |
| **原因**     | 问题可能在 Codegen 或 Machine 执行阶段                |
| **处理**     | 使用上板 dump 能力，二分打印上板数据找到首个出错的 op |

**操作步骤**：

1. 阅读 [precision-binary-search/SKILL.md](precision-binary-search/SKILL.md) 了解详细步骤
2. 修改 kernel 函数签名，添加检查点 tensor 参数
3. 修改 golden 函数，返回检查点数据
4. 修改测试函数，创建检查点 tensor 并对比
5. 从关键计算点开始，二分定位精度问题，直到找到具体出错的 op

**步骤 4：子技能执行完成后的处理**

- 如果子技能返回"已定位到问题 op" → 结束，汇报问题位置
- 如果子技能返回"需要继续二分" → 返回步骤 3，继续调用对应子技能

## 方法对比

| 特性         | 精度工具对比法                                               | 二分对比法                                       |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| **实现方式** | 使用 `pypto.pass_verify_save()` 保存到文件，使用 `torch.save()` 保存 golden | 使用检查点 tensor 作为输入参数，在内存中直接对比 |
| **适用场景** | Tensor Graph 校验失败，需要快速定位首个失败的 op             | Tensor Graph 校验通过，需要上板真实数据          |
| **循环支持** | 只保存 `idx=0` 的数据                                        | 支持保存所有循环迭代数据                         |
| **代码修改** | 不需要修改 kernel 函数签名                                   | 需要修改 kernel 函数签名，添加检查点参数         |
| **使用难度** | 简单，只需添加检查点调用                                     | 较复杂，需要管理检查点 tensor                    |


## 参考资料

- PyPTO API: `docs/api/`
- pass_verify_save API: `docs/api/others/pypto-pass_verify_save.md`