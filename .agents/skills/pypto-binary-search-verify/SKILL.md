---
name: pypto-binary-search-verify
description: PyPTO 算子二分查找调试技能。利用精度工具通过二分查找方法快速定位算子精度问题，支持循环场景下的条件性数据保存。当需要调试 PyPTO 算子精度、定位精度差异来源、进行中间结果对比，或在循环中使用 pass_verify_save 时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子二分查找调试技能

通过二分查找方法快速定位 PyPTO 算子中导致精度问题的具体 op。

## 核心原理

1. **在 kernel 函数中使用 `pypto.pass_verify_save()` 保存中间结果**（循环场景使用 `cond=(idx == 0)`）
2. **在 golden 函数中使用 `numpy.tofile()` 保存中间结果**（循环场景使用 `if idx == 0:`）
3. **对比 kernel 的 `.data` 文件和 golden 的 `.bin` 文件**
4. **二分缩小范围**：结果相同→往后二分，不同→往前二分
5. **定位第一个计算结果不同的 op**
6. **插入原则**：一次性添加的检查点不要多，从关键的节点开始，插入的检查点数据切片必须一致，不一致提示用户，不再进行比对

## 核心原则

### 原则 1：插入检查点（kernel 函数）

```python
# 启用验证选项
verify_options = {"enable_pass_verify": True}

@pypto.frontend.jit(run_mode="npu", verify_options=verify_options)
def kernel(inputs, outputs):
    # 基础场景
    temp1 = pypto.compute_op1(inputs[0])
    pypto.pass_verify_save(temp1, "checkpoint1_after_op1")

    # 循环场景：只保存 idx=0 的数据
    for idx in range(batch_size):
        temp = pypto.compute_op(inputs[idx])
        pypto.pass_verify_save(temp, f"checkpoint_idx{idx}", cond=(idx == 0))
```

### 原则 2：保存 golden 中间结果

```python
def golden(inputs, outputs):
    # 基础场景
    temp1 = compute_op1(inputs[0])
    temp1.cpu().float().numpy().tofile("golden_checkpoint1_after_op1.bin")

    # 循环场景：只保存 idx=0 的数据
    for idx in range(batch_size):
        temp = compute_op(inputs[idx])
        if idx == 0:
            temp.cpu().float().numpy().tofile(f"golden_checkpoint_idx{idx}.bin")
```

**文件命名约定**：
- golden 文件：`golden_{checkpoint_name}.bin`
- jit 文件：`{checkpoint_name}_{number}.data`（自动生成）
- 名称必须匹配（去掉 `golden_` 前缀和数字后缀）
- 循环场景需在名称中包含 `idx` 信息

### 原则 3：二分查找策略

```
输入 [op1] [op2] [op3] ... [opN] 输出
  ↑                              ↑
正确                          不正确

1. 在中间位置插入输出点，对比 golden
2. 结果相同 → 问题在后面 → 往后二分
3. 结果不同 → 问题在前或此处 → 往前二分
4. 重复直到找到第一个结果不同的 op
```

## 完整工作流程

### 步骤 1：插入检查点

在 jit 和 golden 函数中插入对应的检查点（参考原则 1 和 2）。

**循环场景关键点**：
- 使用 `cond=(idx == 0)` 只保存一批数据
- 确保 kernel 和 golden 保存相同的 idx 数据
- 在检查点名称中包含 idx 信息

### 步骤 2：运行测试生成数据

```bash
python3 test_operator.py
```

### 步骤 3：对比检查点

使用通用对比工具或手动对比：

```bash
# 使用通用工具（推荐）
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -v
```

### 步骤 4：继续二分

根据对比结果：
- **匹配** → 问题在后面，在后面插入新检查点
- **不匹配** → 问题在前或此处，在前面插入新检查点

### 步骤 5：定位并修复

重复步骤 1-4，直到定位到具体的 op，然后修复问题。

## 最佳实践

### 1. 检查点命名

使用有意义的名称，反映计算步骤：
```python
# ✓ 推荐
pypto.pass_verify_save(sij, "checkpoint1_after_qk_matmul")
pypto.pass_verify_save(sij, f"checkpoint1_qk_idx{idx}", cond=(idx == 0))

# ✗ 不推荐
pypto.pass_verify_save(sij, "temp1")
```

### 2. 循环场景处理

**关键要点**：
- 使用 `cond=(idx == 0)` 确保只保存一批数据
- kernel 和 golden 必须保存相同的 idx 数据
- 避免生成过多文件

### 3. 渐进式二分

```
第1轮：输入 → 中间 → 输出（3个检查点）
  ↓ 发现中间不匹配
第2轮：在中间位置前后插入检查点（5个检查点）
  ↓ 继续缩小范围
第3轮：在问题范围内插入更多检查点
  ↓
定位到具体 op
```

### 4. 清理调试代码

修复问题后：
```bash
# 移除调试文件
rm -f golden_*.bin
rm -f output/output_*/tensor/checkpoint_*.data

# 移除调试代码
# - 删除 pypto.pass_verify_save() 调用
# - 删除 verify_options 参数
# - 删除 golden 中的 tofile() 调用
```

## 常见问题

### Q1: kernel 和 golden 保存的数据不一致

**原因**：kernel 保存 idx=0，但 golden 保存了其他 idx / golden没有切块计算

**解决**：确保两者使用相同的条件，当前如果切块方式不一致不便于对比，采用不基于精度工具的调试手段。

### Q2: 找不到检查点文件

**检查**：
- jit 代码中是否使用了 `pypto.pass_verify_save()`
- 是否设置了 `verify_options={"enable_pass_verify": True}`
- 文件命名是否符合约定

## 通用对比工具

本技能提供了通用对比脚本，自动完成检查点扫描和对比：

```bash
# 自动检测并对比所有检查点
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py

# 列出所有检查点
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py --list

# 显示详细对比
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py --verbose

# 指定工作目录
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -w models/your_operator -v
```

工具功能：
- ✓ 自动检测最新 output 目录
- ✓ 自动扫描所有检查点文件
- ✓ 智能匹配 jit 和 golden 文件
- ✓ 自动分析并给出二分建议
- ✓ 支持详细元素级对比

## 检查清单

使用二分查找调试时，确保：

- [ ] **步骤 1**：插入检查点
  - [ ] 设置 `verify_options={"enable_pass_verify": True}`
  - [ ] kernel 函数中使用 `pypto.pass_verify_save(tensor, fname)`
  - [ ] golden 函数中使用 `numpy.tofile()` 保存中间结果
  - [ ] 循环场景使用 `cond=(idx == 0)` 和 `if idx == 0:`
  - [ ] 文件命名遵循约定
- [ ] **步骤 2**：运行测试生成数据
- [ ] **步骤 3**：对比检查点
  - [ ] 使用通用工具或手动对比
  - [ ] 查看对比结果
- [ ] **步骤 4**：继续二分
  - [ ] 根据对比结果判断问题位置
  - [ ] 在问题范围内插入新的检查点
- [ ] **步骤 5**：定位并修复
  - [ ] 定位到具体的 op
  - [ ] 修复问题
  - [ ] 重新验证
  - [ ] 清理调试代码

## 参考资料

- PyPTO API: `docs/api/`
- pass_verify_save API: `docs/api/others/pypto-pass_verify_save.md`
