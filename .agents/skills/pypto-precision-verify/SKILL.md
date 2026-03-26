---
name: pypto-precision-verify
description: PyPTO 算子查找精度问题调试技能。利用精度工具通过tensor数据比对快速定位算子精度问题，支持循环场景下的条件性数据保存。当需要调试 PyPTO 算子精度、定位精度差异来源、进行中间结果对比时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子精度问题对比调试技能

利用精度工具一次性对比所有关键计算节点的中间结果，快速定位 PyPTO 算子中导致精度问题的具体 op。

## 核心原理

1. **数据匹配**：golden 和 kernel 函数的实现需对应，计算逻辑和数据切块方式需要完全一致才可以添加检查点，确定数据类型bf16需要转为fp32保存
2. **kernel 保存**：在 kernel 函数中使用 `pypto.pass_verify_save()` 保存中间结果（循环场景使用 `cond=(idx == 0)`）
3. **golden 保存**：在 golden 函数中使用 `numpy.tofile()` 保存中间结果（循环场景使用 `if idx == 0:`）
4. **数据对比**：使用对比工具检查 kernel 的 `.data` 文件和 golden 的 `.bin` 文件
5. **全量对比**：一次性开启所有关键计算节点的检查点，对比所有中间结果
6. **定位问题**：根据对比结果定位第一个计算结果不同的 op
7. **插入原则**：正常情况下一次性添加的检查点不要多，从关键的节点开始，插入的检查点数据切片必须一致
8. **数据类型**：读取数据时根据 csv 文件中的 dtype 自动判断数据类型（BF16/FP32 等），注意：numpy不支持bf16，对于bf16的检查点数据，kernel和golden都要转换为FP32类型再保存

### 检查点对齐三要素

添加检查点时，必须确保以下三点完全一致：

1. **计算语义一致**：检查点变量在 kernel 和 golden 中的计算逻辑必须相同
2. **切块大小一致**：检查点变量在 kernel 和 golden 中取的数据块大小必须相同。
3. **数据维度一致**：保存的 tensor 形状必须完全匹配

如果发现golden和kernel无法对应上检查点，请重写golden，和kernel保证计算逻辑，切块一致，再进行精度对比


## 核心原则

### 原则 1：插入检查点（kernel 函数）

```python
# 必须启用验证选项（重要，不开启无法保存数据文件）
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    "pass_verify_pass_filter": []
}

@pypto.frontend.jit(verify_options=verify_options)
def kernel(inputs, outputs):
    # 基础场景
    temp1 = pypto.compute_op1(inputs[0])
    pypto.pass_verify_save(temp1, "checkpoint1_after_op1")

    # 循环场景：只保存 idx=0 的数据
    # 多层循环使用 cond=((idx1 == 0) * (idx2 == 0)) 中间用 * 连接
    for idx in range(batch_size):
        temp = pypto.compute_op(inputs[idx])
        pypto.pass_verify_save(temp, "checkpoint_idx_$idx", cond=(idx == 0), idx=0)
```

### 原则 2：保存 golden 中间结果

```python
import os
import time

# 创建算子名称目录，在执行路径下
operator_dir = f"{operator}"
os.makedirs(operator_dir, exist_ok=True)

# 创建以时间戳命名的golden数据子目录
output_dir = os.path.join(operator_dir, "golden_data")
os.makedirs(output_dir, exist_ok=True)

def golden(inputs, outputs):
    # 基础场景
    temp1 = compute_op1(inputs[0])
    temp1.cpu().numpy().tofile(f"{output_dir}/golden_checkpoint1_after_op1.bin")

    # 循环场景：只保存 idx=0 的数据
    for idx in range(batch_size):
        temp = compute_op(inputs[idx])
        if idx == 0:
            temp.cpu().numpy().tofile(f"{output_dir}/golden_checkpoint_idx{idx}.bin")
```

**文件命名约定**：
- golden 文件： `golden_{checkpoint_name}.bin`
- jit 文件: `{checkpoint_name}_{number}.data` （自动生成）
- 名称必须匹配（去掉 `golden_` 前缀和数字后缀）
- 循环场景需在名称中包含 `idx` 信息

**检查点命名前缀规则**：
- 为保证对比时按计算顺序执行，检查点名称必须按计算顺序添加数字前缀
- 格式：`{序号}_{检查点名称}`，如 `1_sij`、`2_sij_scale`，golden格式`golden_1_sij.bin`
- 对比工具会按开头的数字排序检查点，确保按计算顺序对比
- 示例：
  ```python
  # ✓ 推荐：按计算顺序添加前缀
  pypto.pass_verify_save(sij, "1_sij_idx0", cond=(b_idx == 0))
  # golden 中对应保存
  sij.cpu().numpy().tofile("golden_1_sij_idx0.bin")
  ```

**文件结构说明**：
```
operator/                          # 算子名称目录
├── golden_data/            # golden数据目录（时间戳命名）
│   ├── golden_1_sij.bin
│   ├── golden_2_sij_scale.bin
│   └── ...
├── operator_verify_result.log    # 验证结果日志文件
└── operator_accuracy_change.png  # 精度变化折线图
```

### 原则 3：全量对比策略

```
一次性开启所有关键计算节点的检查点
对比所有中间结果，定位第一个计算结果不同的 op
```

## 易错点及修正方案

### 1. 执行路径问题

**问题**：Output 文件和bin文件生成在执行路径下，工具需要在正确的目录下执行

**修正**：
```bash
# 用 -w 参数指定工作目录
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -w /path/to/operator -v
```

### 2. 数据类型读取问题

**问题**：读取数据时要根据保存的类型读取（BF16/FP32/INT32 等）

**修正**：对比工具会根据 CSV 文件中的 dtype 自动判断数据类型：
- dtype=8: BF16 格式（2字节），转换为 FP32 进行对比
- dtype=7: FP32 格式（4字节），直接读取、

### 3. 检查点插入位置问题

**问题**：golden 和 kernel 的计算步骤不一致

**原因**：kernel 分步计算并插入多个检查点，golden 合并了某些步骤

**原则**：检查点位置要一一对应
- 如果 kernel 在 A→B→C 三个步骤后都插入检查点，golden 也需要在对应步骤保存
- 不能因为某些步骤可以合并计算就跳过中间检查点
- 保持两边计算逻辑和保存时机完全一致

### 4. 切块计算问题

**问题**：kernel 可能切块计算（如 batch 循环），golden 可能一次性计算

**原则**：保持保存的数据维度一致

**两种解决思路**：
1. **golden 保存对应的切片数据**
   - golden 一次性计算完整数据（仅此情形下可用）
   - 只保存与 kernel 对应的切片（如 `result[0:tile_b]`，根据循环层数修改）
   - 数据范围与 kernel 的 `idx=0` 切片完全一致

2. **golden 改写为和kernel实现完全一致** （推荐）
   - 根据kernel侧实现，改写golden
   - 循环方式，数据读取方式和切块方式需完全对应
   - golden 函数需要模拟 kernel 的循环结构和分块策略


### 5. 精度标准问题

**问题**：不同数据类型需要不同的容差标准

**修正**：对比工具根据数据类型自动设置容差：

| 数据类型 | dtype | rtol | atol |
|---------|-------|------|------|
| BF16 | 8 | 0.05 | 0.005 |
| FP32 | 7 | 1e-5 | 1e-5 |
| FP16 | 6 | 1e-3 | 1e-3 |
| INT32 | 3 | 1e-5 | 1e-5 |

### 6. 量化场景容差设置

**问题**：量化算子由于量化误差，需要更宽松的容差标准

**修正**：使用 `--rtol` 和 `--atol` 参数指定自定义容差：

```bash
# 量化场景推荐容差：rtol=0.0078125, atol=0.0001
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py \
    --rtol 0.0078125 \
    --atol 0.001 \
    -v
```

**说明**：
- 量化算子由于量化误差，正常情况下会有较大的精度差异
- 使用更宽松的容差可以避免误判
- 推荐值：`rtol=0.0078125, atol=0.001`
- 如果仍然有大量 FAIL，可以进一步放宽容差

### 6. 对比逻辑问题

**问题**：只看最大差异容易误判，应统计不匹配率

**修正**：对比工具使用 `np.isclose` 统计不匹配个数：
- 判断条件：不匹配个数 < 总数 * max(rtol, atol)

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

### 步骤 4：分析对比结果

根据对比结果定位问题：
- 查看哪些检查点不匹配
- 不匹配的检查点就是导致精度问题的位置
- 分析不匹配率、最大差异等指标

**⚠️ 重要检查事项**：

1. **检查点数据类型一致性**
   - **numpy BF16不支持直接保存**：kernel和golden都要转换为FP32类型再保存
   - 在kernel中使用 `pypto.cast(tensor, pypto.DT_FP32)` 转换后再调用 `pypto.pass_verify_save`
   - 在golden中使用 `.to(torch.float32)` 转换后再保存

2. **审查log文件内容**
   - 查看生成的 `*_verify_result.log` 文件
   - 如果出现 **NaN** 或 **很大的数值**（如 atol=479360）或者全0值
   - **可能原因**：保存和读取数据类型不一致
   - **解决方法**：检查检查点的保存逻辑和CSV文件中的dtype，确保golden保存的数据类型与kernel一致
   - 如果在量化类场景下中间结果略超过容差阈值，可以适当放宽，这类检查点不视作精度比对失败
   - 在一些特殊场景（torch和kernel取整方式不同），导致较大误差,这类也不视作失败

### 步骤 5：绘制精度变化图（可选）

使用绘图脚本可视化精度变化：
```bash
# 指定 operator_verify_result.log 文件路径（在operator目录下）
python3 .agents/skills/pypto-binary-search-verify/scripts/plot_accuracy.py /path/to/operator/operator_verify_result.log
```

### 步骤 6：定位并修复

根据对比结果定位到具体的 op，然后修复问题。

## 最佳实践

### 1. 检查点命名

使用有意义的名称，反映计算步骤：

### 2. 循环场景处理

**关键要点**：
- 使用 `cond=(idx == 0)` 确保只保存一批数据
- kernel 和 golden 必须保存相同的 idx 数据
- 避免生成过多文件

### 3. 全量对比模式

一次性开启所有关键计算节点的检查点，对比所有中间结果。

**使用场景**：
- 算子计算步骤较多，需要全面检查
- 已经知道大概的问题范围，需要精确定位
- 需要分析整个计算流程的精度变化

### 4. 清理调试代码

修复问题后：
```bash
# 移除调试文件
rm -f operator/golden_data/*.bin
rm -rf output/output_*

# 移除调试代码
# - 删除 pypto.pass_verify_save() 调用
# - 删除 verify_options 参数
# - 删除 golden 中的 tofile() 调用
```

## 常见问题

### Q1: kernel 和 golden 保存的数据不一致

**原因**：kernel 保存 idx=0，但 golden 保存了其他 idx / golden 没有切块计算

**解决**：确保两者使用相同的条件，参考"易错点 4：切块计算问题"

### Q2: 找不到检查点文件

**检查**：
- jit 代码中是否使用了 `pypto.pass_verify_save()`
- 是否设置了 `verify_options={"enable_pass_verify": True}`
- 文件命名是否符合约定
- 是否在正确的目录下执行对比工具

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
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -w /path/to/operator -v

# 指定 golden 文件所在目录（当 golden 文件保存在独立文件夹时使用，推荐）
python3 .agents/skills/pypto-binary-search-verify/scripts/verify_binary_search.py -w /path/to/operator -g /path/to/operator/golden_data -v

```

**工具功能**：
- ✓ 自动检测最新 output 目录
- ✓ 自动扫描所有检查点文件
- ✓ 智能匹配 jit 和 golden 文件
- ✓ 根据数据类型自动设置容差标准
- ✓ 统计不匹配率而非只看最大差异
- ✓ 自动分析并给出二分建议
- ✓ 支持详细元素级对比
- ✓ 按检查点名称开头的数字排序，确保按计算顺序对比

## 检查清单

使用二分查找调试时，确保：

- [ ] **步骤 1**：插入检查点
  - [ ] 设置 `verify_options={"enable_pass_verify": True, "pass_verify_save_tensor": True, "pass_verify_pass_filter": []}`
  - [ ] kernel 函数中使用 `pypto.pass_verify_save(tensor, fname)`
  - [ ] golden 函数中使用 `numpy.tofile()` 保存中间结果
  - [ ] 循环场景使用 `cond=(idx == 0)` 和 `if idx == 0:`
  - [ ] 文件命名遵循约定（添加数字前缀）
  - [ ] 检查点插入位置要一一对应（避免连乘 vs 分步）
  - [ ] 切块计算要保持一致（golden 切块或 kernel assemble）
  - [ ] golden 和 kernel 的切块逻辑完全一致（相同的分块参数、循环结构）
- [ ] **步骤 2**：运行测试生成数据
  - [ ] golden 数据保存到独立文件夹（以算子名称和时间戳命名）
  - [ ] 避免与之前测试的 golden 文件混淆
- [ ] **步骤 3**：对比检查点
  - [ ] 在正确目录下使用通用工具
  - [ ] 如 golden 文件在独立文件夹，使用 `--golden-dir` 参数指定
  - [ ] 查看对比结果和不匹配率
  - [ ] 确认检查点按计算顺序对比
- [ ] **步骤 4**：继续二分
  - [ ] 根据对比结果判断问题位置
  - [ ] 在问题范围内插入新的检查点
  - [ ] 保持检查点命名前缀的连续性
- [ ] **步骤 5**：定位并修复
  - [ ] 定位到具体的 op
  - [ ] 修复问题
  - [ ] 重新验证
  - [ ] 清理调试代码

## 参考资料

- PyPTO API: `docs/api/`
- pass_verify_save API: `docs/api/others/pypto-pass_verify_save.md`
