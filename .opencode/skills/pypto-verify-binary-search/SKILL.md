# Skill: pypto-verify-binary-search

# PyPTO 算子二分查找调试技能 (PyPTO Verify Binary Search)

此技能提供通过二分查找方法定位 PyPTO 算子中导致精度问题的具体 op 的功能。

## 核心原理

当算子输出结果与 golden 不匹配时,使用二分查找方法快速定位问题:

1. **在 jit 函数中使用 `pypto.pass_verify_save()` 保存中间结果**
2. **在 golden 函数中使用 `numpy.tofile()` 保存中间结果**
3. **从 jit 生成的 `.data` 文件和 golden 生成的 `.bin` 文件中读取数据并对比**
4. **二分缩小范围**：结果相同往后二分,不同往前二分
5. **定位第一个计算结果不同的 op**

**关键要求**：
- jit 函数使用 `pypto.pass_verify_save(tensor, fname)` 保存中间结果到文件
- golden 函数使用 `numpy.tofile()` 保存中间结果到文件
- 必须在 `@pypto.jit` 装饰器中设置 `verify_options={"enable_pass_verify": True}`
- `pass_verify_save()` 会自动保存数据到 `output/output_*/tensor/` 目录
- `pass_verify_save(fname)` 生成两个文件：`{fname}.data`（数据）和 `{fname}.csv`（元数据）
- 对比时从 jit 的 `.data` 文件和 golden 的 `.bin` 文件中读取数据进行比对

## 核心原则

### 原则 1：先验证整体结果，正确则结束调试

开始二分查找前，必须先验证整体结果：
- 运行测试，对比算子输出和 golden 输出
- **如果整体结果匹配**（在容差范围内）→ ✓ 结束调试，不需要二分查找
- **如果整体结果不匹配**（超出容差或存在异常）→ ✗ 继续进行二分查找

**重要**：如果整体结果正确，说明实现没有问题，不需要进行二分查找。二分查找仅用于定位导致精度问题的具体 op。

### 原则 2：插入中间输出点

在 jit 函数的关键计算位置使用精度工具输出中间结果：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    # 计算步骤 1
    temp1 = pypto.compute_op1(inputs[0])
    # 插入中间输出
    pypto.pass_verify_save(temp1, "checkpoint1")

    # 计算步骤 2
    temp2 = pypto.compute_op2(temp1)
    pypto.pass_verify_save(temp2, "checkpoint2")

    # ... 后续步骤
```

**重要说明**：
- 必须设置 `verify_options={"enable_pass_verify": True}`
- `pypto.pass_verify_save(tensor, fname)` 需要提供文件名前缀
- 数据保存路径：`${work_path}/output/output_*/tensor/`
- 生成文件：`{fname}.data`（数据）和 `{fname}.csv`（元数据）

### 原则 3：二分查找策略

按照以下策略进行二分查找：

```
输入 [op1] [op2] [op3] ... [opN] 输出
  ↑                              ↑
正确                          不正确

1. 在中间位置插入输出点，对比 golden
2. 如果结果相同 → 问题在中间位置之后 → 往后二分
3. 如果结果不同 → 问题在中间位置之前或此处 → 往前二分
4. 重复直到找到第一个结果不同的 op
```

## 二分查找流程

### 步骤 0：验证整体结果

**重要**：在开始二分查找前，必须先验证整体结果的正确性。

1. **运行测试，比较 jit 输出和 golden 输出**

   使用测试框架或手动对比来验证整体输出：

   ```python
   # 示例：使用 numpy 或 PyTorch 对比整体结果
   import numpy as np
   from utils.np_compare import detailed_allclose_manual as compare

   # 运行 算子 和 golden
   jit_output, jit_residual = jit_function(...)
   golden_output, golden_residual = golden_function(...)

   # 对比结果
   is_output_match = np.allclose(
       np.array(jit_output.flatten().tolist()),
       np.array(golden_output.flatten().tolist()),
       rtol=0.003,
       atol=0.003
   )
   is_residual_match = np.allclose(
       np.array(jit_residual.flatten().tolist()),
       np.array(golden_residual.flatten().tolist()),
       rtol=0.001,
       atol=0.001
   )

   print(f"Output match: {is_output_match}")
   print(f"Residual match: {is_residual_match}")
   ```

2. **判断是否需要二分**

   - **如果整体结果匹配**：
     - ✓ 所有输出都在容差范围内
     - ✓ 没有异常元素（NaN、超出容差）
     - → **结束调试**，不需要进行二分查找
     - → 说明实现正确

   - **如果整体结果不匹配**：
     - ✗ 至少一个输出超出容差
     - ✗ 存在异常元素
     - → **继续执行二分查找**，定位问题 op

3. **记录验证结果**

   ```python
   # 日志记录
   print("=" * 80)
   print("步骤 0：验证整体结果")
   print("=" * 80)

   if is_output_match and is_residual_match:
       print("✓ 整体结果正确，不需要二分查找")
       print("→ 调试结束")
   else:
       print("✗ 整体结果不匹配，开始二分查找")
       if not is_output_match:
           print(f"  - Attention output 不匹配")
       if not is_residual_match:
           print(f"  - Residual 不匹配")
       print("→ 继续：步骤 1")
   print("=" * 80)
   ```

**关键点**：
- 如果整体结果正确，直接结束调试，不要浪费时间进行二分
- 只有确认整体结果不匹配时，才进行后续的二分查找步骤
- 根据哪个输出不匹配，优先关注相关的计算路径

### 步骤 1：准备 golden 中间结果

在 golden 函数中将中间结果保存到文件：

```python
import numpy as np
import torch

def golden(inputs, outputs):
    # 计算步骤 1
    temp1 = compute_op1(inputs[0])
    # 保存到文件
    if isinstance(temp1, torch.Tensor):
        temp1_numpy = temp1.cpu().to(torch.float32).numpy()
    else:
        temp1_numpy = temp1
    temp1_numpy.tofile("golden_checkpoint1.bin")

    # 计算步骤 2
    temp2 = compute_op2(temp1)
    # 保存到文件
    if isinstance(temp2, torch.Tensor):
        temp2_numpy = temp2.cpu().to(torch.float32).numpy()
    else:
        temp2_numpy = temp2
    temp2_numpy.tofile("golden_checkpoint2.bin")

    # ... 后续步骤
```

运行 golden，中间结果会保存到 `.bin` 文件中。

**重要说明**：
- **golden 函数**（PyTorch/NumPy 实现）使用 `numpy.tofile()` 保存中间结果到 `.bin` 文件
- **jit 函数**（PyPTO 算子）使用 `pypto.pass_verify_save(tensor, fname)` 会自动保存数据文件
- jit 的数据文件保存在 `output/output_<timestamp>/tensor/<function_name>/` 路径下，文件名格式为 `{fname}.data`

### 步骤 2：jit 函数中插入中间输出点

在 jit 函数的中间位置插入输出点：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    # 前半部分计算
    temp_mid = pypto.compute_op_mid(...)

    # 中间点输出
    pypto.pass_verify_save(temp_mid, "checkpoint_mid")

    # 后半部分计算
    ...
```

**重要说明**：
- `pypto.pass_verify_save(tensor, fname)` 需要提供文件名前缀
- 数据文件路径：`output/output_<timestamp>/tensor/<function_name>/`
- 生成两个文件：`{fname}.data`（数据）和 `{fname}.csv`（元数据）
- 需要在 jit 函数装饰器中设置 `verify_options={"enable_pass_verify": True}`

### 步骤 3：查找并读取 jit 输出文件

运行 jit 函数后，`pass_verify_save()` 生成的数据文件保存在 `output/output_<timestamp>/tensor/<function_name>/` 目录下。

**查找方法**：

```bash
# 1. 找到最新的output目录
ls -lt output/ | head -2

# 2. 查找tensor目录下的所有.data文件
find output/output_<timestamp>/tensor/ -name "*.data" | head -20
```

**文件命名规则**：
- 格式：`{fname}.data`（数据文件）
- 示例：`checkpoint_mid.data`

**读取方法**：

```python
import numpy as np

def read_jit_data(filename):
    """读取 jit 生成的数据文件

    Args:
        filename: 数据文件路径

    Returns:
        numpy array
    """
    data = np.fromfile(filename, dtype=np.float32)
    print(f"Read {filename}: shape={data.shape}")
    return data

# 示例：读取最新的jit数据文件
import os
import subprocess

# 获取最新output目录
result = subprocess.run(['ls', '-lt', 'output/'], capture_output=True, text=True)
if result.returncode == 0:
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        latest_output_dir = lines[1].split()[-1]
        tensor_dir = f"output/{latest_output_dir}/tensor/kernel/"

        # 读取checkpoint_mid.data
        jit_data = read_jit_data(f"{tensor_dir}/checkpoint_mid.data")
```

### 步骤 4：对比 jit 和 golden 数据

读取 jit 的 `.data` 文件和 golden 的 `.bin` 文件并进行对比：

```python
import numpy as np

def compare_with_golden(jit_data, golden_data, name, rtol=1e-3, atol=1e-3):
    """对比 jit 结果与 golden 结果

    Args:
        jit_data: 从 pass_verify_save 生成的 .data 文件读取的 jit 数据
        golden_data: 从 golden .bin 文件读取的 golden 数据
        name: 检查点名称
        rtol: 相对容忍度
        atol: 绝对容忍度

    Returns:
        bool: 是否匹配
    """
    # 确保形状一致
    min_size = min(jit_data.shape[0], golden_data.shape[0])
    jit_data = jit_data[:min_size]
    golden_data = golden_data[:min_size]

    diff = np.max(np.abs(jit_data - golden_data))
    max_val = np.max(np.abs(golden_data))
    relative_error = diff / (max_val + 1e-10)

    match = relative_error < rtol and diff < atol

    status = "✓ PASS" if match else "✗ FAIL"
    print(f"\n{name}: {status}")
    print(f"  Max diff: {diff}")
    print(f"  Max val: {max_val}")
    print(f"  Relative error: {relative_error}")
    print(f"  Tolerance: rtol={rtol}, atol={atol}")

    return match

# 示例：对比checkpoint 1
golden_data1 = np.fromfile("golden_checkpoint1.bin", dtype=np.float32)
jit_data1 = np.fromfile("output/output_<timestamp>/tensor/kernel/checkpoint1.data",
                      dtype=np.float32)
is_match1 = compare_with_golden(jit_data1, golden_data1, "Checkpoint 1", rtol=1e-3, atol=1e-3)

# 示例：对比checkpoint 2
golden_data2 = np.fromfile("golden_checkpoint2.bin", dtype=np.float32)
jit_data2 = np.fromfile("output/output_<timestamp>/tensor/kernel/checkpoint2.data",
                      dtype=np.float32)
is_match2 = compare_with_golden(jit_data2, golden_data2, "Checkpoint 2", rtol=1e-3, atol=1e-3)
```

**判断依据**：
- 如果 **Checkpoint 1** 匹配 golden → 问题在 Checkpoint 1 之后
- 如果 **Checkpoint 1** 不匹配 golden → 问题在 Checkpoint 1 之前或此处

### 步骤 5：根据结果继续二分

#### 场景 A：中间点匹配 golden

说明问题在中间点之后，在后半部分继续二分：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    # 前半部分（已验证正确）
    temp_mid = pypto.compute_op_mid(...)

    # 在后半部分的中间插入新输出点
    temp_mid2 = pypto.compute_op_mid2(temp_mid)
    pypto.pass_verify_save(temp_mid2, "checkpoint_mid2")

    # 继续后续计算
    ...
```

#### 场景 B：中间点不匹配 golden

说明问题在中间点之前或此处，在前半部分继续二分：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    # 在前半部分的中间插入新输出点
    temp_mid_early = pypto.compute_op_mid_early(...)
    pypto.pass_verify_save(temp_mid_early, "checkpoint_early")

    # 继续计算到中间点
    temp_mid = pypto.compute_op_mid(temp_mid_early)
    pypto.pass_verify_save(temp_mid, "checkpoint_mid")
    ...
```

## 优化技巧

### 技巧 1：减少输出点

只输出需要对比的中间结果，避免过多输出影响性能：

```python
# 输出当前二分轮次需要的点，其他点注释掉
# pypto.pass_verify_save(temp1, "checkpoint1")
pypto.pass_verify_save(temp_mid, "checkpoint_mid")
```

### 技巧 2：使用条件输出

通过 `cond` 参数控制输出条件：

```python
@pypto.jit(run_mode="npu", verify_options={"enable_pass_verify": True})
def jit_kernel(inputs, outputs):
    # 计算各步骤
    ...

    # 只在特定条件下输出（例如只输出第一个元素）
    cond = (idx == 0)
    pypto.pass_verify_save(temp_mid, "checkpoint_mid", cond=cond)
```

### 技巧 3：减少输出文件

由于 `pass_verify_save()` 会生成多个数据文件，需要注意：
- 只在需要对比的关键位置保存数据
- 可以使用 `cond` 参数只保存部分元素
- 每轮二分后可以清理旧的输出文件，避免文件过多

```bash
# 只输出第一个元素
pypto.pass_verify_save(tensor, "checkpoint", cond=(idx == 0))

# 清理旧的输出文件
rm -f output/output_*/tensor/kernel/checkpoint_*.data
```

## 常见问题

### Q1: 中间结果太大无法输出

**解决方法**：
- 使用 `cond` 参数只输出部分元素
- 输出统计信息而不是完整数据

```python
# 只输出第一个元素
pypto.pass_verify_save(tensor, "checkpoint", cond=(idx == 0))

# 或者使用条件判断
cond = (pypto.greater(tensor, threshold).sum() > 0)
pypto.pass_verify_save(tensor, "checkpoint", cond=cond)
```

### Q2: op 太多，二分效率低

**解决方法**：
- 先根据代码逻辑划分大块，对每个块进行二分
- 优先检查可疑的 op（例如复杂的数学运算、类型转换等）

### Q3: 多个 op 都有问题

**解决方法**：
- 找到第一个有问题的 op 并修复后，重新运行
- 继续二分查找下一个有问题的 op
- 重复直到所有问题解决

## 检查清单

使用二分查找调试时，确保：

- [ ] 先验证整体结果
  - [ ] 如果整体结果正确 → 结束调试，不需要二分查找
  - [ ] 如果整体结果不正确 → 继续执行后续步骤
- [ ] 在 jit 函数装饰器中设置 `verify_options={"enable_pass_verify": True}`
- [ ] 在 jit 函数中使用 `pypto.pass_verify_save(tensor, fname)` 输出中间结果
- [ ] 在 golden 函数中使用 `numpy.tofile()` 保存中间结果到文件
- [ ] 每轮只插入必要的中间输出点
- [ ] 查找 `output/output_<timestamp>/tensor/kernel/` 目录下的 `.data` 文件
- [ ] 读取 jit 的 `.data` 文件和 golden 的 `.bin` 文件
- [ ] 使用相对误差和绝对误差进行对比
- [ ] 根据对比结果判断问题位置（相同→往后二分，不同→往前二分）
- [ ] 记录每轮二分的日志
- [ ] 定位问题后检查相关 op 的实现
- [ ] 修复后重新验证

## 参考资料

- PyPTO API: `docs/api/`
- pass_verify_save API: `docs/api/others/pypto-pass_verify_save.md`
- 测试框架: `python/tests/st/interface/`
