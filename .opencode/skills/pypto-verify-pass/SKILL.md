---
name: pypto-verify
description: PyPTO 精度工具校验技能。用于分析 PyPTO 算子编译和运行时的 pass 校验结果，检查所有 pass 是否成功，并定位失败的 pass。当用户需要分析精度工具输出的 pass 校验结果、验证编译流程正确性或调试编译失败问题时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 精度工具校验技能 (PyPTO Verify)

此技能提供 PyPTO 精度工具 pass 校验结果的解析和分析功能。

## 关于精度工具

PyPTO 精度工具通过 pass 机制对算子进行多级优化和验证。每个 pass 是一个独立的优化阶段，验证算子的正确性和性能。

### Pass 校验的重要性

- 检测编译过程中的错误
- 验证每个优化阶段的正确性
- 定位编译失败的具体阶段
- 提供调试信息

## 核心原则

### 编译命令

使用以下命令编译以启用精度工具：

```bash
python3 build_ci.py -f python3 --disable_auto_execute --no_isolation
```

**重要参数说明：**
- `--disable_auto_execute`: 禁用自动执行，用于分析编译过程
- `--no_isolation`: 不使用隔离环境，便于调试

### 精度工具配置

在 jit 函数中启用精度工具：

```python
@pypto.jit(verify_options={
    "enable_pass_verify": True,        # 启用 pass 校验
    "pass_verify_save_tensor": True,    # 保存 tensor 数据
    "pass_verify_pass_filter": ["PassName"]  # 可选：过滤特定 pass
})
def user_kernel(inputs, outputs):
    ...
```

## Pass 校验结果分析

### 结果输出位置

编译后，精度工具输出通常位于：
- 编译日志中的 pass 验证信息
- 保存的 tensor 数据文件（如果 `pass_verify_save_tensor=True`）
- 各个 pass 的执行状态和错误信息

### 成功的 Pass

**成功标志：**
- Pass 执行完成，无错误
- Tensor 数据正确验证
- Pass 优化符合预期

**示例输出：**
```
[INFO] PassName: PASS
[INFO] Verification: Success
```

### 失败的 Pass

**失败标志：**
- Pass 执行异常或中断
- Tensor 验证失败
- 数据不一致

**示例输出：**
```
[ERROR] PassName: FAIL
[ERROR] Verification failed at line XX
```

## 使用流程

### 步骤 1：启用精度工具

1. 在算子代码中配置 `verify_options`
2. 在 jit 函数中设置 `enable_pass_verify=True`

### 步骤 2：编译算子

```bash
export TILE_FWK_DEVICE_ID=0
python3 build_ci.py -f python3 --disable_auto_execute --no_isolation
```

### 步骤 3：分析编译日志

查找以下关键信息：
- `[INFO]` - Pass 执行状态
- `[ERROR]` - Pass 失败信息
- `[WARN]` - 警告信息（可能影响结果）

### 步骤 4：定位失败 Pass

如果存在失败的 pass：
1. 记录失败的 pass 名称
2. 查看失败原因和错误信息
3. 检查相关 tensor 数据（如果已保存）
4. 根据 pass 功能定位代码问题

### 步骤 5：验证修复

修复问题后重新编译，确认：
- 所有 pass 执行成功
- 无错误或警告
- Tensor 数据验证通过

## Pass 常见问题

### RemoveRedundantReshape 失败

**可能原因：**
- 输入 tensor shape 不匹配
- reshape 操作逻辑错误

**排查方法：**
- 检查 `pypto.view()` 和 `pypto.reshape()` 的 shape 参数
- 确认 tensor 维度和大小

### ExpandFunction 失败

**可能原因：**
- 函数展开时的类型或维度错误
- 动态 shape 计算错误

**排查方法：**
- 检查动态 shape 计算逻辑
- 验证符号变量的使用

### DuplicateOp1 失败

**可能原因：**
- 重复操作的引用错误
- 依赖关系不正确

**排查方法：**
- 检查操作依赖关系
- 确认 tensor 生命周期管理

## Pass 校验结果输出格式

### 成功场景

```
=== Pass Verification Result ===
Total Passes: 5
Passed: 5
Failed: 0

[INFO] RemoveRedundantReshape: PASS
[INFO] ExpandFunction: PASS
[INFO] DuplicateOp1: PASS
[INFO] Vectorization: PASS
[INFO] CodeGen: PASS

Result: All passes verified successfully ✓
```

### 失败场景

```
=== Pass Verification Result ===
Total Passes: 5
Passed: 4
Failed: 1

[INFO] RemoveRedundantReshape: PASS
[INFO] ExpandFunction: PASS
[INFO] DuplicateOp1: PASS
[INFO] Vectorization: PASS
[ERROR] CodeGen: FAIL
  [ERROR] Invalid tensor shape at line 123: expected [64,64], got [32,64]

Result: Verification failed at CodeGen pass ✗
```

## 高级功能

### Pass 过滤

只验证特定的 pass：

```python
@pypto.jit(verify_options={
    "enable_pass_verify": True,
    "pass_verify_pass_filter": ["ExpandFunction", "CodeGen"]
})
```

### Tensor 数据对比

使用 `pass_verify_print` 保存关键 tensor 数据：

```python
pypto.pass_verify_print(intermediate_tensor)
```

保存的数据可用于对比 golden 数据，验证计算正确性。

## 检查清单

使用精度工具时，确保：

- [ ] 设置 `enable_pass_verify=True`
- [ ] 使用正确的编译命令
- [ ] 检查编译日志中的 pass 状态
- [ ] 定位失败 pass 并分析原因
- [ ] 验证修复后的结果

## 参考资料

- 精度工具 API: `docs/api/others/pypto-pass_verify_print.md`
- 测试用例: `python/tests/st/interface/test_verify_jit.py`
- 官方示例: `examples/`
