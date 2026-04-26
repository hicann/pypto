# transformers 缓存自动同步模板

本文件包含处理 transformers trust_remote_code 缓存机制的完整解决方案。

---

## 问题诊断

在 modeling_xxx.py 导入时添加调试打印：

```python
print(f"[Debug] __file__ = {__file__}")
# 如果输出是缓存路径（如 ~/.cache/huggingface/modules/transformers_modules/xxx/）
# 则存在缓存问题，本地修改不生效
```

**典型缓存路径**：
- `~/.cache/huggingface/modules/transformers_modules/Qwen3_hyphen_1_dot_7B/`
- `~/.cache/huggingface/modules/transformers_modules/LLaMA_hyphen_7B/`

---

## 完整解决方案代码

在 modeling_xxx.py 导入部分添加自动同步逻辑：

```python
import shutil
import os

# 获取缓存目录和本地目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
cached_pto_kernels = os.path.join(parent_dir, "xxx_pto_kernels")  # 缓存中的算子目录
local_model_dir = os.environ.get("MODEL_DIR", "/path/to/local/model")  # 从环境变量获取
local_pto_kernels = os.path.join(local_model_dir, "xxx_pto_kernels")

# 自动同步
if not os.path.exists(cached_pto_kernels) and os.path.exists(local_pto_kernels):
    shutil.copytree(local_pto_kernels, cached_pto_kernels)
    print(f"[Info] 已同步算子库到缓存")
elif os.path.exists(local_pto_kernels):
    # 检查修改时间，本地更新则覆盖缓存
    local_impl = os.path.join(local_pto_kernels, "xxx_impl.py")
    cached_impl = os.path.join(cached_pto_kernels, "xxx_impl.py")
    if os.path.getmtime(local_impl) > os.path.getmtime(cached_impl):
        shutil.copytree(local_pto_kernels, cached_pto_kernels, dirs_exist_ok=True)
        print(f"[Info] 已更新缓存（本地算子已更新）")
```

---

## 环境变量配置

通过环境变量指定本地模型目录：

```bash
export MODEL_DIR=/npu/s00454010/models/Qwen3-1.7B
```

或在代码中硬编码：

```python
local_model_dir = "/npu/s00454010/models/Qwen3-1.7B"
```

---

## 失败处理

| 问题类型 | 处理方法 |
|---------|---------|
| 缓存目录不存在 | 自动复制（首次同步） |
| 本地代码更新未生效 | 检查修改时间，覆盖缓存 |
| 权限不足 | 确认缓存目录可写（`chmod 755 ~/.cache/huggingface/modules/`） |
| 算子库目录名不匹配 | 确认目录名一致性（如 `qwen_pto_kernels`） |

---

## 适用场景

- 使用 transformers 内置模型（Qwen3、LLaMA、GLM等）
- 通过 `trust_remote_code=True` 加载
- 需要替换模型代码中的算子实现

---

## 验证检查点

- ✅ 导入时打印显示 `[Info] 已同步` 或 `[Info] 已更新`
- ✅ 修改本地算子代码后运行，自动生效
- ✅ `__file__` 显示的路径中存在 `xxx_pto_kernels` 目录