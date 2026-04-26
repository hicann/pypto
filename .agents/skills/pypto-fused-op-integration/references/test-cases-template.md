# 测试集采集流程

本文件说明如何从真实网络采集 tensor 信息，构造必须 pass 的测试用例。

---

## 采集步骤

### 1. 定位打点位置

在模型代码中找到待替换算子的调用位置。

### 2. 插入打印代码

```python
print(f"[DEBUG] input: shape={x.shape}, dtype={x.dtype}")
```

### 3. 运行采集

执行模型推理，记录真实 shape/dtype。

### 4. 创建 test_cases.json

---

## test_cases.json 格式

与 `pypto-op-develop` 统一格式，包含真实网络数据：

```json
{
  "op_name": "rms_norm",
  "source": "Qwen3-1.7B",
  "description": "RMSNorm 算子真实测试用例",
  "test_cases": [
    {
      "id": "case_001",
      "description": "prefill阶段 input_layernorm",
      "seed": 42,
      "input": {
        "hidden_states": {"shape": [1, 13, 2048], "dtype": "float16"},
        "gamma": {"shape": [2048], "dtype": "float16"},
        "eps": {"value": 1e-6}
      },
      "output": {
        "shape": [1, 13, 2048],
        "dtype": "float16"
      },
      "rtol": 1e-3,
      "atol": 1e-3
    },
    {
      "id": "case_002",
      "description": "decode阶段 q_norm",
      "seed": 42,
      "input": {
        "hidden_states": {"shape": [1, 1, 16, 128], "dtype": "float16"},
        "gamma": {"shape": [128], "dtype": "float16"},
        "eps": {"value": 1e-6}
      },
      "output": {
        "shape": [1, 1, 16, 128],
        "dtype": "float16"
      },
      "rtol": 1e-3,
      "atol": 1e-3
    }
  ]
}
```

---

## 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `op_name` | ✅ | 算子名称 |
| `source` | ✅ | 数据来源（模型名称） |
| `description` | 可选 | 整体描述 |
| `test_cases` | ✅ | 测试用例列表 |
| `id` | ✅ | 用例唯一标识 |
| `description` | 可选 | 用例描述 |
| `seed` | 可选 | 随机种子（默认42） |
| `input` | ✅ | 输入 tensor 信息 |
| `output` | ✅ | 输出 tensor 信息 |
| `rtol` | 可选 | 相对容差（默认1e-3） |
| `atol` | 可选 | 绝对容差（默认1e-3） |

---

## 使用流程

test_cases.json 与 test_{op}.py 配合使用：

```bash
python test_{op}.py              # 遍历所有用例
python test_{op}.py case_001     # 运行单个用例
python test_{op}.py --list       # 列出所有用例
```

---

## 测试文件模板

测试文件从 test_cases.json 遍历读取并执行，参考：

- `pypto-op-develop/templates/test-template.py`
- `pypto-op-develop/templates/test_cases-template.json`

---

## 文件位置

```
models/{model_name}/xxx_pto_kernels/utils/test_cases/
├── test_cases.json     # 真实 tensor 信息
└── test_{op}.py        # 精度测试（遍历读取 JSON）
```

---

## 关键原则

真实用例是必须 pass 的基准，覆盖所有调用场景。