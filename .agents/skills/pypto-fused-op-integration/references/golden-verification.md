# Golden 验证详细步骤

---

## 场景判断方法

**判断范围**：只看被替换的网络逻辑（不看整个文件）

**判断标准**：

| 场景 | 特征 | 示例 |
|------|------|------|
| **A** | 只使用基础算子 | torch.matmul + torch.softmax |
| **B** | 使用融合算子 | torch_npu.contrib.flash_attention |

---

## 场景A：未引用 torch_npu

**无需验证**：Golden 直接复制原始代码，理解无偏差，直接进入算子开发。

---

## 场景B：引用 torch_npu ★ 必须验证

### 验证流程

#### 1. 创建 test_cases_golden.json

包含 torch_npu 融合算子的参数信息：

```json
{
  "op_name": "flash_attention",
  "source": "Qwen3-1.7B",
  "description": "Golden 与 torch_npu 对比测试用例",
  "test_cases": [
    {
      "id": "case_001",
      "description": "prefill阶段 flash_attention",
      "seed": 42,
      "input": {
        "query": {"shape": [1, 13, 16, 128], "dtype": "float16"},
        "key": {"shape": [1, 13, 8, 128], "dtype": "float16"},
        "value": {"shape": [1, 13, 8, 128], "dtype": "float16"}
      },
      "torch_npu_params": {
        "scale": 0.125,
        "dropout": 0.0
      },
      "rtol": 1e-3,
      "atol": 1e-3
    }
  ]
}
```

#### 2. 创建 test_{op}_golden.py

遍历读取 test_cases_golden.json，对比 torch Golden 与 torch_npu：

```python
#!/usr/bin/env python3
# test_xxx_golden.py
import os
import sys
import json
import argparse
import torch
import numpy as np
from numpy.testing import assert_allclose

from xxx_golden import xxx_golden

def get_device_id():
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        return 0
    return int(os.environ["TILE_FWK_DEVICE_ID"])

def load_test_cases(json_path="test_cases_golden.json"):
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    with open(json_path, "r") as f:
        return json.load(f)

def run_single_case(case_data, device_id=None):
    """对比 torch Golden 与 torch_npu 融合算子。"""
    case_id = case_data["id"]
    description = case_data.get("description", "")
    
    print("=" * 60)
    print(f"Test: {case_id} — {description}")
    print("=" * 60)
    
    device = f"npu:{device_id}" if device_id else "cpu"
    
    # 构造输入数据
    torch.manual_seed(case_data.get("seed", 42))
    
    inputs = case_data["input"]
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    
    query = torch.randn(inputs["query"]["shape"], 
                        dtype=dtype_map[inputs["query"]["dtype"]], device=device)
    key = torch.randn(inputs["key"]["shape"],
                      dtype=dtype_map[inputs["key"]["dtype"]], device=device)
    value = torch.randn(inputs["value"]["shape"],
                        dtype=dtype_map[inputs["value"]["dtype"]], device=device)
    
    # torch_npu 融合算子（原始实现）
    import torch_npu
    npu_params = case_data.get("torch_npu_params", {})
    output_npu = torch_npu.contrib.flash_attention(query, key, value, **npu_params)
    
    # torch Golden（理解实现）
    output_golden = xxx_golden(query, key, value, **npu_params)
    
    # 精度对比（三段式标记）
    max_diff = np.abs(output_npu.cpu().numpy() - output_golden.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")
    
    try:
        assert_allclose(
            output_npu.cpu().numpy(),
            output_golden.cpu().numpy(),
            rtol=case_data.get("rtol", 1e-3),
            atol=case_data.get("atol", 1e-3),
        )
        print("[PRECISION_PASS] Golden 与 torch_npu 一致")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(description="Golden vs torch_npu test")
    parser.add_argument("case_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument("--json", type=str, default="test_cases_golden.json",
                        help="Test cases JSON file")
    args = parser.parse_args()
    
    test_cases = load_test_cases(args.json)
    cases = test_cases.get("test_cases", [])
    
    if args.list:
        print(f"\nTest cases from {args.json}:\n")
        for case in cases:
            print(f"  {case['id']}  — {case.get('description', '')}")
        return
    
    device_id = get_device_id()
    import torch_npu
    torch.npu.set_device(device_id)
    
    to_run = cases if not args.case_id else [c for c in cases if c["id"] == args.case_id]
    
    try:
        for case_data in to_run:
            run_single_case(case_data, device_id)
        print("\n" + "=" * 60)
        print("All Golden tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    main()
```

**规范要点**：
- 从 test_cases_golden.json 遍历读取
- 三段式标记：`[PRECISION_PASS]` / `[PRECISION_FAIL]` / `Runtime error`
- 强制使用 `assert_allclose`

#### 3. 运行验证

```bash
python test_xxx_golden.py              # 遍历所有用例
python test_xxx_golden.py case_001     # 运行单个用例
python test_xxx_golden.py --list       # 列出所有用例
```

输出 `[PRECISION_PASS]` → Golden 理解正确。

#### 4. 替换 Golden 到整网验证

```python
USE_GOLDEN = True
python run_model.py --input "测试"

assert output_normal, "整网输出异常"
print("[PASS] Golden 替换后整网输出正确")
```

---

## 验证检查点

- 场景A：无
- 场景B：
  - ✅ test_cases_golden.json 格式正确
  - ✅ test_{op}_golden.py 遍历读取 JSON
  - ✅ torch Golden 与 torch_npu 一致（diff < 1e-3）
  - ✅ 整网替换后输出正常

---

## 失败处理

| 问题 | 处理 |
|------|------|
| torch Golden 与 torch_npu 不一致 | 重新理解融合算子语义（参数、scale、偏移） |
| 整网输出异常 | 检查参数传递、dtype、shape |