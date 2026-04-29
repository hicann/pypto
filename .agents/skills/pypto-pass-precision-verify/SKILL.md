---
name: pypto-pass-precision-verify
description: 验证PyPTO Pass侧精度问题，定位问题来源（前端/Pass/Codegen/Machine），并指导修复。当出现以下情况时使用：(1) PyPTO算子上板结果与torch不一致；(2) 验证日志显示精度报错（tensor_graph FAIL、Pass Verify FAIL）；(3) OP报错涉及动态shape/validshape需打印上板数据验证；(4) 所有验证PASS但精度异常；(5) 用户提到Pass精度调试、精度问题定位、验证报错分析。
---

## 快速诊断

| 场景 | 错误码 | 处理方法 |
|-----|-------|---------|
| 前端问题 | `0xB4001U` | 调用 `pypto-precision-compare` 技能 |
| OP 报错 | `0xB200FU` | 检查 IR 图，动态shape需打印验证 |
| Pass精度问题 | `0xB4001U` | PreCheck/PostCheck → pass_compare → 上板比对 |
| 无报错但精度异常 | 无 | 调用 pypto-precision-compare 二分前端 |

```
精度问题 → 查看验证日志 → 按错误码选择处理流程：
├─ 0xB4001U (tensor_graph) → 前端问题 → pypto-precision-compare
├─ 0xB200FU (OP报错) → IR图分析 → 动态shape则打印验证
├─ 0xB4001U + Pass名 → Pass精度 → PreCheck/PostCheck
└─ 无报错 → 二分前端 → pypto-precision-compare
```

---

## 目录

1. [简介](#简介)
2. [环境与配置](#环境与配置)
3. [操作步骤](#操作步骤)
4. [错误码速查表](#错误码速查表)
5. [问题处理流程](#问题处理流程)
6. [打印上板信息](#打印上板信息)
7. [IR图分析](#ir图分析)
8. [注意事项](#注意事项)

---

## 简介

验证 PyPTO Pass 侧精度问题，定位问题来源（前端/Pass/Codegen/Machine）。

> `tensor_graph Verify FAIL` → 前端问题，直接调用 `pypto-precision-compare` 技能。

---

## 环境与配置

### 环境变量配置

| 环境变量 | 设置时机 | 说明 |
|---------|---------|------|
| `ASCEND_WORK_PATH` | 运行测试前必须设置 | 组件日志输出目录 |
| `ASCEND_GLOBAL_LOG_LEVEL` | 建议设为 0（DEBUG） | 获取详细调试信息 |
| `TILE_FWK_DEVICE_ID` | NPU 模式运行前必须设置 | 指定 NPU 设备 ID |

**设置示例**：
```bash
# 必需：设置工作目录
export ASCEND_WORK_PATH="/path/to/work/directory"

# 必需：设置日志级别（0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR）
export ASCEND_GLOBAL_LOG_LEVEL=0

# 可选：设置设备 ID（NPU 模式）
export TILE_FWK_DEVICE_ID=0
```

**验证环境变量**：
```bash
echo "ASCEND_WORK_PATH: $ASCEND_WORK_PATH"
echo "ASCEND_GLOBAL_LOG_LEVEL: $ASCEND_GLOBAL_LOG_LEVEL"
echo "TILE_FWK_DEVICE_ID: $TILE_FWK_DEVICE_ID"
```

---

## 操作步骤

### 步骤一：配置校验开关

#### verify_options 配置

在 PyPTO 算子实现文件中配置：

```python
verify_options = {
    "enable_pass_verify": True,            # 启用Pass验证（必须）
    "pass_verify_pass_filter": "all",      # 验证所有Pass
    "pass_verify_save_tensor": True,       # 保存中间数据
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    output: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output[:] = input0 + input1
```

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| `enable_pass_verify` | 启用 Pass 验证 | `False` |
| `pass_verify_pass_filter` | 过滤要验证的 Pass | `[]` |
| `pass_verify_save_tensor` | 保存 Pass 中间数据 | `False` |

> **注意**：Shape 必须具体，不能使用占位符；返回类型必须使用输出参数形式。

#### tile_fwk_config.json 配置

配置文件位置：`framework/src/interface/configs/tile_fwk_config.json`

```json
{
    "global": {
        "pass": {
            "default_pass_configs": {
                "print_graph": true,
                "dump_graph": true,
                "pre_check": true,    // Pass精度问题时开启
                "post_check": true    // Pass精度问题时开启
            }
        }
    }
}
```

| 配置项 | 说明 | 使用场景 |
|-------|------|---------|
| `pre_check` | Pass 执行前校验 | Pass精度问题时开启 |
| `post_check` | Pass 执行后校验 | Pass精度问题时开启 |
| `dump_graph` | 保存 IR 图 | 分析 Pass 处理结果 |
| `print_graph` | 打印 IR 图 | 快速查看图结构变化 |

> 大数据量时（Shape T>1000）：缩小shape参数、删除LOOP的unrolllist参数、增大tileshape（cube_tile_shapes、vec_tile_shapes）。

### 步骤二：编译运行

```bash
python3 -m pip install . --verbose --no-build-isolation
python3 your_test_case.py
```

输出目录：`./output/output_*`（验证数据）、`$ASCEND_WORK_PATH/log/`（日志）

### 步骤三：分析验证结果

错误码定义：`framework/src/interface/interpreter/verify_error.h`

> **判断标准**：只看 CodegenPreproc Pass 是否通过。中间 Pass 报错但 CodegenPreproc PASS → 可忽略。

---

## 错误码速查表

| 错误码 | 名称 | 阶段 | 处理方法 |
|-------|------|------|---------|
| `0xB4001U` | VERIFY_RESULT_MISMATCH | 前端/Pass | 参考[问题处理流程](#问题处理流程) |
| `0xB200FU` | RUNTIME_EXCEPTION | Pass | 检查 OP 属性，参考 IR 图 |
| `0xB0001U` | VERIFY_NOT_ENABLE | 环境 | 检查 `torch >= 2.1.0` |
| 其他 | — | 未知 | 联系开发人员 |

---

## 问题处理流程

### 情况一：tensor_graph FAIL → 调用 `pypto-precision-compare`

### 情况二：Pass级别FAIL

> **前置配置**（必须）：按照 [操作步骤-步骤一](#步骤一配置校验开关) 配置 `verify_options` 和 `tile_fwk_config.json`。

**2.1 OP报错**：对比 Before/After IR，确认是否误报。

动态shape场景：IR显示符号变量（如 `sym_15_dim_0`）→ 参考 [references/print_npu_data.md](./references/print_npu_data.md) 打印验证。

**2.2 精度问题**：

**处理流程**：
```
配置PreCheck/PostCheck → 编译运行 → 观察日志报错
    ├─ 有报错 → 终止，告知用户
    └─ 无报错 → 上板比对定位问题Pass → pass_compare定位问题Op
```

---

**上板比对定位问题Pass**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 保存前端输出 | 修改前端代码，将pypto输出保存为.pt文件 |
| 2 | 对比数据 | 使用 `compare_verify_data.py` 对比数据 |
| 3 | 判断一致性 | 确认精度工具报错的Pass输出是否与上板数据一致 |
| 4 | 二分定位Pass | 逐个Pass对比，找到与上板数据一致的Pass边界 |

**前端代码保存示例**：

```python
import torch

# 调用pypto算子
output = your_pypto_kernel(input0, input1)

# 保存pypto上板输出为.pt文件
torch.save(output, "pypto_output.pt")
```

**对比脚本使用**：

```bash
# 对比单个Pass数据
python3 scripts/compare_verify_data.py \
    --pypto-output pypto_output.pt \
    --verify-data ./output/output_xxx/verify_xxx/Pass_XX_PassName/tensor~TENSOR_xxx~PassName~xxx.data

# 对比指定Pass的所有数据文件
python3 scripts/compare_verify_data.py \
    --pypto-output pypto_output.pt \
    --verify-path ./output/output_xxx/verify_xxx/Pass_XX_PassName
```

> **脚本详情**：`scripts/compare_verify_data.py`

**精度工具数据位置**：

Pass 输出数据在 Pass 子目录中：
```
./output/output_*/verify_*/Pass_XX_PassName/tensor~TENSOR_xxx~PassName~0~0~xxxx.data
```

> **判断依据**：若Pass输出与上板数据一致 → 问题在该Pass**之后**的Pass。

---

**pass_compare.py定位问题Op**：

定位到问题Pass后，使用 `pass_compare.py` 进一步定位具体Op：

```bash
python3 tools/verifier/pass_compare.py \
    --p <FailedPass> <GoldenPass> \
    --verify_path=/path/to/verify_data
```

| 参数 | 说明 |
|-----|------|
| `<FailedPass>` | 上板比对定位出的问题Pass名称 |
| `<GoldenPass>` | 问题Pass的前一个Pass（输出与上板一致的Pass） |
| `--verify_path` | 验证数据目录路径 |

> **定位结果**：pass_compare.py会输出有差异的Op列表，即为问题Op。

### 情况三：无报错但精度异常 → 调用 `pypto-precision-compare`

定位问题Op后，如需打印上板数据验证 → 参考 [references/print_npu_data.md](./references/print_npu_data.md)

---

## 打印上板信息

**前置条件**：已通过 `pypto-precision-compare` 定位到具体Op。

用于：打印上板tensor数据、验证动态shape/offset值。

详细方法请参考：**[references/print_npu_data.md](./references/print_npu_data.md)**

### 打印环境配置

**tile_fwk_config.json 配置**：

```json
{
    "global": {
        "codegen": {
            "fixed_output_path": true,    // 固定CCE输出路径
            "force_overwrite": false,     // 不覆盖已修改的CCE文件
            "parallel_compile": 1         // 单线程编译
        }
    }
}
```

| 配置项 | 正确值 | 说明 |
|-------|-------|------|
| `fixed_output_path` | `true` | CCE固定生成在 `./kernel_aicore/` |
| `force_overwrite` | `false` | 不覆盖手动修改的CCE文件 |
| `parallel_compile` | `1` | 单线程编译，便于调试 |

**aicore_print.h 打印开关**：

确保 `framework/src/interface/machine/device/tilefwk/aicore_print.h` 中：
```c
#define ENABLE_AICORE_PRINT 1   // 必须为 1
```

### 可打印内容

| 内容 | 方法 | 说明 |
|-----|------|------|
| GM tensor数据 | `AiCorePrintGmTensor` | DDR/GM上的tensor |
| UB tensor数据 | `AiCorePrintUbTensor` | UB上的tensor |
| Shape变量值 | `AicoreLogF` | 动态shape实际值 |
| Offset值 | `AicoreLogF` | 动态offset实际值 |

### 脚本工具

```bash
# 初始化配置
python3 scripts/print_npu_data.py --init --work-path /path/to/work

# 列出CCE文件
python3 scripts/print_npu_data.py --work-path /path/to/work --list-cce

# 打印tensor数据
python3 scripts/print_npu_data.py --work-path /path/to/work --print-idx 0 --tensor gmTensor_4

# 打印shape值
python3 scripts/print_npu_data.py --work-path /path/to/work --print-idx 0 --print-shape sym_15_dim_0
```

详见：[scripts/print_npu_data.py](./scripts/print_npu_data.py)

---

## IR图分析

判断误报、辅助定位。详见 `pypto/.agents/skills/pypto-pass-error-locator/references/ir-analysis-guide.md`

```bash
# 查询OP详情
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file <IR文件> --op-magic <ID>

# 列出所有OP
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file <IR文件> --list-ops
```

---

## 注意事项

1. Pass精度判断：只看 CodegenPreproc 是否通过
2. tensor_graph FAIL → 调用 pypto-precision-compare
3. 无报错但精度异常 → 调用 pypto-precision-compare
4. 动态shape验证：参考 references/print_npu_data.md
5. 打印配置：`fixed_output_path=true`, `force_overwrite=false`
6. 打印限制：元素数量 ≤ 80
7. 配置备份：修改配置前建议备份原文件

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [references/print_npu_data.md](./references/print_npu_data.md) | 打印上板信息指南 |
| [scripts/print_npu_data.py](./scripts/print_npu_data.py) | 打印上板信息脚本 |
| [scripts/compare_verify_data.py](./scripts/compare_verify_data.py) | 数据对比脚本 |