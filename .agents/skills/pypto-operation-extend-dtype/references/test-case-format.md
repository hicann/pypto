# 测试用例格式规范

本文件描述 pypto operation ST 测试的 CSV 测试用例格式规范、JSON 自动生成机制与测试执行方法。

## 核心原则

> **CSV 是唯一需要手动编辑的测试数据源。JSON 文件由测试脚本从 CSV 自动生成，不要手动编辑。**

## 文件位置

```
framework/tests/st/operation/test_case/{Op}_st_test_cases.csv   ← 手动编辑（数据源）
framework/tests/st/operation/test_case/{Op}_st_test_cases.json  ← 自动生成（勿手动编辑）
```

其中 `{Op}` 为 PascalCase 的 operation 名称，如 `Add`、`Compare`、`Where`。

## CSV 格式

### 表头定义

```
case_name,input_shape,input_dtype,input_format,input_datarange,output_shape,output_dtype,output_format,view_shape,tile_shape
```

### 字段说明

| 字段 | 格式 | 示例 | 说明 |
|------|------|------|------|
| case_name | 字符串 | `Add_test_27` | 测试用例名称，格式为 `{Op}_test_{N}` |
| input_shape | 引号包裹的列表 | `"[32, 32], [32, 32]"` | 每个输入 tensor 的 shape，逗号分隔 |
| input_dtype | 逗号分隔 | `fp32, fp32` | 每个输入 tensor 的 dtype（小写名） |
| input_format | 逗号分隔 | `ND, ND` | 每个输入 tensor 的格式 |
| input_datarange | 引号包裹的列表 | `[-10, 10], [-10, 10]` | 每个输入 tensor 的数据范围 [min, max] |
| output_shape | 列表 | `[32, 32]` | 输出 tensor 的 shape |
| output_dtype | 单值 | `fp32` | 输出 tensor 的 dtype |
| output_format | 单值 | `ND` | 输出 tensor 的格式 |
| view_shape | 列表 | `[16, 16]` | view shape |
| tile_shape | 列表 | `[16, 16]` | tile shape |

### 示例行

```
Add_test_1,"[1, 1], [1, 1]","fp32, fp32","ND, ND","[-10, 10], [-10, 10]","[1, 1]",fp32,ND,"[1, 1]","[16, 16]"
Add_test_23,"[32, 32], [32, 32]","int8, int8","ND, ND","[-10, 10], [-10, 10]","[32, 32]",int8,ND,"[16, 16]","[16, 16]"
Add_test_24,"[32, 32], [32, 32]","uint8, uint8","ND, ND","[-10, 10], [-10, 10]","[32, 32]",uint8,ND,"[16, 16]","[16, 16]"
```

### 注意事项

1. 含逗号的字段必须用双引号包裹
2. `case_name` 必须在 CSV 中唯一，编号接续现有最大编号
3. 新增测试用例的编号应接续现有最大编号，不要重复
4. 数据范围根据 dtype 选择合理值，避免溢出
5. CSV 行的 0-based 索引（不含表头）即为 `case_index`，测试执行脚本通过该索引选择用例

## JSON 格式（自动生成，勿手动编辑）

> **重要**：JSON 文件由 `run_operation_test_with_config.py` 脚本通过 `TestCaseLoader`（`framework/tests/cmake/scripts/helper/test_case_loader.py`）从 CSV 自动转换生成。每次运行测试时会覆盖 JSON 文件。手动编辑 JSON 会被覆盖，且格式可能与自动生成的不一致。

### 自动生成机制

`TestCaseLoader` 的转换流程（`test_case_loader.py`）：
1. 用 pandas 读取 CSV 文件
2. 为每行数据生成 `case_index`（0-based 行索引）
3. 解析 input/output tensor 描述
4. 序列化为 JSON 格式并写入 `{Op}_st_test_cases.json`

### JSON 顶层结构

```json
{
    "test_cases": [
        { ... },
        { ... }
    ]
}
```

### 单个测试用例结构

```json
{
    "case_index": 27,
    "case_name": "Add_test_28",
    "operation": "Add",
    "input_tensors": [
        {
            "name": "input0",
            "shape": [32, 32],
            "dtype": "int64",
            "format": "ND",
            "need_trans": false,
            "data_range": {
                "min": -100,
                "max": 100
            }
        },
        {
            "name": "input1",
            "shape": [32, 32],
            "dtype": "int64",
            "format": "ND",
            "need_trans": false,
            "data_range": {
                "min": -10,
                "max": 10
            }
        }
    ],
    "output_tensors": [
        {
            "name": "output0",
            "shape": [32, 32],
            "dtype": "int64",
            "format": "ND",
            "need_trans": false
        }
    ],
    "view_shape": [16, 16],
    "tile_shape": [16, 16],
    "params": {
        "on_board": true,
        "func_id": -1
    },
    "index": 0
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| case_index | int | 测试用例序号，与 CSV 中的编号对应 |
| case_name | string | 测试用例名称，与 CSV 中的 case_name 一致 |
| operation | string | operation 名称（PascalCase） |
| input_tensors | array | 输入 tensor 描述数组 |
| input_tensors[].name | string | tensor 名称，如 `input0`、`input1` |
| input_tensors[].shape | array[int] | tensor 形状 |
| input_tensors[].dtype | string | tensor dtype（小写名，如 `fp32`、`int64`） |
| input_tensors[].format | string | tensor 格式，通常为 `ND` |
| input_tensors[].need_trans | bool | 是否需要转置，通常为 `false` |
| input_tensors[].data_range | object | 数据范围 `{min, max}` |
| output_tensors | array | 输出 tensor 描述数组 |
| output_tensors[].name | string | 输出 tensor 名称 |
| output_tensors[].shape | array[int] | 输出形状 |
| output_tensors[].dtype | string | 输出 dtype |
| output_tensors[].format | string | 输出格式 |
| output_tensors[].need_trans | bool | 通常为 `false` |
| view_shape | array[int] | view shape |
| tile_shape | array[int] | tile shape |
| params | object | 测试参数 |
| params.on_board | bool | 是否在真实板子上运行 |
| params.func_id | int | 功能 ID，通常为 `-1` |
| index | int | 索引，通常为 `0` |

## dtype 小写名对照

测试用例中使用小写 dtype 名称（非 DT_* 枚举），对照表参见 [dtype-mapping.md](dtype-mapping.md) 中的「用户常用名到 DT_* 映射」章节。

## 测试运行方式

> **重要**：不要直接运行 gtest 二进制。必须通过 `run_operation_test_with_config.py` 脚本执行测试，该脚本会自动完成 CSV→JSON 转换、编译、golden 数据生成、NPU 执行和结果比对。直接运行 gtest 会因缺少 golden 参考数据而全部失败。

### 正确的测试执行方式

```bash
cd /mnt/workspace/gitCode/cann/pypto
python3 tools/scripts/run_operation_test_with_config.py {Op} -s {start_index} -e {end_index} -d {device_id}
```

参数说明：
- `{Op}`：operation 的 PascalCase 名称（如 `Add`、`Sub`、`Compare`）
- `-s {start_index}`：起始 case_index（CSV 数据行的 0-based 索引，不含表头）
- `-e {end_index}`：结束 case_index（含）
- `-d {device_id}`：NPU 设备 ID（通常为 `0`）

示例：
```bash
# 运行 Add 的 case_index 5 到 10（即 Add_test_6 到 Add_test_11）
python3 tools/scripts/run_operation_test_with_config.py Add -s 5 -e 10 -d 0

# 运行 Add 的最后两个用例（int64/uint64，CSV 第 27/28 行，case_index 26/27）
python3 tools/scripts/run_operation_test_with_config.py Add -s 26 -e 27 -d 0
```

### 测试脚本完整执行流程

`run_operation_test_with_config.py`（`tools/scripts/run_operation_test_with_config.py`）会自动完成以下步骤：

1. **CSV → JSON 转换**：`TestCaseLoader` 读取 CSV，生成 JSON 文件
2. **编译**：调用 `build_ci.py -f=cpp` 编译 pypto framework（gtest filter 为 `Test{Op}/{Op}OperationTest.Test{Op}/*`）
3. **Golden 数据生成**：调用 `framework/tests/st/operation/python/vector_operator_golden.py` 中注册的 golden 函数，用 PyTorch 生成参考数据
   - golden 函数通过 `@GoldenRegister.reg_golden_func` 装饰器注册
   - 例如 Add 的 golden 逻辑为 `inputs[0] + inputs[1]`（见 `vector_operator_golden.py` 中的 `gen_add_op_golden`）
4. **NPU 执行**：在指定 NPU 设备上运行 gtest 测试
5. **结果比对**：对比 golden 与实际输出，生成测试报告（Excel 文件）

### 测试结果判读

- **PASS**：编译成功 + golden 生成成功 + NPU 执行成功 + 精度比对通过
- **常见失败原因**：
  - `Failed to open file for writing input data`：直接运行 gtest 而未通过脚本，缺少 golden 数据
  - `Data type DT_xxx is not in supported types for op: XXX`：当前 NPU 架构不支持该 dtype（如 A2 板子运行仅 A5 支持的 dtype）
  - `Generate golden failed`：golden 脚本中未注册该 operation 或 dtype 不被 PyTorch 支持

### 架构限制

如果新增 dtype 仅在 A5 架构支持（pto-isa a2a3 不支持），则只能在 A5 板子（Ascend 950PR/950DT）上验证。在 A2 板子（Ascend 910）上运行时，`GetSupportedDataTypesByArch` 返回的 A2A3 集合不含该 dtype，dtype 检查会拦截并报错 `Data type DT_xxx is not in supported types`，这是预期行为。
