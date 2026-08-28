---
name: pypto-operation-extend-dtype
description: 为现有 PyPTO operation 新增 dtype 支持。自动完成 pto-isa 依赖检查、C++ 源码修改、API 文档更新、测试用例编写与运行。触发词：新增 dtype、增加数据类型、添加 dtype 支持、extend dtype、支持更多数据类型、int64 支持、uint64 支持、添加 int8 支持、扩展 dtype。
---

# pypto-operation-extend-dtype

为现有 PyPTO operation 新增 dtype 支持，覆盖从依赖检查到测试验证的完整流程。

## 输入

用户会提供：
- **operation 名称**：如 `add`、`sub`、`compare`、`where`、`gather`、`concat`、`transpose`、`relu` 等
- **目标 dtype**：如 `int64`、`uint64`、`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`bool`、`bf16`、`fp8e4m3` 等
- **目标架构**（可选）：如 `a2a3`、`a5`，默认两个架构都检查

示例输入：
> 请帮我为 add 这个 operation 增加 int64 和 uint64 两种 dtype 的支持。

## 参考文件

| File | Purpose | Load Timing |
|------|---------|-------------|
| [references/dtype-mapping.md](references/dtype-mapping.md) | DT_* 枚举与 C++ 类型、pto-isa 类型名的完整映射表 | 阶段 1 和 阶段 2 开始时读取 |
| [references/operation-location-guide.md](references/operation-location-guide.md) | operation 名称到源码、文档、测试、pto-isa 头文件的定位指南 | 阶段 1 开始时读取 |
| [references/pr-3427-pattern.md](references/pr-3427-pattern.md) | PR 3427 和 PR 4887 的完整变更模式分析（三种模式），作为实现参考 | 阶段 3 开始时读取 |
| [references/test-case-format.md](references/test-case-format.md) | CSV 测试用例格式规范、JSON 自动生成机制与测试执行指南 | 阶段 4 开始时读取 |
| [scripts/check_dtype_support.py](scripts/check_dtype_support.py) | 检查 pypto operation 源码中是否已支持目标 dtype | 阶段 1 执行 |
| [scripts/check_pto_isa_support.py](scripts/check_pto_isa_support.py) | 检查 pto-isa 头文件中是否已支持目标 dtype | 阶段 2 执行 |
| [templates/dtype-extension-checklist.md](templates/dtype-extension-checklist.md) | 完整变更清单模板 | 阶段 3 开始时读取 |
| [templates/csv-test-case-template.csv](templates/csv-test-case-template.csv) | CSV 测试用例模板 | 阶段 4 编写测试时参考 |
| [templates/json-test-case-template.json](templates/json-test-case-template.json) | JSON 格式参考（仅用于了解字段结构，JSON 由脚本自动生成，不要手动编辑） | 阶段 4 了解 JSON 格式时参考 |
| [templates/doc-dtype-table-template.md](templates/doc-dtype-table-template.md) | API 文档 dtype 表格模板 | 阶段 4 更新文档时参考 |

## 前置条件

- pypto 源码位于 `/mnt/workspace/gitCode/cann/pypto`
- pto-isa 源码位于 `/mnt/workspace/gitCode/cann/pto-isa`
- 如需运行测试，需要已编译的 pypto 环境

## 工作流程

执行五个阶段。阶段 1 和阶段 2 可并行运行。如果阶段 1 或阶段 2 发现不支持，则提前终止并报告。

### 阶段 1：检查 pypto 是否已支持目标 dtype

1. 读取 [references/operation-location-guide.md](references/operation-location-guide.md) 以定位 operation 的源码文件。
2. 读取 [references/dtype-mapping.md](references/dtype-mapping.md) 以获取目标 dtype 对应的 DT_* 枚举名。
3. 运行检查脚本：

   ```bash
   python3 scripts/check_dtype_support.py \
     --pypto-root /mnt/workspace/gitCode/cann/pypto \
     --operation add \
     --dtypes int64,uint64
   ```

4. 分析脚本输出的 JSON 结果：
   - 如果 dtype 已在 **a2a3Types** 和 **a5Types** 中都存在 → 报告「已支持」，任务结束。
   - 如果 dtype 仅在其中一个架构集合中存在 → 报告部分支持情况，继续阶段 2 确认 pto-isa 是否允许扩展另一架构。
   - 如果 dtype 完全不存在 → 继续阶段 2。

5. 如果需要手动确认，在 operation 的 `.cpp` 源码中搜索 `{OP}_A2A3_TYPES` 和 `{OP}_A5_TYPES`（或对应的 `supportedTypes` 集合），确认 dtype 的存在性。

### 阶段 2：检查 pto-isa 是否已支持目标 dtype

> 新增 dtype 必须先在 pto-isa 层有底层实现，否则 pypto 层无法使用。

1. 读取 [references/operation-location-guide.md](references/operation-location-guide.md) 以定位 pto-isa 头文件路径。
2. 运行检查脚本：

   ```bash
   python3 scripts/check_pto_isa_support.py \
     --pto-isa-root /mnt/workspace/gitCode/cann/pto-isa \
     --operation add \
     --dtypes int64,uint64
   ```

3. 分析脚本输出的 JSON 结果：
   - 如果 pto-isa **支持**该 dtype（在 `static_assert` 中显式列出，或通过 `if constexpr` 分支，或通过 `sizeof(T)` 检查间接支持）→ 继续阶段 3。
   - 如果 pto-isa **不支持**该 dtype → 报告「pto-isa 未支持此 dtype」，列出 pto-isa 中该 operation 的 `static_assert` 支持的已有类型。
   - **注意**：部分 operation（如 Concat）可能通过 pypto 框架层的通用数据搬运路径（如 copy/memcpy）支持某些 dtype，而不依赖 pto-isa 的专用指令。如果 pto-isa 不支持但 operation 可能有通用路径，报告中应提示用户确认是否继续。如果用户确认继续，则进入阶段 3。

4. 脚本检查逻辑说明：
   - 在 pto-isa 的 `include/pto/npu/{a2a3,a5}/T{Op}.hpp` 中搜索 `TAddCheck`（或对应 `{Op}Check`）函数中的 `static_assert`。
   - `static_assert` 中会列出所有支持的 C++ 类型（如 `std::is_same_v<T, int64_t>`）。
   - 如果操作有 `if constexpr` 分支（如 `Int64Binary`），说明已有专门的 int64/uint64 实现。
   - 如果 `static_assert` 中未列出目标类型，但头文件中有基于 `sizeof(T)` 的通用检查（如 `sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8`），说明该操作通过通用路径支持多种 dtype，间接支持目标类型。
   - 脚本输出的 `dtype_status` 中 `supported_via_sizeof: true` 表示通过 sizeof 检查间接支持。

### 阶段 3：修改 C++ 源码

1. 读取 [references/pr-3427-pattern.md](references/pr-3427-pattern.md) 以理解标准变更模式。
2. 读取 [templates/dtype-extension-checklist.md](templates/dtype-extension-checklist.md) 以跟踪变更项。
3. 定位 operation 的 `.cpp` 源码文件（参见 [references/operation-location-guide.md](references/operation-location-guide.md)）。
4. 检查 operation 使用哪种 dtype 集合模式。有三种模式，根据源码实际写法选择对应的修改方式：

   **变更模式 A（架构区分型）**——源码中使用 `{OP}_A2A3_TYPES` 和 `{OP}_A5_TYPES` 两个独立集合，通过 `GetSupportedDataTypesByArch()` 分派：

   ```cpp
   // 修改前
   static const std::unordered_set<DataType> ADD_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
   static const std::unordered_set<DataType> ADD_A5_TYPES = {DT_INT32, DT_FP32, DT_INT16, DT_FP16, DT_BF16, DT_UINT8, DT_INT8};

   // 修改后（在 A5 集合中新增 DT_INT64, DT_UINT64）
   static const std::unordered_set<DataType> ADD_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
   static const std::unordered_set<DataType> ADD_A5_TYPES = {DT_INT32, DT_FP32, DT_INT16, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_INT64, DT_UINT64};
   ```

   **变更模式 B（统一扩展型）**——源码中使用 `{OP}_A2A3_TYPES` 和 `{OP}_A5_TYPES` 但两个集合内容相同，通过 `GetSupportedDataTypesByArch()` 分派：

   ```cpp
   // 修改前 — 两个集合内容相同
   const auto& supportedTypes = GetSupportedDataTypesByArch(TRANSPOSE_A2A3_TYPES, TRANSPOSE_A5_TYPES);

   // 修改后（在两个集合中都新增）
   static const std::unordered_set<DataType> TRANSPOSE_A2A3_TYPES = {..., DT_INT8, DT_UINT8};
   static const std::unordered_set<DataType> TRANSPOSE_A5_TYPES = {..., DT_INT8, DT_UINT8};
   ```

   **变更模式 C（直接修改 supportedTypes 型）**——源码中直接使用局部变量 `supportedTypes`，不区分架构，不调用 `GetSupportedDataTypesByArch()`。参见 PR [#4887](https://gitcode.com/cann/pypto/pull/4887)：

   ```cpp
   // 修改前 — CheckCat 函数中的局部变量
   std::unordered_set<DataType> supportedTypes = {DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                                                  DT_INT32, DT_UINT32, DT_FP16, DT_FP32, DT_BF16};

   // 修改后（直接在集合中追加 DT_INT64, DT_UINT64）
   std::unordered_set<DataType> supportedTypes = {DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                                                  DT_INT32, DT_UINT32, DT_FP16, DT_FP32, DT_BF16,
                                                  DT_INT64, DT_UINT64};
   ```

5. **如何判断使用哪种模式**：在源码中搜索 operation 对应的 dtype 检查代码：
   - 如果找到 `{OP}_A2A3_TYPES` / `{OP}_A5_TYPES` 且内容不同 → 模式 A
   - 如果找到 `{OP}_A2A3_TYPES` / `{OP}_A5_TYPES` 且内容相同 → 模式 B
   - 如果只找到局部变量 `supportedTypes = {...}` 且无 `GetSupportedDataTypesByArch` 调用 → 模式 C
6. **重要**：如果 operation 有多个函数重载（如 Tensor/Tensor、Tensor/Element、Element/Tensor），每个重载中的 supportedTypes 都需要修改。
7. 如果 pto-isa 中该 dtype 需要特殊的 `if constexpr` 分支（如 int64 使用 `Int64Binary`），确认 pypto 源码中没有额外的 dtype 限制逻辑需要修改。

### 阶段 4：更新文档与编写测试用例

#### 4a. 更新 API 文档

1. 读取 [templates/doc-dtype-table-template.md](templates/doc-dtype-table-template.md) 以了解文档格式。
2. 定位文档文件：`docs/zh/api/operation/pypto-{op}.md`。
3. 找到「约束说明」章节中的 Tensor 数据类型说明部分。
4. 根据架构区分情况更新 dtype 列表：

   **架构区分型文档**（使用 `<!-- npu -->` 标签区分）：

   ```markdown
   3. Tensor数据类型说明：
      <!-- npu="950" id4 -->
      - Ascend 950PR/Ascend 950DT：DT_INT32，DT_FP32，DT_INT16，DT_FP16，DT_BF16，DT_UINT8，DT_INT8，DT_INT64，DT_UINT64。
      <!-- end id4 -->
      <!-- npu="A3" id5 -->
      - Atlas A3 训练系列产品/Atlas A3 推理系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。
      <!-- end id5 -->
      <!-- npu="910b" id6 -->
      - Atlas A2 训练系列产品/Atlas A2 推理系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。
      <!-- end id6 -->
   ```

   **统一型文档**（直接在 dtype 列表中追加）：

   ```markdown
   Tensor支持的数据类型为：DT_FP16，DT_FP32，DT_BF16，DT_INT8，DT_UINT8。
   ```

5. 保持文档中 `<!-- npu -->` 标签的格式一致性，不要破坏 id 编号。

#### 4b. 编写测试用例

> **重要**：只需编辑 CSV 文件。JSON 文件由测试执行脚本（`run_operation_test_with_config.py`）在运行时从 CSV 自动生成，**不要手动编辑 JSON 文件**——手动编辑的内容会被覆盖，且格式可能与自动生成的不一致。

1. 读取 [references/test-case-format.md](references/test-case-format.md) 以了解 CSV 格式规范和测试执行机制。
2. 读取 [templates/csv-test-case-template.csv](templates/csv-test-case-template.csv) 作为 CSV 行格式参考。
3. 定位 CSV 测试用例文件：`framework/tests/st/operation/test_case/{Op}_st_test_cases.csv`
4. 在 CSV 文件末尾追加新测试行，覆盖每个新增 dtype：

   ```
   Add_test_27,"[32, 32], [32, 32]","int64, int64","ND, ND","[-100, 100], [-10, 10]","[32, 32]",int64,ND,"[16, 16]","[16, 16]"
   Add_test_28,"[32, 32], [32, 32]","uint64, uint64","ND, ND","[0, 1000], [0, 1000]","[32, 32]",uint64,ND,"[16, 16]","[16, 16]"
   ```

   **数据范围选择规则**：
   - 有符号整数（int8/int16/int32/int64）：`[-100, 100]` 或 `[-1000, 1000]`
   - 无符号整数（uint8/uint16/uint32/uint64）：`[0, 100]` 或 `[0, 1000]`
   - 浮点数（fp16/fp32/bf16）：`[-10, 10]` 或 `[-100, 100]`
   - bool：`[0, 1]`

5. 确认新增行的 `case_name` 编号接续 CSV 中现有最大编号（如现有最大为 `Add_test_26`，则新增 `Add_test_27`、`Add_test_28`）。CSV 行的 0-based 索引即为 `case_index`，后续测试执行通过该索引选择用例。

### 阶段 5：编译与测试验证

> **重要**：不要直接运行 gtest 二进制（`./tile_fwk_stest --gtest_filter=...`）。直接运行会因为缺少 golden 参考数据而全部失败。必须通过 `run_operation_test_with_config.py` 脚本执行，它会自动完成 CSV→JSON 转换、编译、golden 数据生成、NPU 执行和结果比对。

1. 使用测试执行脚本运行新增的测试用例：

   ```bash
   cd /mnt/workspace/gitCode/cann/pypto
   python3 tools/scripts/run_operation_test_with_config.py {Op} -s {start_index} -e {end_index} -d {device_id}
   ```

   参数说明：
   - `{Op}`：operation 的 PascalCase 名称（如 `Add`、`Sub`、`Compare`）
   - `-s {start_index}`：起始 case_index（CSV 数据行的 0-based 索引，不含表头）
   - `-e {end_index}`：结束 case_index（含）
   - `-d {device_id}`：NPU 设备 ID（通常为 `0`）

   示例（运行 Add 的新增 int64/uint64 用例，位于 CSV 第 27/28 行，case_index 为 26/27）：

   ```bash
   python3 tools/scripts/run_operation_test_with_config.py Add -s 26 -e 27 -d 0
   ```

2. 该脚本会自动完成以下完整流程：
   - **CSV → JSON 转换**：从 CSV 读取测试用例，生成对应的 JSON 文件
   - **编译**：调用 `build_ci.py` 编译 pypto framework（gtest filter 已硬编码为 `Test{Op}/{Op}OperationTest.Test{Op}/*`）
   - **Golden 数据生成**：调用 `framework/tests/st/operation/python/vector_operator_golden.py` 中注册的 golden 函数，用 PyTorch 生成参考数据
   - **NPU 执行**：在指定 NPU 设备上运行测试
   - **结果比对**：对比 golden 与实际输出，生成测试报告

3. 验证结果：
   - 编译成功，无错误
   - Golden 数据生成成功（日志中出现 `Generate golden success`）
   - 测试全部通过，新增 dtype 的测试用例执行正确
   - 如果测试失败，分析失败原因：
     - `Data type DT_xxx is not in supported types for op: XXX`：当前 NPU 架构不支持该 dtype（如 A2 板子运行仅 A5 支持的 dtype），需在对应架构的板子上验证
     - golden 生成失败：检查 `vector_operator_golden.py` 中是否已注册该 operation 的 golden 函数
     - 精度比对失败：可能是数据范围不当、shape 不匹配等

4. **架构限制说明**：如果新增 dtype 仅在 A5 架构支持（pto-isa a2a3 不支持），则只能在 A5 板子（Ascend 950PR/950DT）上验证。在 A2 板子（Ascend 910）上运行会因 `GetSupportedDataTypesByArch` 返回的 A2A3 集合不含该 dtype 而被拦截，这是预期行为。

## 输出

完成所有阶段后，输出结构化摘要：

```markdown
## dtype 扩展结果

- operation: {op}
- 新增 dtype: {dtype_list}
- 支持架构: {a2a3/a5/both}

## 变更文件清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| framework/src/interface/operation/vector/{file}.cpp | 修改 | 在 {OP}_A2A3_TYPES / {OP}_A5_TYPES 中新增 dtype |
| docs/zh/api/operation/pypto-{op}.md | 修改 | 更新约束说明中的 dtype 表格 |
| framework/tests/st/operation/test_case/{Op}_st_test_cases.csv | 修改 | 新增 N 条测试用例 |

> 注意：`{Op}_st_test_cases.json` 不在变更清单中——它由测试脚本从 CSV 自动生成，不要手动编辑。

## 测试结果

- 编译: PASS / FAIL
- Golden 生成: PASS / FAIL
- C++ ST 测试: PASS / FAIL (N/M 用例通过)

## 已知问题

- <如实列出未验证项、环境限制或数据缺口>
```

## 约束

1. 不得跳过 pto-isa 依赖检查直接修改 pypto 源码。
2. 如果 pto-isa 不支持目标 dtype，必须报告并终止，不得强行添加。
3. 文档中的 `<!-- npu -->` 标签格式必须保持一致，不得破坏已有的 id 编号体系。
4. **不要手动编辑 JSON 测试用例文件**——JSON 由 `run_operation_test_with_config.py` 脚本从 CSV 自动生成，手动编辑会被覆盖。只需编辑 CSV 文件。
5. **不要直接运行 gtest 二进制执行测试**——必须通过 `run_operation_test_with_config.py` 脚本执行，该脚本会自动生成 golden 数据。直接运行 gtest 会因缺少 golden 数据而全部失败。
6. 如果 operation 有多个函数重载，所有重载的 supportedTypes 都需要修改。
7. 修改源码时保持现有代码风格和格式，不做无关改动。
