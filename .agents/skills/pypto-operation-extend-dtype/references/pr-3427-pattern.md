# PR 变更模式分析

本文件是对两个 GitCode PR 的完整变更模式分析，作为新增 dtype 支持的实现参考。

- [PR #3427](https://gitcode.com/cann/pypto/pull/3427) — 架构区分型（模式 A/B）
- [PR #4887](https://gitcode.com/cann/pypto/pull/4887) — 直接修改 supportedTypes 型（模式 C）

---

## PR 3427：架构区分型

- **标题**：feat(operation): Extend dtype support with arch distinction for Compare, Where, Gather, Scatter, Concat, Transpose, Relu
- **状态**：已合并（2026-05-28）
- **变更文件数**：33 个（全部为修改，无新增/删除）
- **覆盖 operation**：Compare（eq/ge/gt/le/lt）、Where、Gather、Scatter/ScatterTensor、Concat、Transpose、Relu

## 变更文件分类

### 按文件类型分类

| 文件类型 | 数量 | 说明 |
|---------|------|------|
| C++ 源码 (.cpp) | 4 | operation 实现 + 共享工具 |
| C++ 头文件 (.h) | 1 | 共享工具头文件 |
| API 文档 (.md) | 11 | 各 operation 的中文 API 文档 |
| CSV 测试用例 | 8 | ST 测试的 CSV 数据文件 |
| JSON 测试用例 | 8 | ST 测试的 JSON 数据文件 |

### 按 operation 分类

| Operation | C++ 源码 | 文档 | 测试用例 |
|-----------|---------|------|---------|
| Compare (eq/ge/gt/le/lt) | compare.cpp | eq/ge/gt/le/lt .md (5个) | Compare_st_test_cases.csv/json |
| Where | where.cpp | where.md | Where_st_test_cases.csv/json |
| Gather | indexing.cpp | gather.md | Gather_st_test_cases.csv/json |
| Scatter | （通过 common 工具） | scatter_.md | Scatter_st_test_cases.csv/json |
| ScatterTensor | （通过 common 工具） | （无文档变更） | ScatterTensor_st_test_cases.csv/json |
| Concat | tensor_transformation.cpp | concat.md | Concat_st_test_cases.csv/json |
| Transpose | （通过 common 工具） | transpose.md | Transpose_st_test_cases.csv/json |
| Relu | （通过 common 工具） | relu.md | Relu_st_test_cases.csv/json |
| Common（共享） | operation_common.cpp, operation_common.h | - | - |

## 核心变更模式

### 模式 1：架构区分型（C++ 源码修改）

适用于不同架构支持不同 dtype 集合的 operation。

**修改前**：
```cpp
std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
```

**修改后**：
```cpp
static const std::unordered_set<DataType> a2a3Types = {DT_FP16, DT_FP32, DT_BF16};
static const std::unordered_set<DataType> a5Types   = {DT_FP16, DT_FP32, DT_BF16, DT_INT16};
const auto& supportedTypes = GetSupportedDataTypesByArch(a2a3Types, a5Types);
```

**PR 中的实际变更示例**（compare.cpp）：

```cpp
// 修改前
std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};

// 修改后 — A5 新增 DT_INT16
static const std::unordered_set<DataType> CMP_A2A3_TYPES = {DT_FP16, DT_FP32, DT_BF16};
static const std::unordered_set<DataType> CMP_A5_TYPES   = {DT_FP16, DT_FP32, DT_BF16, DT_INT16};
const auto& supportedTypes = GetSupportedDataTypesByArch(CMP_A2A3_TYPES, CMP_A5_TYPES);
```

**关键要点**：
1. 将局部变量改为 `static const` 全局/文件级常量，避免重复构造
2. 使用 `{OP}_A2A3_TYPES` 和 `{OP}_A5_TYPES` 命名约定
3. 通过 `GetSupportedDataTypesByArch()` 进行架构分派
4. 所有重载函数中的 supportedTypes 都需要修改

### 模式 2：统一扩展型（C++ 源码修改）

适用于所有架构统一支持相同 dtype 集合的 operation。

**PR 中的实际变更示例**（tensor_transformation.cpp 中的 Concat）：

```cpp
// 修改前
std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_INT32, DT_INT16, DT_INT8, DT_BF16};

// 修改后 — 统一新增 DT_UINT8, DT_UINT16, DT_UINT32
static const std::unordered_set<DataType> TRANSPOSE_A2A3_TYPES = {DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_FP32, DT_BF16};
static const std::unordered_set<DataType> TRANSPOSE_A5_TYPES = {DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_FP32, DT_BF16};
const auto& supportedTypes = GetSupportedDataTypesByArch(TRANSPOSE_A2A3_TYPES, TRANSPOSE_A5_TYPES);
```

### 模式 3：架构区分型（API 文档修改）

**修改前**（pypto-eq.md）：
```markdown
Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_FP32。
```

**修改后**：
```markdown
Tensor支持的数据类型为：Atlas A2系列产品/Atlas A3系列产品：DT_FP16，DT_FP32，DT_BF16；Atlas A5系列产品：DT_FP16，DT_FP32，DT_BF16，DT_INT16。
```

### 模式 4：使用 npu 标签的文档格式（部分 operation）

部分 operation 文档使用 `<!-- npu -->` 标签区分产品线：

```markdown
3. Tensor数据类型说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR/Ascend 950DT：DT_INT32，DT_FP32，DT_INT16，DT_FP16，DT_BF16，DT_UINT8，DT_INT8。
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 训练系列产品/Atlas A3 推理系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 训练系列产品/Atlas A2 推理系列产品：DT_INT32，DT_INT16，DT_FP16，DT_FP32，DT_BF16。
   <!-- end id6 -->
```

修改时只需在对应产品线的 dtype 列表中追加新类型即可。

### 模式 5：测试用例添加（CSV）

在 CSV 文件末尾追加新行，每个新增 dtype 一行：

```csv
case_name,input_shape,input_dtype,input_format,input_datarange,output_shape,output_dtype,output_format,view_shape,tile_shape
Compare_test_20,"[32, 32], [32, 32]","bf16, bf16","ND, ND","[-10, 10], [-10, 10]","[32, 32]",bool,ND,"[16, 16]","[16, 16]"
Compare_test_21,"[32, 32], [32, 32]","int16, int16","ND, ND","[-100, 100], [-100, 100]","[32, 32]",bool,ND,"[16, 16]","[16, 16]"
```

**数据范围参考**：

| dtype | 建议数据范围 |
|-------|------------|
| int8 | `[-10, 10]` 或 `[-100, 100]` |
| uint8 | `[0, 10]` 或 `[0, 100]` |
| int16 | `[-100, 100]` 或 `[-1000, 1000]` |
| uint16 | `[0, 100]` 或 `[0, 1000]` |
| int32 | `[-100, 100]` 或 `[-1000, 1000]` |
| uint32 | `[0, 1000]` 或 `[0, 10000]` |
| int64 | `[-1000, 1000]` |
| uint64 | `[0, 1000]` |
| fp16 | `[-10, 10]` 或 `[-100, 100]` |
| fp32 | `[-10, 10]` 或 `[-100, 100]` |
| bf16 | `[-10, 10]` 或 `[-100, 100]` |
| bool | `[0, 1]` |

### 模式 6：测试用例添加（JSON）

在 JSON 文件的 `test_cases` 数组中追加新对象：

```json
{
    "case_index": 20,
    "case_name": "Compare_test_21",
    "operation": "Compare",
    "input_tensors": [
        {"name": "input0", "shape": [32, 32], "dtype": "int16", "format": "ND", "need_trans": false, "data_range": {"min": -100, "max": 100}},
        {"name": "input1", "shape": [32, 32], "dtype": "int16", "format": "ND", "need_trans": false, "data_range": {"min": -100, "max": 100}}
    ],
    "output_tensors": [
        {"name": "output0", "shape": [32, 32], "dtype": "bool", "format": "ND", "need_trans": false}
    ],
    "view_shape": [16, 16],
    "tile_shape": [16, 16],
    "params": {"on_board": true, "func_id": -1},
    "index": 0
}
```

## PR 中各 operation 新增的 dtype

| Operation | A2/A3 新增 | A5 新增 | 新增测试用例数 |
|-----------|-----------|---------|-------------|
| Compare | 无 | DT_INT16 | 7 |
| Where | DT_INT32, DT_INT16 | DT_INT32, DT_INT16, DT_UINT8, DT_INT8 | 5 |
| Gather | DT_UINT8, DT_UINT16, DT_UINT32, DT_BOOL, DT_BF16, DT_INT8 | 同 A2/A3 | 6 |
| Scatter | DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_BF16 | 同 A2/A3 | 5 |
| ScatterTensor | DT_INT8, DT_UINT8, DT_INT16, DT_INT32 | 同 A2/A3 | 4 |
| Concat | DT_UINT8, DT_UINT16, DT_UINT32 | 同 A2/A3 | 3 |
| Transpose | DT_INT8, DT_UINT8 | 同 A2/A3 | 2 |
| Relu | DT_INT16, DT_INT32 | 同 A2/A3 | 2 |

## 共享工具变更

`operation_common.cpp` 和 `operation_common.h` 中的 `GetSupportedDataTypesByArch` 函数在 PR 中主要是格式调整，其核心逻辑（通过 `NPUArch::DAV_3510` 判断 A5 架构）保持不变。该函数是所有架构区分型 operation 的基础。

## 注意事项

1. PR 中的部分 operation（如 Transpose、Relu）没有对应的 `.cpp` 源码变更，说明它们的 dtype 扩展是通过共享路径或已有的 `GetSupportedDataTypesByArch` 调用实现的。在实际操作中，需要检查目标 operation 的源码是否有独立的 `supportedTypes` 定义。
2. Scatter/ScatterTensor 没有独立的 `.cpp` 文件变更，说明其 dtype 检查逻辑在共享代码中。
3. 测试用例的 `case_index` 和 `case_name` 必须在 CSV 和 JSON 中保持一致。
4. 对于有 `<!-- npu -->` 标签的文档，注意 `950` 对应 A5 架构，`A3` 和 `910b` 对应 A2/A3 架构。

---

## PR 4887：直接修改 supportedTypes 型（模式 C）

- **标题**：feat(operation): Concat supports int64 and uint64
- **状态**：已合并（2026-07-17）
- **变更文件数**：2 个
- **覆盖 operation**：Concat
- **新增 dtype**：DT_INT64, DT_UINT64

### 变更文件列表

| # | 文件 | 变更类型 | 说明 |
|---|------|---------|------|
| 1 | `framework/src/interface/operation/vector/tensor_transformation.cpp` | 修改 | 直接在 `supportedTypes` 局部变量中追加 dtype |
| 2 | `docs/zh/api/operation/pypto-concat.md` | 修改 | 在 dtype 列表中追加新类型 |

### 核心变更模式（模式 C）

适用于不区分架构、直接使用局部变量 `supportedTypes` 的 operation。

**C++ 源码变更**（`tensor_transformation.cpp` 中的 `CheckCat` 函数）：

```cpp
// 修改前
std::unordered_set<DataType> supportedTypes = {DT_INT8,   DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                                               DT_UINT32, DT_FP16,  DT_FP32,  DT_BF16};

// 修改后 — 直接在集合中追加 DT_INT64, DT_UINT64
std::unordered_set<DataType> supportedTypes = {DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                                               DT_FP16, DT_FP32,  DT_BF16,  DT_INT64,  DT_UINT64};
```

**文档变更**（`pypto-concat.md`）：

```markdown
<!-- 修改前 -->
Tensor支持的数据类型为：DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_FP32, DT_BF16。

<!-- 修改后 — 直接追加 -->
Tensor支持的数据类型为：DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_FP32, DT_BF16, DT_INT64, DT_UINT64。
```

### 模式 C 的特点

1. **不区分架构**：所有架构统一支持相同的 dtype 集合，无需调用 `GetSupportedDataTypesByArch()`
2. **不使用 static const 命名集合**：dtype 集合是函数内的局部变量 `supportedTypes`
3. **变更极简**：只需在集合初始化列表中追加新 DT_* 值，无需新增变量或调用
4. **文档同样简单**：直接在 dtype 列表末尾追加，无需区分产品线

### 模式选择决策树

```
源码中是否有 GetSupportedDataTypesByArch 调用？
├── 是 → 检查 {OP}_A2A3_TYPES 和 {OP}_A5_TYPES 内容是否相同
│       ├── 不同 → 模式 A（架构区分型）
│       └── 相同 → 模式 B（统一扩展型）
└── 否 → 模式 C（直接修改 supportedTypes）
```

### PR 4887 与 PR 3427 的对比

| 维度 | PR 3427（模式 A/B） | PR 4887（模式 C） |
|------|---------------------|-------------------|
| 架构区分 | 区分 A2A3/A5 | 不区分 |
| dtype 集合形式 | `static const` 命名集合 | 局部变量 `supportedTypes` |
| 架构分派函数 | `GetSupportedDataTypesByArch()` | 无 |
| 文档格式 | 按 `<!-- npu -->` 标签或架构文本区分 | 统一 dtype 列表 |
| 变更复杂度 | 较高（需修改两个集合） | 极低（直接追加） |
| 典型 operation | add, sub, compare, where | concat（部分场景） |

> 注意：同一个 operation 在不同时期可能采用不同模式。例如 Concat 在 PR 3427 中使用模式 B（通过 `GetSupportedDataTypesByArch`），但在 PR 4887 中因为新增 int64/uint64 而发现 `CheckCat` 函数中直接使用了局部变量 `supportedTypes`（模式 C）。实际修改时应以源码当前的实际写法为准。
