# PR 6074：operation dtype 声明迁移到 JSON 配置（新架构）

- **PR**：https://gitcode.com/cann/pypto/pull/6074
- **标题**：feat(operation): operation supported dtype 由 C++ 内联集合迁移到 JSON 配置
- **性质**：架构重构，取代历史 PR 3427 / 4887 的 `GetSupportedDataTypesByArch` 机制

---

## 概述

本 PR 将「每个 operation 在其 `.cpp` 源码中内联声明的 supported dtype 集合」迁移为「集中式的 JSON 配置文件」，由 `ConfigManager` 统一加载并按 `Opcode` 查询。这是 dtype 扩展机制的根本性变更：

- **删除**了 `GetSupportedDataTypesByArch` 函数，以及 `SupportedDataTypesMap` / `DataTypesSet` 类型别名、`kEmptyDataTypes` 常量。
- **删除**了散布在 17 个 `.cpp` 文件中的 `{OP}_A2A3_TYPES` / `{OP}_A5_TYPES` / `{OP}_KIRIN9030_TYPES` / `{OP}_KIRINX90_TYPES` 静态集合与 `{OP}_SUPPORTED_TYPES` 映射表。
- **新增**了 `ConfigManager::GetOpSupportedInputDtypes(Opcode)`，读取 JSON 配置返回当前架构下某 opcode 支持的输入 dtype。

## 旧架构（已删除）

dtype 集合硬编码在每个 operation 源码内，通过架构查表分派：

```cpp
static const std::unordered_set<DataType> ADD_A2A3_TYPES = {DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16};
static const std::unordered_set<DataType> ADD_A5_TYPES = {DT_INT32, DT_FP32, DT_INT16, DT_FP16, DT_BF16, DT_UINT8, DT_INT8};
static const SupportedDataTypesMap ADD_SUPPORTED_TYPES = {
    {NPUArch::DAV_1001, ADD_A2A3_TYPES},
    {NPUArch::DAV_2201, ADD_A2A3_TYPES},
    {NPUArch::DAV_3510, ADD_A5_TYPES},
};
const auto& supportedTypes = GetSupportedDataTypesByArch(ADD_SUPPORTED_TYPES);
```

问题：每个 operation 每个 overload 都要维护重复的 C++ 静态集合，扩展 dtype 需要在源码多处改动，架构与 dtype 关系分散。

## 新架构

### JSON 配置文件

dtype 声明集中在 `framework/src/interface/configs/platform_op_supported_dtypes/` 下，按架构分文件：

| 文件 | npu_arch | NPUArch | 架构 |
|------|----------|---------|------|
| `a2a3_supported_op_dtypes.json` | 2201 | DAV_2201 | A2/A3 |
| `a5_supported_op_dtypes.json` | 3510 | DAV_3510 | A5 |
| `kirin9030_supported_op_dtypes.json` | 3113 | DAV_3113 | kirin9030 |
| `kirinx90_supported_op_dtypes.json` | 3003 | DAV_3003 | kirinX90 |

JSON 结构：

```json
{
    "npu_arch": 2201,
    "ops": {
        "ADD":  { "input_dtypes": ["int16", "int32", "fp16", "float32", "bf16"] },
        "ADDS": { "input_dtypes": ["int16", "int32", "fp16", "float32", "bf16"] },
        "MAXIMUM": { "input_dtypes": ["int16", "int32", "fp16", "float32", "bf16"] }
    }
}
```

关键点：

- `npu_arch` 是 `NPUArch` 枚举的整数值。
- `ops` 的 key 是 opcode 字符串（`OpcodeManager` 注册的 str，如 `OP_ADD` → `"ADD"`、标量版 `OP_ADDS` → `"ADDS"`、`Maximum` → `"MAXIMUM"`）。
- `input_dtypes` 是 `STR_DATA_TYPE_MAP` 的小写友好名。注意 DT_FP32 的 JSON 键是 `float32`（不是 `fp32`）。

### opcode 字符串对照（部分）

存在张量/标量、或组合分派的 operation，一个 operation 对应多个 opcode：

| operation | opcode 字符串 |
|-----------|---------------|
| Add / Sub / Mul / Div | `ADD`/`ADDS`、`SUB`/`SUBS`、`MUL`/`MULS`、`DIV`/`DIVS` |
| Maximum / Minimum | `MAXIMUM`/`MAXS`、`MINIMUM`/`MINS` |
| Remainder | `REM` / `REMS` / `REMRS` |
| FloorDiv | `FLOORDIV` / `FLOORDIVS` |
| Bitwise And/Or/Xor | `BITWISEAND`/`BITWISEANDS`、`BITWISEOR`/`BITWISEORS`、`BITWISEXOR`/`BITWISEXORS`、`BITWISENOT` |
| Bitwise shift | `BITWISERIGHTSHIFT`/`BITWISERIGHTSHIFTS`/`SBITWISERIGHTSHIFT`、`BITWISELEFTSHIFT`/... |
| Compare | `CMP` / `CMPS` |
| Where | `WHERE_TT` / `WHERE_TS` / `WHERE_ST` / `WHERE_SS` |
| Scatter | `SCATTER` / `SCATTER_ELEMENT` / `SCATTER_UPDATE` |
| Gather | `GATHER` / `GATHER_ELEMENT` / `GATHER_IN_UB` |
| Transpose | `TRANSPOSE_MOVEIN` / `TRANSPOSE_MOVEOUT` / `TRANSPOSE_VNCHWCONV` |
| Amax / Amin / Sum | `ROWMAX_SINGLE` / `ROWMIN_SINGLE` / `ROWSUM_SINGLE` |
| Full | `VEC_DUP` |

### C++ 读取

```cpp
// framework/src/interface/configs/config_manager.h/.cpp
const std::unordered_set<DataType>& ConfigManager::Instance().GetOpSupportedInputDtypes(const Opcode& opcode);
```

- 内部按 `Platform::Instance().GetSoc().GetNPUArch()` 查表，只读 `input_dtypes` 类别。
- 未配置时返回空集，dtype 检查会报错 `Data type DT_xxx is not in supported types`。

`ConfigManager` 内的存储结构（`platformSupportedOpDtypesMap_`）：

```cpp
std::unordered_map<NPUArch,
    std::unordered_map<std::string,
        std::unordered_map<Opcode, std::unordered_set<DataType>>>> platformSupportedOpDtypesMap_;
```

即 `NPUArch` → 类别（目前仅 `input_dtypes`）→ `Opcode` → dtype 集合。

实现要点：

- `LoadPlatformSupportedOpDtypes()`：读取 `GetPyptoLibPath() + "/configs/platform_op_supported_dtypes"` 下所有 `*.json`，把 `npu_arch` 转 `NPUArch`，opcode 字符串经 `OpcodeManager::Inst().GetOpcode(...)` 转 `Opcode`，写入 map。
- 采用懒加载（`std::call_once`）：首次调用 `GetOpSupportedInputDtypes` 时才加载并 dump，避免在 `dlopen` 静态初始化阶段过早构造 `OpcodeManager`（否则会 segfault）。
- `DumpPlatformSupportedOpDtypes()`：记录加载路径与每个 arch/category/opcode/dtypes，用于确认读的是安装包中的 JSON（而非源码）。

operation 层的调用形式：

```cpp
const auto& supportedTypes = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_ADD);
CheckTensorDataType(self.GetStorage(), supportedTypes, "ADD");
```

## 新 dtype 添加方式

现在扩展 dtype 不再改 `.cpp` 源码，只需在对应架构的 JSON 中追加 dtype：

1. 在 `.cpp` 中查找该 overload 使用的 opcode（`GetOpSupportedInputDtypes(Opcode::OP_XXX)`）。
2. 在目标架构 JSON 的 `ops.{OPCODE}.input_dtypes` 中追加小写 dtype 名。
3. 有多个分发 opcode 时（如 `ADD`/`ADDS`）每个都要改。

完整流程见 [dtype-mapping.md](dtype-mapping.md) 与 [SKILL.md](../SKILL.md)。

## 例外：无独立 opcode 的复合 operation

部分复合 operation 没有独立 opcode，其 dtype 检查仍内联在 `.cpp` 中：

- `Clip`：内部拆为 `Maximum`/`Minimum`，dtype 检查使用内联的 `CLIP_A2A3_TYPES`/`CLIP_A5_TYPES`（本地 ternary 按架构选择），不在 JSON 中。

## 与历史 PR 3427 / 4887 的对比

| 维度 | PR 3427 / 4887（旧） | PR 6074（新） |
|------|----------------------|--------------|
| dtype 来源 | `.cpp` 内联 static const 集合 | JSON 配置文件 |
| 架构分派 | `GetSupportedDataTypesByArch(map)` | `ConfigManager::GetOpSupportedInputDtypes(Opcode)` |
| 扩展 dtype 改动点 | 修改 `.cpp` 源码 | 修改 JSON 配置 |
| 架构层级 | A2A3/A5（可选 kirin 集合） | a2a3/a5/kirin9030/kirinx90 四文件 |
| 运行时加载 | 编译期硬编码 | 懒加载（`std::call_once`） |

## 相关文件

- `framework/src/interface/configs/config_manager.h` / `.cpp` — 加载 + 查询 + dump 实现
- `framework/src/interface/configs/platform_op_supported_dtypes/*.json` — dtype 配置
- `framework/src/interface/operation/operation_common.h` / `.cpp` — 删除了 `GetSupportedDataTypesByArch`
- `framework/src/interface/operation/vector/*.cpp` — 17 个文件改用 `GetOpSupportedInputDtypes`（并删除内联集合）
- `CMakeLists.txt` — 新增 `install(DIRECTORY ...)` 安装 JSON 配置：

```cmake
install(DIRECTORY framework/src/interface/configs/platform_op_supported_dtypes/
        DESTINATION ${PyPTO_WhlName}/lib/configs/platform_op_supported_dtypes
        COMPONENT pypto)
```

## 验证

- 编译：`python3 build_ci.py --clean --no_isolation`（0 错误）。
- 运行：`source python_compile.sh`，在 `debug/plog/pypto-log-*.log` 中确认：
  - `Load platform supported op dtypes from <site-packages>/pypto/lib/configs/platform_op_supported_dtypes`（读的是安装包，非源码）。
  - `Parsed platform op dtype config {a2a3,a5,kirin9030,kirinx90}_supported_op_dtypes.json`。
  - `platformSupportedOpDtypesMap arch=DAV_2201/DAV_3510/DAV_3113/DAV_3003` 及每个 `opcode=... dtypes={...}`。
