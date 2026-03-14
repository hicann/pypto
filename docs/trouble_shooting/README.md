# PYPTO 框架错误码方案

本文档说明 PYPTO 各组件错误码的整体范围与约定。各组件具体错误码定义与排查说明见下方链接。

## 1. 总体范围

所有 ASSERT ERROR 报错均需由错误码承载。各组件错误码范围如下：

| 范围             | 组件/用途     |
|------------------|---------------|
| F1XXXX           | 外部限制（用户写法问题、场景限制、版本兼容性等） |
| F2-F3XXXX        | FUNCTION      |
| F4-F5XXXX        | PASS          |
| F6XXXX           | CODEGEN       |
| F7-F8XXXX        | MACHINE       |
| F9XXXX           | SIMULATION    |
| FAXXXX           | DISTRIBUTED   |
| FBXXXX           | VERIFY        |
| FCXXXX           | OPERATION     |
| FC0-FC2XXX       | VECTOR        |
| FC3-FC5XXX       | MATMUL        |
| FC6-FC8XXX       | CONV          |
| FC9XXX           | 视图类OP      |

## 2. 各组件错误码文档

| 组件        | 范围       | 文档 |
|-------------|------------|------|
| FUNCTION    | F2-F3XXXX  | [function.md](function.md) |
| PASS        | F4-F5XXXX  | [pass.md](pass.md) |
| CODEGEN     | F6XXXX     | [codegen.md](codegen.md) |
| MACHINE     | F7-F8XXXX  | [machine.md](machine.md) |
| SIMULATION  | F9XXXX     | [simulation.md](simulation.md) |
| DISTRIBUTED | FAXXXX     | [distributed.md](distributed.md) |
| VERIFY      | FBXXXX     | [verify.md](verify.md) |
| OPERATION   | FCXXXX     | [operation.md](operation.md) |
| VECTOR      | FC0-FC2XXX | [vector.md](vector.md) |
| MATMUL      | FC3-FC5XXX | [matmul.md](matmul.md) |
| CONV        | FC6-FC8XXX | [conv.md](conv.md) |
| 视图类OP     | FC9XXX     | [view_op.md](view_op.md) |

## 3 原则

- **报错与日志**：各组件 ASSERT/CHECK/ERROR 报错处须使用本组件 error 头文件中的错误码，并在日志或异常中携带该码。
- **按组件管理**：各组件错误码头文件在各自组件目录下维护，不集中到统一头文件。
- **文档补充**：若单靠 ErrorMsg 无法说明原因或难以定位，须为该错误码补充 trouble_shooting 文档（原因、排查步骤与解决方案）。
- **关联 Skill**：各组件错误码文档中可为每个错误码注明 **关联 Skill**，指向 [.opencode/skills](../.opencode/skills) 下对应技能（如 `pypto-environment-setup`），便于排查时加载该技能进行环境诊断、算子开发或性能调优等。
- **大流程到子流程**：错误码分类应先梳理大流程（Category），再细化子流程（Scene）；头文件内按此顺序组织枚举。
- **单一职责**：一个错误码只对应一种场景（一类问题），语义单一、便于根据码值快速定位原因。

## 4 用法举例（以 MACHINE 为例）

报错或打 ERROR 日志时带上错误码，日志中显示为 **Fxxxxx**（如 F70004）。枚举值即错误码，可直接转成数值使用。例如原日志：

```cpp
MACHINE_LOGE("xxx failed.");
```

应改为带错误码的写法（枚举直接转数值打印）：

```cpp
#include "machine/utils/machine_error.h"

MACHINE_LOGE(SchedErr::HANDSHAKE_TIMEOUT, "hand shake timeout.");
```

效果
[ERROR] PYPTO(411657):2026-03-13 14:25:42.400 [host_machine.cpp:135][MACHINE]:ErrCode: F70006! hand shake timeout.