# 简介

本文档明确框架错误码及日志配置规范，并提供问题定位指南。

## DFX目标与要求

1. **问题归属可区分**：打屏报错必须携带错误码，支持快速区分“外部写法问题”与“框架内部问题”。
2. **外部问题可修复**：外部写法问题须通过`ErrorMsg`明确“哪里错、为什么错、如何改”。
3. **内部问题可定界**：框架内部问题须通过`ErrorCode`直接定界到对应组件。
4. **根因定位可落地**：框架内部问题支持结合`plog`日志与`docs/zh/trouble_shooting/*.md`组件文档定位并修复。

## ErrorCode

### 范围映射与组件文档

所有中断流程中的`PyptoError/CHECK/ASSERT/ERROR`均需由错误码承载。范围映射如下：

| 范围       | 归属与用途 | 文档 |
|------------|------------|------|
| F0XXXX     | 外部写法问题 | - |
| F1XXXX     | 框架内部公共问题 | - |
| F2-F3XXXX  | FUNCTION组件内部问题 | [function.md](function.md) |
| F4-F5XXXX  | PASS组件内部问题 | [pass.md](pass.md) |
| F6XXXX     | CODEGEN组件内部问题 | [codegen.md](codegen.md) |
| F7-F8XXXX  | MACHINE组件内部问题 | [machine.md](machine.md) |
| F9XXXX     | SIMULATION组件内部问题 | [simulation.md](simulation.md) |
| FAXXXX     | DISTRIBUTED组件内部问题 | [distributed.md](distributed.md) |
| FBXXXX     | VERIFY组件内部问题 | [verify.md](verify.md) |
| FCXXXX     | OPERATION组件内部问题 | [operation.md](operation.md) |
| FC0-FC2XXX | VECTOR子类内部问题 | [vector.md](vector.md) |
| FC3-FC5XXX | MATMUL子类内部问题 | [matmul.md](matmul.md) |
| FC6-FC8XXX | CONV子类内部问题 | [conv.md](conv.md) |
| FC9XXX     | 视图类OP子类内部问题 | [view_op.md](view_op.md) |

### 规范原则

- **统一定义**：错误码统一定义在`framework/include/tilefwk/error_code.h`，组件侧头文件仅做兼容包含。
- **归属一致**：`PyptoError/CHECK`表示外部写法问题（`F0XXXX`）；`ASSERT/ERROR`表示框架内部问题（`F1XXXX`及之后组件范围）。
- **文档可追溯**：若单靠`ErrorMsg`无法说明原因或难以定位，需在`docs/zh/trouble_shooting/*.md`补充原因、排查步骤与解决方案。
- **单码单义**：一个错误码仅对应一个场景，避免一码多义。
- **Skill可联动**：可在组件文档中标注关联Skill（如`pypto-environment-setup`），辅助自动化排查。

## 日志环境变量

以下变量用于控制CANN日志输出行为（级别、打屏、落盘路径、文件数量等）。详情参考昇腾社区官方文档取值约束，本节仅给出常用作用与示例。

| 环境变量 | 作用（简要） | 示例 |
|---|---|---|
| `ASCEND_GLOBAL_LOG_LEVEL` | 设置全局日志级别（控制整体日志详细程度）。 | `export ASCEND_GLOBAL_LOG_LEVEL=0` |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否打屏输出日志；`1`表示打屏，`0`表示按默认方式落盘。 | `export ASCEND_SLOG_PRINT_TO_STDOUT=1` |
| `ASCEND_MODULE_LOG_LEVEL` | 按模块设置日志级别（用于定向放大某些模块日志）。 | `export ASCEND_MODULE_LOG_LEVEL="PASS=0:PYPTO=1"` |
| `ASCEND_GLOBAL_EVENT_ENABLE` | 控制全局事件日志开关（用于事件类问题排查）。 | `export ASCEND_GLOBAL_EVENT_ENABLE=1` |
| `ASCEND_HOST_LOG_FILE_NUM` | 控制单进程日志文件保留数量（超出后滚动删除最早日志）。 | `export ASCEND_HOST_LOG_FILE_NUM=1000` |
| `ASCEND_PROCESS_LOG_PATH` | 指定进程日志落盘目录（不存在时会自动创建）。 | `export ASCEND_PROCESS_LOG_PATH=/tmp/ascend_plog` |
| `ASCEND_WORK_PATH` | 指定CANN运行工作目录，PyPTO编译产物（output目录、kernel_aicore/kernel_aicpu等）统一落盘至`$ASCEND_WORK_PATH/pypto`。 | `export ASCEND_WORK_PATH=/tmp/ascend_work` |

组合示例（调试时常用）：

```bash
export ASCEND_MODULE_LOG_LEVEL=PASS=0:PYPTO=1即设置PyPTO对应PASS组件日志级别为debug，其余日志级别为info
export ASCEND_HOST_LOG_FILE_NUM=1000
export ASCEND_PROCESS_LOG_PATH=/tmp/ascend_plog
```
