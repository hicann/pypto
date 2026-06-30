# SIMULATION组件错误码

- **范围**：F9XXXX
- 本文档说明SIMULATION组件的错误码定义、场景说明与排查建议。

## 错误码定义与使用说明

相关错误码的枚举与码值统一定义在`framework/include/tilefwk/error_code.h`（仿真侧见`CostModel`命名空间下各枚举）。

## 排查建议

根据日志中不同ErrorCode关联到下述排查建议：

### EXTERNAL_ERROR外部错误

#### INVALID_CONFIG

1. **检查配置是否合法**：确认传入的配置是否满足长度、格式等要求
2. **检查配置是否存在**：确认传入的配置是否是仿真所需配置

#### INVALID_CONFIG_NAME

1. 检查配置名称是否符合要求，仿真配置详情参见`framework/src/cost_model/simulation/config/xxx.h`。

#### FILE_FORMAT_ERROR

1. **检查文件格式是否错误**：确认json文件的内容是否符合JSON格式要求

#### FILE_CONTENT_ERROR

1. 检查文件的内容是否符合约定的要求

#### INVALID_PATH

1. 检查文件路径是否正确

#### FILE_OPEN_FAILED

1. 确认文件是否有读取权限
2. 确认文件路径是否正确
3. 确认文件是否已损坏

#### PYTHON_CMD_ERROR

1. 检查执行Python命令是否正确，确认Python环境及依赖是否可用

### INTERNAL_ERROR内部错误

内部错误请联系仿真的oncall解决

### FORWARD_SIM

#### BUILD_FUNCTION_ERROR

1. 构建Function错误，请联系仿真的oncall解决

#### SIMULATION_INIT_ERROR

1. CostModel初始化错误，请联系仿真的oncall解决

#### SCHEDULE_TASK_ERROR

1. 任务调度出错，请联系仿真的oncall解决

#### RESOLVE_DEPENDENCY_ERROR

1. 依赖解析错误，请联系仿真的oncall解决

#### SIMULATION_RUN_ERROR

1. 仿真运行时异常，请联系仿真的oncall解决

#### INVALID_PIPE_TYPE

1. **无效的pipe类型**：请在`framework/src/cost_model/simulation/common/ISA.h`的SCHED_CORE_PIPE_TYPE数据结构中添加新的pipe类型

#### SHAPE_INVALID

1. 输入数据无效的shape，请检查输入数据的shape是否有效

#### CYCLES_ERROR

1. 时钟周期错误，请联系仿真的oncall解决

#### CALENDAR_ERROR

1. 日历调度异常，请联系仿真的oncall解决

#### DEAD_LOCK

1. 请在`output/output_xxx/CostModelSimulationOutput/graphs`下找到报错对应的dot文件进行分析
2. 如果还无法定位到根因请联系管理员解决

### POST_SIM

#### UNKNOWN

1. 后仿真阶段未知错误，请联系仿真的oncall解决

### PRECISION_SIM

#### NO_SO_EXISTS

1. 检查精度仿真所需的.so文件是否存在
2. **通过日志定位缺失的.so路径**：先开启仿真日志输出 `export ASCEND_SLOG_PRINT_TO_STDOUT=1`，然后在日志中搜索 `can not load library:`，找到缺失的.so文件路径

#### CANN_LOAD_FAILED

1. **确认是否加载了CANN环境**: `source xxx/set_env.sh`

#### CMD_ERROR

1. 检查打印出来的终端命令是否正确

#### LEAF_CALLEE_ATTR_NULL

1. 叶子被调用函数属性为空，请联系仿真的oncall解决

#### CANNSIM_FAILED

1. **通过cannsim启动**：使用 `cannsim record 'python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950` 方式启动精度仿真

## 通用排查建议

### 1. 启用详细日志

在遇到SIMULATION组件错误时，可以启用详细日志获取更多信息：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=PYPTO=0 # 0：Debug，1：Info，2：Warning，3：Error(默认)
export ASCEND_PROCESS_LOG_PATH=./debug_logs # 指定日志落盘路径
```

如果要把PYPTO日志改为在终端输出，可以输入以下命令：

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # 日志由落盘改为终端输出
```
