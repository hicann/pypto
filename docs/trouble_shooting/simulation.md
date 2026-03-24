# SIMULATION 组件错误码

- **范围**：F9XXXX
- 本文档说明 SIMULATION 组件的错误码定义、场景说明与排查建议。


## 错误码定义与使用说明

相关错误码的统一定义，参见 `framework/src/cost_model/simulation/utils/simulation_error.h` 文件。


## 排查建议

根据日志中不同ErrorCode关联到下述排查建议：

### EXTERNAL_ERROR 外部错误
#### INVALID_CONFIG
1. **检查配置是否合法**：确认传入的配置是否满足长度、格式等要求
2. **检查配置是否存在**：确认传入的配置是否是仿真所需配置

#### CONFIG_OUT_OF_RANGE
1. 检查配置数量是否超出uint64_t范围

#### INVALID_CONFIG_NAME
1. 检查配置名称是否符合要求，仿真配置详情参见`framework/src/cost_model/simulation/config/xxx.h`中

#### PERMISSION_CHECK_ERROR
1. 检查文件是否有读取或写入权限

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



### FORWARD_SIM

#### INVALID_PIPE_TYPE
1. **无效的pipe类型**：请在`framework/src/cost_model/simulation/common/ISA.h`的SCHED_CORE_PIPE_TYPE数据结构中添加新的pipe类型

#### INVALID_DATA_TYPE
1. 请检查数据类型是否合法有效

#### DEAD_LOCK
1. 请在`output/output_xxx/CostModelSimulationOutput/graphs`下的找到报错对应的dot文件进行分析
2. 如果还无法定位到根因请联系管理员解决

#### FUNC_NOT_SUPPORT
1. 请在`framework/src/cost_model/simulation_ca/A2A3/model/def.h`的GetProgram中增加新的cce指令



### PRECISION_SIM

#### NO_SO_EXISTS
1. 检查精度仿真所需的.so文件是否存在并放在正确的路径下

#### CANN_LOAD_FAILED
1. **确认是否加载了CANN环境**: `source xxx/set_env.sh`


## 通用排查建议

### 1. 启用详细日志

在遇到 SIMULATION 组件错误时，可以启用详细日志获取更多信息：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=PYPTO=0 # 0：Debug，1：Info，2：Warning，3：Error(默认)
export ASCEND_PROCESS_LOG_PATH=./debug_logs # 指定日志落盘路径
```

如果要把PYPTO日志改为在终端输出，可以输入以下命令：

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # 日志由落盘改为终端输出
```
