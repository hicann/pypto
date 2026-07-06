# 环境变量参考

本文档汇总了 PyPTO 开发、编译、运行和调试过程中涉及的所有环境变量，按功能分类。各变量的详细说明请参见对应的专题文档。

---

## 一、环境配置

### ASCEND_HOME_PATH

#### 功能描述
CANN toolkit 根目录路径，通过 `set_env.sh` 脚本配置。PyPTO 通过该变量判断当前环境是否具备 NPU 运行能力：若该变量已设置，默认在 NPU 上执行；否则回退到仿真模式。同时也是定位 pto-isa 头文件和库文件的基准路径。

- 类型：字符串（绝对路径）
- 默认值：无（需通过 `source set_env.sh` 配置）

#### 配置示例
```bash
# 默认路径安装（root 用户）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 指定路径安装
source ${install_path}/ascend-toolkit/set_env.sh
```

#### 使用约束
- 必须在运行 PyPTO 前完成配置，否则框架将自动回退到仿真模式。
- 该变量由 CANN 安装包提供，不应手动硬编码。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### PTO_TILE_LIB_CODE_PATH

#### 功能描述
pto-isa 源码路径，用于编译和运行 PyPTO 算子。CANN toolkit 安装后自带 pto-isa，通常无需手动设置。当内置版本不满足需求时，可通过该环境变量指向独立的 pto-isa 源码目录。

- 类型：字符串（绝对路径）
- 默认值：CANN toolkit 内置路径

#### 配置示例
```bash
# 使用 CANN 内置 pto-isa（推荐）
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/$(uname -m)-linux

# 使用独立 pto-isa 源码
git clone https://gitcode.com/cann/pto-isa.git
export PTO_TILE_LIB_CODE_PATH="$PWD/pto-isa"
```

#### 使用约束
- 路径下必须包含 `include/pto/` 目录。
- PyPI 方式安装 PyPTO 时，pto-isa 已随 CANN 包安装，无需单独设置。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### TILE_FWK_DEVICE_ID

#### 功能描述
指定 PyPTO 算子执行时使用的 NPU 设备卡号。框架和测试用例通过该变量确定目标设备。

- 类型：整数
- 取值范围：`0` ~ `N-1`（N 为可用 NPU 数量）
- 默认值：`0`

#### 配置示例
```bash
export TILE_FWK_DEVICE_ID=0
```

#### 使用约束
- 指定的设备卡号必须为可用状态，可通过 `npu-smi info` 确认。
- 多进程场景下应避免多个进程使用同一设备卡号。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### TILE_FWK_DEVICE_ID_LIST

#### 功能描述
分布式测试场景下，指定使用的 NPU 设备组。多个设备 ID 以逗号分隔，用于多卡分布式用例执行。

- 类型：字符串（逗号分隔的整数列表）
- 默认值：无

#### 配置示例
```bash
export TILE_FWK_DEVICE_ID_LIST="0,1,2,3"
```

#### 使用约束
- 仅在分布式测试场景下使用。
- 指定的设备数量需满足分布式用例的卡数要求。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### TILE_FWK_OUTPUT_DIR

#### 功能描述
指定 PyPTO 编译产物和运行结果的输出根目录。设置后，所有 output 文件（计算图、泳道图、verify 数据等）将落盘到该目录下。

- 类型：字符串（路径）
- 默认值：当前工作目录下的 `./output`

#### 配置示例
```bash
export TILE_FWK_OUTPUT_DIR=/tmp/pypto_output
```

#### 使用约束
- 当 `compile_debug_mode=2`（固定 CCE）时，设备侧代码输出路径由 `ASCEND_WORK_PATH` 决定，`TILE_FWK_OUTPUT_DIR` 不参与设备侧代码路径计算。
- 该变量优先级高于默认路径，但不影响日志落盘路径。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

## 二、编译与构建

### PYPTO_THIRD_PARTY_PATH

#### 功能描述
指定 PyPTO 编译所需的第三方开源软件源码包路径。当编译环境无法访问 `cann-src-third-party` 仓库自动下载时，需手动准备源码包并通过该变量指定路径。

- 类型：字符串（绝对路径）
- 默认值：无（自动下载）

#### 配置示例
```bash
# 手动准备第三方源码包后设置
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>

# 然后执行编译
python3 -m pip install . --verbose
```

#### 使用约束
- 路径下需包含 `json-3.11.3` 和 `libboundscheck-v1.1.16` 的源码包。
- 仅在源码编译安装时生效，PyPI 安装无需设置。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### PYPTO_BUILD_EXT_ARGS

#### 功能描述
向 PyPTO 编译过程（`setup.py`）传递额外的 C++ 编译选项。可编辑安装模式下，`setuptools` 不支持通过 `pip --config-setting` 传递配置，需通过该变量预设编译参数。

- 类型：字符串（空格分隔的编译选项）
- 默认值：无

#### 配置示例
```bash
# 编译 Debug 版本并开启编译器详细输出
export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug --cmake-verbose'

# 指定 CMake Generator 为 Unix Makefiles
export PYPTO_BUILD_EXT_ARGS='--cmake-build-type=Debug --cmake-verbose --cmake-generator="Unix Makefiles"'

# 执行可编辑安装
python3 -m pip install -e . --verbose
```

#### 使用约束
- 仅在源码编译安装时生效。
- 常规安装（非 `-e` 模式）应使用 `pip --config-setting` 参数传递编译选项。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### MPI_HOME

#### 功能描述
MPI 安装路径，用于分布式用例的编译和运行。PyPTO 的分布式功能依赖 MPI（推荐版本 >= 3.2.1）。

- 类型：字符串（绝对路径）
- 默认值：无

#### 配置示例
```bash
export MPI_HOME=/usr/local/mpich
export PATH=${MPI_HOME}/bin:${PATH}
```

#### 使用约束
- 仅在需要运行分布式用例时必需。
- 需确保 MPI 已正确安装且 `mpirun` 可执行。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

## 三、日志与调试

### ASCEND_GLOBAL_LOG_LEVEL

#### 功能描述
设置 CANN 全局日志级别，控制整体日志详细程度。

- 类型：整数
- 取值范围：
  - `0`：DEBUG（最详细）
  - `1`：INFO
  - `2`：WARN
  - `3`：ERROR（默认）
  - `4`：NULL（关闭日志）
- 默认值：`3`（ERROR）

#### 配置示例
```bash
# 开启 DEBUG 级别日志（排查问题时常用）
export ASCEND_GLOBAL_LOG_LEVEL=0

# 开启 INFO 级别日志
export ASCEND_GLOBAL_LOG_LEVEL=1
```

#### 使用约束
- DEBUG 级别日志量较大，可能影响执行性能，仅建议在排查问题时开启。
- 支持按模块精细控制，参见 `ASCEND_MODULE_LOG_LEVEL`。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### ASCEND_SLOG_PRINT_TO_STDOUT

#### 功能描述
控制日志是否输出到终端标准输出。默认情况下日志落盘到文件，设置该变量后日志将改为输出到终端。

注意：不同日志子系统的行为存在差异。Host log manager（`log_manager.cpp`）在打屏模式下会跳过文件日志初始化，日志不会落盘；interpreter logger（`interpreter_log.cpp`）在打屏模式下会同时输出到终端和文件。因此，排查问题时如需保留完整的文件日志，建议不要开启该变量，而是通过 `ASCEND_PROCESS_LOG_PATH` 指定落盘目录后查看文件日志。

- 类型：整数
- 取值范围：`0`（落盘，默认）、`1`（打屏输出）
- 默认值：`0`

#### 配置示例
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

#### 使用约束
- 打屏模式下 host log 不会落盘，可能导致排查问题时遗漏关键日志。如需保留文件日志，请勿开启该变量。
- 打屏模式下日志量可能较大，建议配合 `ASCEND_GLOBAL_LOG_LEVEL` 控制输出级别。
- 在仿真环境中排查 `.so` 加载问题时推荐使用。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### ASCEND_MODULE_LOG_LEVEL

#### 功能描述
按模块设置日志级别，用于定向放大某些模块的日志输出，而不影响全局日志级别。格式为 `模块名=级别`，多个模块以冒号分隔。

- 类型：字符串
- 格式：`MODULE1=LEVEL1:MODULE2=LEVEL2`
- 已知模块名：`PASS`、`PYPTO`、`CODEGEN`、`MACHINE`
- 级别取值：`0`（DEBUG）、`1`（INFO）、`2`（WARN）、`3`（ERROR）
- 默认值：无

#### 配置示例
```bash
# 设置 PASS 模块为 DEBUG，PYPTO 模块为 INFO
export ASCEND_MODULE_LOG_LEVEL="PASS=0:PYPTO=1"

# 设置 CODEGEN 模块为 INFO
export ASCEND_MODULE_LOG_LEVEL=CODEGEN=1
```

#### 使用约束
- 模块名不区分大小写。
- 该变量优先级高于 `ASCEND_GLOBAL_LOG_LEVEL`，即对指定模块使用此处设置的级别，其余模块仍使用全局级别。
- 格式错误（如缺少 `=` 或级别为空）时，对应模块的配置会被忽略。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### ASCEND_GLOBAL_EVENT_ENABLE

#### 功能描述
控制全局事件日志开关，用于事件类问题的排查。开启后，框架内部的关键事件（如 Verify 校验结果、Pass 执行状态等）将以 EVENT 级别记录到日志中。

- 类型：整数
- 取值范围：`0`（关闭）、`1`（开启）
- 默认值：关闭

#### 配置示例
```bash
export ASCEND_GLOBAL_EVENT_ENABLE=1
```

#### 使用约束
- 事件日志主要用于问题排查，日常运行无需开启。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### ASCEND_HOST_LOG_FILE_NUM

#### 功能描述
控制单进程日志文件保留数量。当日志文件数量超过设定值时，自动滚动删除最早的日志文件。

- 类型：整数
- 取值范围：正整数（`0` 或负数将使用默认行为）
- 默认值：框架内置默认值

#### 配置示例
```bash
# 保留最近 1000 个日志文件
export ASCEND_HOST_LOG_FILE_NUM=1000
```

#### 使用约束
- 设置过小时可能导致排查问题时日志已被滚动删除。
- 建议排查问题时设置为较大值（如 `1000`）。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### ASCEND_PROCESS_LOG_PATH

#### 功能描述
指定进程日志（plog）的落盘目录。日志文件按进程和线程分别生成，目录不存在时会自动创建。

- 类型：字符串（路径）
- 默认值：无（使用 CANN 默认路径）

#### 配置示例
```bash
export ASCEND_PROCESS_LOG_PATH=/tmp/ascend_plog

# 调试时常用组合
export ASCEND_PROCESS_LOG_PATH=$(pwd)/logs
export ASCEND_GLOBAL_LOG_LEVEL=0
```

#### 使用约束
- 确保指定路径具有写权限。
- 日志文件命名格式为 `pypto-log-*.log`。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### ASCEND_WORK_PATH

#### 功能描述
指定 CANN 运行工作目录，同时也作为 PyPTO 编译产物的落盘根路径。PyPTO 的 output 目录、kernel_aicore / kernel_aicpu 等设备侧代码统一输出至 `$ASCEND_WORK_PATH/pypto` 下。

- 类型：字符串（路径）
- 默认值：当前工作目录

#### 配置示例
```bash
export ASCEND_WORK_PATH=/tmp/ascend_work
```

#### 使用约束
- 当 `compile_debug_mode=2`（固定 CCE）时，设备侧代码输出路径为 `$ASCEND_WORK_PATH/pypto/<name>`。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

## 四、性能采集

### DUMP_DEVICE_PERF

#### 功能描述
使能 AI CPU / AI Core 联合性能数据采集。开启后，框架在执行过程中采集 AI CPU 调度与 AI Core 执行过程中端到端的耗时数据，并能够在终端打屏，输出相关性能统计信息。采集结果落盘到 `output/output_时间戳/` 目录下。

- 类型：字符串
- 取值范围：`true`（开启）、`false` 或未设置（关闭）
- 默认值：关闭

#### 配置示例
```bash
export DUMP_DEVICE_PERF=true
python3 examples/02_intermediate/operators/softmax/softmax.py

# 采集完成后分析数据
python tools/scripts/machine_perf_trace.py analyze output/output_<时间戳>/machine_trace_perf_data_0.json
```

#### 使用约束
- 最多支持 200 轮数据采集和打屏，超出部分将被截断。
- 当前只支持采集 200 次 devTask 构建的数据，超出时日志中会出现告警。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### PROF_PMU_EVENT_TYPE

#### 功能描述
选择 PMU（Performance Monitoring Unit）数据采集模式。PMU 用于监控 AICore 的硬件性能事件，不同模式对应不同的事件组。

- 类型：整数
- 取值范围：`1`、`2`、`4`、`5`、`6`、`7`、`8`
- 默认值：`2`

#### 配置示例
```bash
export PROF_PMU_EVENT_TYPE=2

# 采集 PMU 数据
msprof --task-time=l3 --output=./prof_data python xxx.py

# 解析数据
python tools/profiling/tilefwk_pmu_to_csv.py -p PROF_xxx/device_x/data -pe=$PROF_PMU_EVENT_TYPE --arch dav_3510
```

#### 使用约束
- 采集前需修改源码中的编译宏（`PMU_COLLECT=1`、`PERF_PMU_TEST_SWITCH=1`）并重新编译，当前仅支持串行采集。完整的编译宏适配、采集模式选择、数据采集与解析流程详见[采集PMU数据](../debug/performance.md)。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

## 五、运行时控制

### PYPTO_LAUNCH_SCHED_SAME_CLUSTER

#### 功能描述
控制是否强制在同一 Cluster 内分配调度线程（AICPU）。在同一 Cluster 分配调度线程能够获得更好的核间流水性能，但在整网场景中，除 PyPTO 外还有其他组件使用 AICPU，强制同 Cluster 可能因 AICPU 资源不足导致执行超时。

- 类型：字符串
- 取值范围：`true`（默认，同 Cluster 分配）、`false`（不强制同 Cluster）
- 默认值：`true`

#### 配置示例
```bash
# 整网场景下关闭同 Cluster 约束，避免 AICPU 资源不足
export PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false
```

#### 使用约束
- 设置为 `false` 时，可以配合 `launch_sched_aicpu_num`（通过 `runtime_options` 配置）指定可用的 AICPU 数量。
- 开启同 Cluster 分配时，`launch_sched_aicpu_num` 配置不生效。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### PTO_DATADUMP_ENABLE

#### 功能描述
使能上板执行时的 tensor dump 功能。开启后，框架在 NPU 上执行算子时会自动 dump leaf function 的输入输出数据，用于与模拟计算结果进行精度对比分析。

- 类型：字符串
- 取值范围：`true`（开启）、`false` 或未设置（关闭）
- 默认值：关闭

#### 配置示例
```bash
# 方式一：命令行设置
export PTO_DATADUMP_ENABLE=true

# 方式二：Python 代码中设置
import os
os.environ["PTO_DATADUMP_ENABLE"] = "true"
```

#### 使用约束
- 需配合 `verify_options` 中的 `enable_pass_verify` 和 `pass_verify_save_tensor` 使用。
- dump 数据落盘在 `output/output_*/tensor/` 目录下。
- 仅在 NPU 上板执行时生效，仿真模式下无效。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

---

### TORCH_DEVICE_BACKEND_AUTOLOAD

#### 功能描述
控制 PyTorch（版本 > 2.5）是否自动加载所有 `torch.backends` 扩展。当环境中已安装 `torch_npu` 但未安装 CANN 时，自动加载会因找不到依赖项而报错 `ImportError: libhccl.so`。设置该变量为 `0` 可禁用自动加载，避免启动异常。

- 类型：整数
- 取值范围：`0`（禁用自动加载）、`1`（启用，默认）
- 默认值：`1`

#### 配置示例
```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

#### 使用约束
- 仅在 PyTorch 版本 > 2.5 且环境中 CANN 未完整安装时需要设置。
- 正常安装 CANN 的环境中无需设置。

#### 支持的型号
- Ascend 950PR
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品


