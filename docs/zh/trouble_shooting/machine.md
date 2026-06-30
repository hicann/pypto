# MACHINE组件错误码

- **范围**：F7-F8XXXX
- 本文档说明MACHINE组件的错误码定义、场景说明与排查建议。

---

## 错误码定义与使用说明

相关错误码的枚举与码值统一定义在 [`framework/include/tilefwk/error_code.h`](../../../framework/include/tilefwk/error_code.h)（见MachineError、HostBackEndErr、SchedErr、RtErr等）。

## 排查建议

### AICORE ERROR/The aicore execution is abnormal

以下提供三种定位方式，**推荐优先使用Agent自动定位或一键脚本**。

### Agent自动定位

在OpenCode中直接描述问题，AI agent自动执行全流程，无需手动敲命令。

#### 触发方式

```text
我的测试用例出现了 aicore error，测试命令是 python3 test_my_op.py，路径在 ./
```

#### 三条执行路径

AI agent根据用户输入自动分流：

| 用户输入 | 执行路径 |
|---|---|
| 仅提供test_cmd | 完整流程（步骤1-6） |
| 提供CCE文件路径（含`.cpp`文件） | 模式A：从二分查找开始 |
| 提供CCE文件+问题代码行号 | 模式B：直达源码映射 |

#### 快捷模式对话示例

**模式A**（已知CCE文件）：

```text
帮我定位 aicore error 的问题代码行，CCE 文件是 /path/to/TENSOR_xxx.cpp，
测试命令是 python3 test_my_op.py
```

**模式B**（已知CCE文件+行号）：

```text
帮我映射 aicore error 的源码位置，CCE 文件是 /path/to/TENSOR_xxx.cpp，
问题行号是 42
```

#### Pypto_Test框架适配

在描述中提及"Pypto_Test框架"，AI agent自动识别（不区分大小写、下划线/中划线），并为所有步骤传递适配参数：

```text
我的测试用例出现了 aicore error，用的是 Pypto_Test 框架，测试命令是 python3 test_my_op.py
```

**关联SKILL**：[pypto-aicore-error-locator](../../../.agents/skills/pypto-aicore-error-locator/SKILL.md)

---

### 一键脚本定位

**关联脚本**：[locate_aicore_error.py](../../../tools/scripts/locate_aicore_error.py)—一键完成定位，所有模式均支持`--use-pypto-test-framework`。

#### 完整流程

```bash
python3 tools/scripts/locate_aicore_error.py \
    --pypto-path <pypto_path> \
    --run-path <run_path> \
    --test-cmd <test_cmd>
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--pypto-path` | 否 | pypto项目根目录（默认`./`） |
| `--run-path` | 否 | 运行测试的目录（默认`./`） |
| `--test-cmd` | 完整流程必填 | 触发aicore error的测试命令 |
| `--device-log-path` | 否 | device log落盘路径（默认`./wk`） |
| `--skip-machine-check` | 否 | 跳过步骤2（machine框架调度问题检查） |
| `--skip-rebuild` | 否 | 跳过重新编译（使用当前安装的pypto） |
| `--use-pypto-test-framework` | 否 | 使用Pypto_Test框架模式 |

#### 模式A：已知CCE文件

已定位到问题CCE文件，从二分查找开始：

```bash
python3 tools/scripts/locate_aicore_error.py \
    --cce-file <cce_file_path> \
    --test-cmd <test_cmd> \
    --run-path <run_path>
```

自动完成：验证CCE文件 → 二分查找 → 源码映射。

#### 模式B：已知CCE文件+行号

已定位到CCE问题代码行，直达源码映射，**不需要`--test-cmd`**：

```bash
python3 tools/scripts/locate_aicore_error.py \
    --cce-file <cce_file_path> \
    --cce-line <line_number> \
    --run-path <run_path>
```

---

<details>
<summary>人工手动分步定位指导（展开查看完整步骤）</summary>

**前置步骤：初始测试—确认aicore error可复现**

在修改任何代码之前，先运行一次测试确认aicore error确实存在：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1 && <test_cmd>
```

检查打屏输出，必须出现`aicore error`。**如果未出现，则不适用该定位方法！**

> 使用Pypto_Test框架时，所有步骤中标注`[--use-pypto-test-framework]`的参数均需添加。

### 1. 排除machine框架调度问题

> **说明**：使用单个脚本自动完成全部操作（定位aicore_entry.h → 注释CallSubFuncTask → 运行测试 → 恢复文件），**直接修改已安装pypto包，无需重新编译安装**。

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/exclude_machine_framework.py \
  --test-cmd <test_cmd> \
  --run-path <run_path> \
  [--use-pypto-test-framework]
```

**判断**：

- **脚本退出码0**（注释后无aicore error）：问题在**kernel代码**中 → 继续后续步骤
- **脚本退出码1**（注释后仍有aicore error）：问题在**machine调度框架** → 停止
- **脚本退出码2**（定位aicore_entry.h或CallSubFuncTask失败）：手动排查环境

### 2. 定位问题CCE文件

> **说明**：使用单个脚本自动完成全部操作（启用追踪日志 → 条件重编译 → 运行测试 → 分析日志 → 定位并验证CCE文件）。脚本内部自动处理并行编译错误（`ld.lld: error: undefined`）。

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/locate_problem_cce.py \
  --pypto-path <pypto_path> \
  --test-cmd <test_cmd> \
  --run-path <run_path> \
  --device-log-path <device_log_path> \
  [--use-pypto-test-framework]
```

**判断**：

- **脚本退出码0**：成功定位，输出`CCE_FILE=<path>`和`PROGRAM_JSON=<path>`，记录这两个路径供后续步骤使用
- **脚本退出码1**：未找到问题CCE文件 → 停止
- **脚本退出码2**：并行编译错误无法自动修复（parallel_compile已为1但仍报错），**停止执行**

### 3. 准备二分查找—确定错误范围+获取初始范围

> **说明**：单个脚本合并原有两步操作：先注释所有T操作行并测试确定错误是否在T操作中，再根据结果计算二分查找的初始范围。

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/setup_binary_search.py <cce_file> <test_cmd> <run_path> [--use-pypto-test-framework]
```

结果说明：输出`ERROR_IN_T=True|False`、`LEFT=<left>`、`RIGHT=<right>`三个值。`ERROR_IN_T=True`时仅对T操作行二分；`ERROR_IN_T=False`时对除同步行外的所有行二分。

### 4. 执行二分查找迭代

> **说明**：此步骤通过迭代注释缩小范围，每次迭代仅执行一条命令（受超时限制），需多轮逐步收敛至问题行。

多轮迭代执行以下命令，每轮根据输出的`NEXT_LEFT` / `NEXT_RIGHT`更新参数：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/binary_search_iteration.py <cce_file> <test_cmd> <run_path> <left> <right> <error_in_t> [--use-pypto-test-framework]
```

结果说明：输出新的`NEXT_LEFT`和`NEXT_RIGHT`值。当`NEXT_LEFT == NEXT_RIGHT`时，输出`FOUND <problem_line>`，即为问题代码行。

### 5. 映射到前端源代码

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/locate_source_line.py <cce_file> <program_json_path> <problem_line>
```

结果说明：通过funcHash在program.json中匹配，输出前端源代码文件路径和行号。

### 6. 恢复初始状态

定位完成后，使用单个脚本恢复步骤2修改的配置文件并重新编译安装pypto：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/restore_initial_state.py --pypto-path <pypto_path>
```

脚本自动完成：恢复`tile_fwk_config.json`和`device_switch.h`的`.backup`备份 → 重新编译安装pypto。

</details>

---

### 怀疑和MACHINE内存处理有关的精度问题

1. **检查输入初始化**：
保证输入/输出初始化

2. **检查Tensor连续性**：
部分MACHINE相关算子或接口要求输入/输出在指定内存布局下**连续**（`tensor.is_contiguous()`为True）。非连续tensor（如某些view、transpose、slice结果）可能导致精度异常或运行错误。
eg：reshape/view、交换维度、索引/切片后的tensor可能非连续；若前序是COPY_IN或尾轴reduce，需在前端或调用侧保证传入的tensor在对应格式下连续。

3. **扩大workspace大小**:
`python/pypto/frontend/parser/entry.py`

```python
workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
```

```python
workspace_tensor = torch.empty(workspace_size * 10, dtype=torch.uint8, device=device)
```

如果问题不复现，则是workspace计算问题

4. **workspace从torch管理，改为内部自管理**
`framework/src/machine/runtime/device_launcher.cpp`

```cpp
    static void PrepareDevProgArgs(DevAscendProgram *devProg, DeviceLauncherConfig &config,
                                  [[maybe_unused]]bool isDevice) {
        ...
        if (config.workspaceAddr) {
            kArgs.workspace = (int64_t *)config.workspaceAddr;
        } else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
            kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        ...
    }
```

```cpp
    static void PrepareDevProgArgs(DevAscendProgram *devProg, DeviceLauncherConfig &config,
                                  [[maybe_unused]]bool isDevice) {
        ...
        if (0) {
            kArgs.workspace = (int64_t *)config.workspaceAddr;
        } else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
            kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        ...
    }
```

如果问题不复现，则是workspace使用问题，存在内存踩踏等

5. **leaf function粒度的内存重叠检测**
（1）打开Operation信息Dump开关
`framework/src/machine/utils/device_switch.h`

```cpp
#define ENABLE_DUMP_OPERATION 1
```

（2）打开DEBUG日志，指定日志落盘路径

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_PROCESS_LOG_PATH=./my_log
```

（3）重新编pypto whl包并安装

（4）启动性能数据采集功能，保证生成dyn_topo.txt

    ```python
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1}
    )
    ```

（5）执行用例

（6）执行内存检测脚本

命令格式：

```bash
python3 tools/schema/schema_memory_check.py -d <device_log_dir_absolute_path> -t <topo_file_absolute_path>
```

示例（请替换为实际绝对路径）：

```bash
python3 tools/schema/schema_memory_check.py -d /path/to/my_log/debug/device-8/ -t /path/to/output/output_20260314_112655_964781_3025352/dyn_topo.txt
```

如无异常，提示device task无内存重叠；
如存在异常，提示内存重叠的device task以及leaf function。

注：
（1）如果报错 memory reuse must happen for full match. 则两个需要内存复用的 rawtensor 范围不一致；
（2）如果报错 memory reuse must happen for same dimension. 则两个内存复用的 rawtensor 的 shape 不一致；
上述两种情况非内存重叠，脚本内存检查依赖不会发生上述情况 ，因此脚本会断言，直接提示日志信息错误。

**关联Skill**：[pypto-memory-overlap-detector](../../../.agents/skills/pypto-memory-overlap-detector/SKILL.md)

6. **复杂特性排除**：
使用`pypto-precision-debug` skill，关闭unroll_list、合轴特性、配置submit_before_loop=True使loop串行执行、确定valid_shape配置正确性、+0.0等，缩小定位范围。

---

### 怀疑精度问题与运行时依赖异常有关

当算子出现精度异常，且怀疑根因是Stitch依赖边丢失（producer尚未写完consumer就提前读取）时，可使用**运行时依赖正确性校验工具**进行系统化分析。

1. **工具入口与位置**：

```txt
tools/verify_dep_correctness.py     # 工具入口
tools/dep_verifier/                 # 规则引擎包
```

2. **开启dump并执行用例**：

在用例的`jit`装饰器中启用`runtime_debug_mode`，框架会在执行期间自动生成校验所需的dump数据：

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 3}
)
def my_op(...):
    ...
```

重新执行用例：

```bash
python your_op_script.py
```

3. **Dump文件说明**：

执行后，日志输出目录（通常在`output/output_<timestamp>/`）下会生成以下文件：

**根目录：**

| 文件 | 生成时机 | 用途 |
|------|---------|------|
| `dyn_topo.txt` | DeviceTask完成后 | 运行时完整依赖拓扑，包含每个task的`seqNo`、`taskId`、静态后继数量、所有后继task列表 |

**`dep_verify_dump/`子目录（框架自动创建）：**

| 文件 | 生成时机 | 用途 |
|------|---------|------|
| `static_topo.csv` | 编译期encode完成后 | 编译期静态拓扑，记录每个function每个op的incast/outcast slot列表及静态后继，是运行时校验的基准 |
| `slot_mapping.csv` | 编译期slot简化完成后 | 前端slot索引 → 运行时slot索引映射，含tensor名称、function名称、slot角色（INPUT/OUTPUT/INOUT/INTERNAL） |
| `slot_cell_table.csv` | 编译期CellMatch初始化后 | 每个slot的stitch策略（partial/fullcover）、cell切分形状、cell数量等元数据 |
| `dyn_stitch_edges.csv` | 每次运行时`HandleOneStitch`建边成功后 | 逐条记录每条stitch边的类型（fullCover/partial/default/reuse）、所经slot、producer/consumer的funcKey、funcIdx、opIdx、taskId |
| `dyn_slot_access.csv` | 运行时stitch阶段cell match路径 | 以cell粒度记录每次写（W）或读（R）事件，含seqNo、slotIdx、funcIdx、opIdx、taskId、命中cell索引列表 |

注：`dyn_topo.txt`在根目录不变；其余5个dump文件统一放在`dep_verify_dump/`子目录，与其他dump隔离。

4. **运行校验工具**：

```bash
python tools/verify_dep_correctness.py <dump_dir>
```

其中`<dump_dir>`为包含上述文件的输出目录，例如：

```bash
python tools/verify_dep_correctness.py ./output/output_20260514_103201_123456_7890123/
```

工具会依次执行以下校验规则：

| 规则ID | 类别 | 校验内容 |
|---------|------|---------|
| `rule_static_integrity` | 缺失依赖 | 编译期`static_topo.csv`声明的函数内部静态后继，必须在`dyn_topo.txt`对应task中完整保留 |
| `rule_stitch_legality` | 非法读写关联 | 每条stitch边引用的producer/consumer op必须在`static_topo`中存在；非reuse类型的边对应slot必须落在producer.outcastSlots ∩ consumer.incastSlots |
| `rule_cell_write_conflict` | 并发写冲突 | 同一seqNo内同一slot同一cell有多个writer，若这些writer在DAG中不构成全序链且非合法并行写，则报错 |

5. **输出结果**：

无问题时控制台打印：

```txt
PASS
```

发现问题时按类别聚合打印摘要，例如：

```txt
FAIL: 3 issue(s) detected.

[Missing producer/consumer dependency (data flow broken)]
  - tensor 'tmp' (slot=3, cell=0): compile-time dependency for kernel funcKey=12, opIdx=2 is not preserved at runtime (missing successor(s) [327682])

[Concurrent write overlap (producers may overwrite each other)]
  - tensor 'tmp' (slot=3, cell=0): 128 producer instances write to the same region without a determined order, later writes may overwrite earlier ones

[Illegal read/write linkage (kernel I/O does not match)]
  - tensor 'out' (slot=7): read/write linkage references an undeclared kernel op (producer funcKey=5/opIdx=1, consumer funcKey=8/opIdx=0)
```

同时在`<dump_dir>/dep_check_report.csv`生成详细报告，列格式为：

```txt
category, rule, slot, tensor, func, cell, message
```

6. **报告字段解读**：

| 字段 | 含义 |
|------|------|
| `category` | 问题类别：`ConcurrentWriteOverlap` / `MissingDependency` / `IllegalReadWriteLinkage` |
| `rule` | 触发的规则ID，例如`rule_cell_write_conflict` |
| `slot` | 运行时slot索引（与`slot_mapping.csv::runtimeSlotIdx`对应） |
| `tensor` | 对应的前端tensor变量名（从`slot_mapping.csv`反查） |
| `func` | 写入该slot的function名称 |
| `cell` | 发生冲突的cell索引（仅cell级规则填写） |
| `message` | 具体描述，含funcKey、opIdx、taskId等定位信息 |

通过`slot`和`tensor`字段可以快速定位到用户代码中的具体tensor；通过`message`中的`funcKey`/`opIdx`可以对照`static_topo.csv`和`dyn_topo.txt`进一步分析依赖丢失的根因。

---

### encode阶段actualRawMagic断言触发

**问题特征**：运行用例时在encode阶段触发断言，错误信息包含`Shape size mismatch`或`Data size mismatch`，并伴随`rawTensor->actualRawmagic`、`rawShape`、`actualrawShape`等字段输出，报错位置为`framework/src/machine/utils/dynamic/dev_encode.cpp`中`InitRawTensorAndMemoryRequirement`函数。

**问题背景**：encode阶段会对所有带有`actualRawmagic`的RawTensor进行内存复用一致性校验——即要求复用链两端（当前rawTensor与其`actualRawmagic`指向的actualRaw）在rawShapeSize（元素数乘积）或rawDataSize（字节大小）上严格一致。涉及`reshape`、`assemble`等内存复用操作时，若pass侧未同步更新相关tensor的特征信息，就会触发此类断言。

**定位步骤**：

1. **从报错第一现场获取基础信息**：

   触发断言时，错误日志中会包含以下关键字段（对应`dev_encode.cpp`第412~420行）：

   ```txt
   Shape size mismatch: <rawShapeSize> != <actualRawShapeSize>,
   rootMagic=<...>, rootHash=<...>,
   rawShape=<...>, actualrawShape=<...>,
   rawTensor->rawMagic=<...>, rawTensor->actualRawmagic=<...>, actualRaw->rawMagic=<...>
   ```

   记录其中的`rawMagic`、`actualRawmagic`、`rawShape`、`actualrawShape`，作为后续计算图定位的依据。

   若当前报错信息不够完整（缺少rawMagic等字段），可参考`dev_encode.cpp`中该ASSERT的上下文，在断言前增加相关字段打印，重编whl包后复现。

2. **打开pass计算图dump开关**：

   修改`framework/src/interface/configs/tile_fwk_config.json`，将`global.pass.default_pass_configs`下的`dump_graph`改为`true`：

   ```json
   "default_pass_configs": {
       "print_graph": false,
       "print_program": false,
       "dump_graph": true,
       ...
   }
   ```

   重新编译whl包并安装。

3. **复现用例，获取pass计算图**：

   再次执行用例，在`build/output/pass/`目录下会生成各pass阶段的计算图文件。

4. **定位目标pass范围**：

   建议重点排查**pass4 ~ pass27**之间的tilegraph：
   - pass4 ExpandFunction：之前为tensorgraph阶段，尚未进入pass的tile展开粒度分析
   - pass27 SubgraphToFunction:之后框架开始切分root function为leaf function，图结构较为分散，不便于整体定位

   在该范围内，根据步骤1获取的`rawMagic` / `actualRawmagic`，在计算图中定位对应tensor节点。

5. **结合计算图分析问题原因**：

   确定tensor节点后，沿数据流向分析其上下游操作，重点关注以下场景：

   - **reshape操作**：reshape会为输出tensor设置`actualRawmagic`，使其复用输入tensor的内存地址
   - **assemble操作**：assemble在图优化阶段会将输入tensor的raw指针替换为目标大tensor的raw，若reshape在此之前已设置`actualRawmagic`，该字段可能未随raw替换同步更新
   - **其他内存复用操作**：view、inplace等操作同样涉及`actualRawmagic`的设置与传递

   若复用链两端的rawShapeSize在某个pass之后出现不一致，说明该pass对其中一侧的rawshape进行了改写，但未同步另一侧。

6. **判断问题归属**：

   （1）**用例写法问题**：检查用例中reshape、assemble等操作的组合方式是否满足API约束（如`inplace=True`的限制、valid_shape与rawshape的一致性要求等）。若存在不满足约束的写法，调整用例。

   （2）**框架侧问题**：若用例写法无误，联系**pass同事**进行进一步分析，排查框架在处理actualRawmagic传递时是否存在遗漏或错误更新的场景。

注：

- actualRawmagic断言失败本质是编译期一致性校验，去掉断言后若运行精度仍正确，说明该路径的运行时读写并未越界，属于校验口径过严或pass侧信息同步遗漏问题，需与pass同事联合分析
- 此类问题若在动态shape（含负数维度）场景下触发，`dev_encode.cpp`会跳过动态维度的校验（`isDynamicShape`分支），排查时需注意区分静态与动态shape场景

---

### Workspace内存异常偏大

**问题特征**：运行用例时内存分配失败，报错信息为以下两类之一：

- torch申请失败：

```txt
torch.OutOfMemoryError: NPU out of memory. Tried to allocate 6.69 GiB (NPU 5; 60.96 GiB total capacity; ...)
```

- device内存申请失败：

```txt
rtMalloc failed. size:5254523547
```

**问题背景**：Workspace内存分配由以下阶段组成：

1. **图编译阶段**：在`dev_encode.cpp`的`EncodeDevAscendProgram`中估算各类内存池预算
2. **Launch阶段（用户指定）**：通过`dynWorkspaceSize`指定额外的动态内存（一般不会是此处问题）
3. **Launch阶段（实际分配）**：在`device_launcher.h`中通过`devMem.AllocDev(devProg->workspaceSize, ...)`一次性分配整个内存池
4. **Device运行时**：在`dev_workspace.h`中按预算对内存池进行suballocation

实际发生malloc的只有阶段3。内存池超大说明阶段1的预算估算出了问题。

Workspace的内存预算结构（定义于`dev_encode_program.h`）：

```cpp
struct {
    struct {
        uint64_t rootInnerSpilledMem;           // Root Function Inner Tensor内存
        uint64_t devTaskInnerExclusiveOutcasts; // DeviceTask内部Exclusive Outcast内存
        uint64_t maxStaticOutcastMem;          // 最大静态Outcast单体大小
        uint64_t maxDynamicAssembleOutcastMem; // 最大动态Assemble Outcast单体大小
        uint64_t devTaskBoundaryOutcastNum;    // Boundary Outcast slot数量（E×2+A×2）
        uint64_t devTaskInnerTemporalOutcastNum; // Inner Temporal Outcast slot数量（A×K）

        uint64_t MaxOutcastMem() const {
            return std::max(maxStaticOutcastMem, maxDynamicAssembleOutcastMem);
        }
        uint64_t Total() const {
            return rootInnerSpilledMem + devTaskInnerExclusiveOutcasts +
                   MaxOutcastMem() * (devTaskBoundaryOutcastNum + devTaskInnerTemporalOutcastNum);
        }
    } tensor;
    uint64_t aicoreSpilled;
    struct {
        uint64_t general;
        uint64_t stitchPool;
        uint64_t Total() const { return general + stitchPool; }
    } metadata;
    struct {
        uint64_t dumpTensor;
        uint64_t leafDump;
    } debug;
} memBudget;
```

**定位步骤**：

1. **开启日志与计算图dump**：

   （1）至少设置INFO日志级别：

   ```bash
   export ASCEND_GLOBAL_LOG_LEVEL=1
   export GLOBAL_LOG_LEVEL=1
   export ASCEND_PROCESS_LOG_PATH=<日志落盘路径>
   ```

   （2）打开计算图dump（便于后续定位问题tensor在图中的位置）：

   `framework/src/interface/configs/tile_fwk_config.json`

   ```json
   "dump_graph": true
   ```

   （3）重新编译whl包并安装，执行用例。

2. **查看workspace总体大小构成**：

   在日志路径`./debug`下搜索`[workspaceSize]`关键字，可以看到如下格式的日志：

   ```txt
   [workspaceSize] Metadata=12062240, workspaceSize=599916544, tensor=592543744, aicoreSpillen=7372800, debug.DumpTensor=0, leafDumpWorkspace=0.
   [workspaceSize] Tensor:rootInnerSpilledMem=182452224, devTaskInnerOutCasts=29360128, slotted=12582912x1380(slots).
   [workspaceSize] OutcastSlots: boundary=100, innerTemporal=1280.
   ```

   其中`workspaceSize`为总大小，`tensor`、`Metadata`、`aicoreSpillen`、`debug.DumpTensor`、`leafDumpWorkspace`为各子项。通过对比可快速确定哪个大类占用最多。

   > 当前workspace大小类问题主要集中在**Tensor Workspace**，metadata在总量中占比通常不高。若出现metadata类报错（包含`"Memory not enough"`且`WsProperty:metadata`、或包含`"Slab alloc null"`字样），建议直接联系machine同事分析。

3. **分析Tensor Workspace各项占比**：

   Tensor日志格式 `Tensor:rootInnerSpilledMem=A, devTaskInnerOutCasts=B, slotted=CxD(slots)` 中：

   - `A` = `memBudget.tensor.rootInnerSpilledMem`
   - `B` = `memBudget.tensor.devTaskInnerExclusiveOutcasts`
   - `C` = `memBudget.tensor.MaxOutcastMem()`（单个 slotted Outcast 最大大小）
   - `D` = `devTaskBoundaryOutcastNum + devTaskInnerTemporalOutcastNum`（Boundary + Inner Temporal 总 slot 数）
   - Slotted Outcast 总内存 = `C × D`

   分项日志 `OutcastSlots: boundary=X, innerTemporal=Y`：

   - `X` = `memBudget.tensor.devTaskBoundaryOutcastNum`（固定 boundary ping-pong，含 exclusive 与 assemble 的 `×2`）
   - `Y` = `memBudget.tensor.devTaskInnerTemporalOutcastNum`（assemble 随 stitch 深度缩放部分）

   结合每个Root Function级别的日志进一步缩小范围：

   ```txt
   [workspaceSize] MaxRootInnerMem is 182452224, maxDevTaskInnerExclusiveOutcastMem is 4194304.
   [workspaceSize] Rootfunction: TENSOR_LOOP_s2_Unroll8_PATH3_hiddenfunc0_root ->MaxRootInnerMem is 182452224, maxDevTaskInnerExclusiveOutcastMem is 4194304.
   ```

   `MaxRootInnerMem`和`maxDevTaskInnerExclusiveOutcastMem`是**经过unroll和stitch膨胀后**的值。若多个Root Function中某一个的值远超其他，即为问题来源。

   **情况一：rootInnerSpilledMem / devTaskInnerOutCasts均匀偏大**

   先确认是否为`stitch_function_max_num`或`unroll_list`配置过大导致。内存膨胀关系大致为（近似，非精确公式）：

   ```txt
   rootInnerSpilledMem ≈ per_root_budget / unroll × WorkspaceRecyclePeriod
   devTaskInnerOutCasts ≈ per_root_budget / unroll × EstimatedStitchingCount
   ```

   其中`WorkspaceRecyclePeriod ≈ stitch_function_max_num × MAX_UNROLL_TIMES`。可通过降低`stitch_function_max_num`或`unroll_list`在牺牲并行度的前提下降低内存。若单个loop的内存需求已经偏高（原始`rootInnerTensorWsMemoryRequirement`超过20MB），则需分析pass的内存复用策略及算子本身写法。

   **情况二：Boundary Outcast 内存偏大（slotted 项中单体大小 C 异常大）**

   由于 Tiling 会将 Tensor 切至中等大小（如不超过 512×512），超大的 `MaxOutcastMem()` 通常意味着未经 Tiling 的超大 Tensor 进入了子图。常见原因包括 Inplace 操作、Fixed Address Tensor、shmemData 等例外 case。

   若 `innerTemporal` 分项偏大，优先检查 `stitch_function_max_num` / `unroll_list` 配置；若 `boundary` 分项偏大，检查 exclusive/assemble outcast 类型数是否过多。

4. **定位问题Tensor**：

   日志中会对shape超过512×512的Tensor打印警告：

   ```txt
   [workspaceSize] Root=[TENSOR_LOOP_s2_Unroll8_PATH3_hiddenfunc0_root], symbol=[atten_out],rawmagic=[3066]: staticMemReq=[12582912] is too larger, which might indicate an error
   ```

   每条记录包含Root Function名称、Tensor符号名（匿名Tensor为空）、rawmagic以及静态内存大小。结合步骤3中确定的异常Root Function名称和异常内存值进行匹配。

   例如，若步骤3中Boundary Outcast单体大小为`12582912`，在警告日志中搜索该值即可定位到具体Tensor。

5. **结合计算图确认问题位置**：

   在步骤1打开`dump_graph`后，`pypto/output/pass/`目录下会生成各pass阶段的计算图。根据步骤4获取的Root Function名称和rawmagic，在计算图中搜索对应节点，确认该Tensor的上下游操作及shape来源,并能够跳转到对应代码段。

   定位到此即可联系相关组件同事介入分析。

6. **影响workspace大小的关键配置项**：

   | 配置项 | 影响范围 | 说明 |
   |--------|----------|------|
   | `stitch_function_max_num` | rootInnerSpilledMem、devTaskInnerOutCasts、innerTemporal slot 数 | 控制stitch并行数，直接影响 WorkspaceRecyclePeriod 和 assemble temporal slot 数 |
   | `unroll_list` / max_unroll | rootInnerSpilledMem、devTaskInnerOutCasts | 控制loop展开次数，影响CalcUnrolledRootBudget |

注：

- 动态shape场景下`maxStaticMemReq`为0（无法从符号shape推算静态大小），此类Tensor不会出现在超大Tensor的警告中
- `aicoreSpilled`为AICore栈溢出到workspace的内存，若该项异常偏大，需检查算子的`stackWorkSpaceSize`
- `debug.DumpTensor`和`leafDumpWorkspace`为调试模式下的额外内存开销，正常模式下为0
**关联Skill**：[pypto-environment-setup](../../../.agents/skills/pypto-machine-workspace/SKILL.md)

---

### F70006 HANDSHAKE_TIMEOUT

1. **确认设备与驱动**：NPU设备可用、驱动正常，`npu-smi info`无异常。
2. **确认资源与负载**：当前进程/容器内NPU占用是否过高，是否存在多进程争用同一设备。
3. **确认超时配置**：若存在握手/同步超时配置项，检查是否过短或与环境不符。
4. **查日志上下文**：结合同线程前后日志（如“Schedule run init succ”之后、AbnormalStop相关）确认是首次握手失败还是运行中异常。

**关联Skill**：[pypto-environment-setup](../../../.agents/skills/pypto-environment-setup/SKILL.md)（环境与NPU设备诊断、`npu-smi`、驱动与编译运行）

---

### Host侧捕获异常打印汇编堆栈信息

**问题特征**：执行用例在host打屏输出堆栈信息
例如：

```txt
floating point exception !!!
libtile_fwk_interface.so(npu::tile_fwk::Pad(long, long)+0xe) [0X77188ae025ae]
```

**定位步骤**：

1. **编译带有Debug信息的pypto包并安装**：

```bash
python3 build_ci.py -f=python3 --build_type=Debug
pip install build_out/pypto*whl --force-reinstall --no-dep
```

2. **重新执行问题用例**

3. **查找二进制文件位置**：
如果不清楚包安装在哪里可以使用find全局搜索

```bash
find / -name "libtile_fwk_interface.so"
```

4. **反汇编得到具体代码行**：
例如：

```bash
objdump -d -C -l /path/to/libtile_fwk_interface.so | grep -A 20 "npu::tile_fwk::Pad(long, long)>"
```

可得到触发问题的函数具体行号。

5.**关联skill**：[pypto-host-stacktrace-analyzer](../../../.agents/skills/pypto-host-stacktrace-analyzer/SKILL.md)

---

### 整网环境中出现AICPU执行超时
PYPTO可以通过PYPTO_LAUNCH_SCHED_SAME_CLUSTER环境变量设置是否强制在同一Cluster分配调度线程。在同一Cluster分配调度线程能够获得更好的性能，但是在整网场景中，除了Pypto，还有其他组件使用aicpu，强制在同一个Cluster分配线程可能因为aicpu资源不够导致aicpu执行超时，这种情况下，需要将该环境变量设置为false，并通过launch_sched_aicpu_num配置可用的aicpu数量。
需要特别说明的是，当开启同Cluster分配时, launch_sched_aicpu_num配置会不生效。

### AiCore Print使用方法

**功能说明**：AiCore Print用于在AICore kernel中打印tensor数据和调试信息，支持GM、UB、L1内存层次和多种数据类型。

#### 对外接口

| 接口名称 | 功能 | 适用场景 | Ascend 950PR |
|---------|------|---------|:---:|
| **AiCoreLogF** | 格式化日志打印 | 打印地址、标量、提示信息 | 支持 |
| **AiCorePrintShape** | 打印 Shape 信息 | 查看 tensor shape 维度 | 支持 |
| **AiCorePrintGmTensor** | 打印 GM Tensor | 查看 Global Memory 数据 | 支持 |
| **AiCorePrintUbTensor** | 打印 UB Tensor | 查看 Unified Buffer 数据（仅 AIV kernel） | 支持 |
| **AiCorePrintL1Tensor** | 打印 L1 Tensor | 查看 Circular Buffer 数据（仅 AIC kernel） | 不支持 |
| **AiCorePrintL0CTensor** | 打印 L0C Tensor | 查看 Accumulator Buffer 数据（仅 AIC kernel） | 支持 |

#### 支持的数据类型

AiCore Print支持以下数据类型：

**浮点类型**：

- Ascend 950PR：支持
- **fp32**：`float`
- **fp16**：`half`
- **bf16**：`bfloat16_t`

**整数类型**：

- Ascend 950PR：支持
- **int8**：`int8_t`
- **uint8**：`uint8_t`
- **int16**：`int16_t`
- **uint16**：`uint16_t`
- **int32**：`int32_t`
- **uint32**：`uint32_t`
- **int64**：`int64_t`
- **uint64**：`uint64_t`

**FP8类型**（平台限制）：

- Ascend 950PR：支持
- **fp8_e4m3**：`float8_e4m3_t`
- **fp8_e5m2**：`float8_e5m2_t`
- **fp8_e8m0**：`float8_e8m0_t`
- **hifloat8**：`hifloat8_t`

**平台限制说明**：FP8 和 HiFloat8 类型仅在 Ascend 950PR 上支持（`SUPPORT_FP8_HF8_PRINT=1`，对应 `__NPU_ARCH__ == 3510`）。

其他平台不支持FP8/HiFloat8打印功能。

#### 使用步骤

##### 1. 启用追踪日志

修改配置文件：

`framework/src/interface/configs/tile_fwk_config.json`

```json
"fixed_output_path": true,
"force_overwrite": false,
```

修改头文件：

`framework/src/interface/machine/device/tilefwk/aicore_print.h`

```cpp
#define ENABLE_AICORE_PRINT 1
```

##### 2. 重新编译安装

```bash
rm -rf build_out/ && python build_ci.py && pip install build_out/pypto*whl --force-reinstall --no-deps
```

##### 3. 在kernel CCE文件中添加打印代码

**重要流程说明**：

**何时删除kernel_aic*目录**：

- 首次运行或切换用例：删除kernel_aic*目录
- 同一用例重复运行：保留kernel_aic*目录（保留修改）

**步骤3.1：首次运行生成kernel CCE文件**

首次运行或切换用例：

```bash
rm -rf kernel_aic* output/ wk/
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && python xxx.py
```

同一用例重复运行（已添加打印代码）：

```bash
rm -rf output/ wk/
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && python xxx.py
```

**步骤3.2：在生成的CCE文件中添加打印代码**

查看生成的kernel文件：

```bash
ls kernel_aicore/*.cpp
```

修改步骤：
（1）在文件开头添加：`#include "tilefwk/aicore_print.h"`
（2）在合适位置（数据加载或计算后的同步点）添加打印调用

打印接口调用格式：

```cpp
AiCoreLogF(param->ctx, "format string", args...);
AiCorePrintShape(param->ctx, Shape2Dim(dim0, dim1), "name");
AiCorePrintGmTensor(param->ctx, (__gm__ T*)addr, end, begin, "name");
AiCorePrintUbTensor(param->ctx, (__ubuf__ T*)addr, end, begin, "name");
AiCorePrintL1Tensor(param->ctx, (__cbuf__ T*)addr, end, begin, l1_staging, "name");
AiCorePrintL0CTensor(param->ctx, (__cc__ T*)addr, end, begin, l0cShape0, l0cShape1, l0c_staging, "name");
```

**步骤3.3：配置L1/L0C staging buffer（仅AiCorePrintL1Tensor / AiCorePrintL0CTensor需要）**

```cpp
// L1 staging buffer（从workspace分配）
__gm__ T* l1_staging = (__gm__ T*)(param->funcData->workspaceAddr);

// L0C staging buffer（从workspace分配，需32字节对齐）
__gm__ T* l0c_staging = (__gm__ T*)(param->funcData->workspaceAddr);
```

**关键注意事项**：首次运行或切换用例删除kernel_aic*，同一用例重复运行保留修改。

##### 4. 运行测试并查看打印结果

**重要**：以下命令必须**一次性完整执行**（使用`&&`连接），不要拆分为多个命令：

```bash
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && rm -rf output/ wk/ && python xxx.py && grep -rn "DumpAicoreLog" ./wk
```

**命令说明**：

1. `export ASCEND_PROCESS_LOG_PATH=./wk`：设置日志输出目录为`./wk`
2. `export ASCEND_GLOBAL_LOG_LEVEL=1`：设置日志级别为INFO（级别1），开启日志输出
3. `rm -rf output/ wk/`：清理旧日志和编译产物，避免干扰
4. `python xxx.py`：运行测试用例，触发kernel编译和执行
5. `grep -rn "DumpAicoreLog" ./wk`：搜索并打印所有AiCore Print输出（包含tensor数据和调试信息）

#### 不同数据类型打印示例

以下示例展示每种数据类型的打印用法。打印代码需在合适位置插入（如TLoad/TAdd后的同步点）。

##### 浮点类型

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ float*)gmTensor_fp32.GetAddr(), 8, 0, "fp32_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ half*)ubTensor_fp16.GetAddr(), 16, 0, "fp16_ub");

__gm__ bfloat16_t* l1_staging_bf16 = (__gm__ bfloat16_t*)(param->funcData->workspaceAddr);
AiCorePrintL1Tensor(param->ctx, (__cbuf__ bfloat16_t*)l1Tensor_bf16.GetAddr(), 16, 0, l1_staging_bf16, "bf16_l1");
```

##### 整数类型

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ int8_t*)gmTensor_int8.GetAddr(), 16, 0, "int8_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ uint8_t*)ubTensor_uint8.GetAddr(), 16, 0, "uint8_ub");

AiCorePrintUbTensor(param->ctx, (__ubuf__ int16_t*)ubTensor_int16.GetAddr(), 8, 0, "int16_ub");

AiCorePrintGmTensor(param->ctx, (__gm__ uint16_t*)gmTensor_uint16.GetAddr(), 8, 0, "uint16_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ int32_t*)ubTensor_int32.GetAddr(), 16, 0, "int32_ub");

AiCorePrintGmTensor(param->ctx, (__gm__ uint32_t*)gmTensor_uint32.GetAddr(), 8, 0, "uint32_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ int64_t*)gmTensor_int64.GetAddr(), 8, 0, "int64_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ uint64_t*)ubTensor_uint64.GetAddr(), 8, 0, "uint64_ub");
```

##### FP8类型（平台限制）

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ float8_e4m3_t*)gmTensor_fp8e4m3.GetAddr(), 8, 0, "fp8e4m3_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ float8_e5m2_t*)gmTensor_fp8e5m2.GetAddr(), 8, 0, "fp8e5m2_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ float8_e8m0_t*)gmTensor_fp8e8m0.GetAddr(), 8, 0, "fp8e8m0_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ hifloat8_t*)gmTensor_hf8.GetAddr(), 8, 0, "hifloat8_gm");
```

##### 其他接口

AiCorePrintShape：

```cpp
AiCorePrintShape(param->ctx, Shape2Dim(sym_161_dim_0, sym_161_dim_1), "sym_161");
AiCorePrintShape(param->ctx, Shape3Dim(dim0, dim1, dim2));
AiCorePrintShape(param->ctx, Shape4Dim(dim0, dim1, dim2, dim3), "conv_out");
```

L1 Tensor打印示例：

```cpp
__gm__ half* l1_staging = (__gm__ half*)(param->funcData->workspaceAddr);
AiCorePrintL1Tensor(param->ctx, (__cbuf__ half*)l1Tensor.GetAddr(), 16, 0, l1_staging, "fp16_l1");
```

L0C Tensor打印示例（L0C数据通过DMA搬运到GM staging buffer后打印）：

```cpp
__gm__ int32_t* l0c_staging = (__gm__ int32_t*)(param->funcData->workspaceAddr);
AiCorePrintL0CTensor(param->ctx, (__cc__ int32_t*)l0cTensor.GetAddr(), 1024, 0, 32, 32, l0c_staging, "int32_l0c");
```

AiCoreLogF：

```cpp
AiCoreLogF(param->ctx, "GM address=%p", ((__gm__ float*)gmTensor.GetAddr()));
AiCoreLogF(param->ctx, "Shape=[%ld,%ld]", dim0, dim1);
AiCoreLogF(param->ctx, "INT8 input loaded");
```

#### 注意事项

1. **L1/L0C staging buffer对齐**：l1_staging和l0c_staging地址必须32字节对齐，workspaceAddr默认满足要求。

2. **打印数量控制**：PRINT_BUFFER_SIZE当前为128KB（定义于`framework/src/interface/machine/device/tilefwk/aicpu_common.h`），若触发overflow warning，需增大该值后重新编译。

3. **FP8/HiFloat8 支持平台**：仅 Ascend 950PR（`__NPU_ARCH__ == 3510`）支持（见 `SUPPORT_FP8_HF8_PRINT` 宏定义）。

4. **AiCorePrintL1Tensor 支持平台**：Ascend 950PR 不支持；Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品支持（见 `SUPPORT_L1_COPY` 宏定义）。

5. **AIC (Cube核)中不能使用AiCorePrintUbTensor**：AIC (Cube核)的标量处理器(SP)没有到UB地址空间的物理通路，无法从UB标量读取数据。编译期已通过`static_assert`拦截，在AIC kernel中调用`AiCorePrintUbTensor`会触发编译报错：

   ```text
   error: static assertion failed: [AIC UB Print Error] AiCorePrintUbTensor is not supported on AIC (Cube) kernel.
   ```

   UB数据检查请在AIV (Vector核) kernel中完成，或在AIC中使用`AiCorePrintGmTensor`打印已搬到GM的数据。

6. **AiCoreLogF在AIC中打印UB数据值会触发运行时错误**：`AiCoreLogF`在AIC kernel中使用`%f`、`%d`等格式打印UB数据值时（如`((__ubuf__ float*)addr)[521]`），编译器会生成一条从UB地址空间的标量load指令，AIC SP不支持此操作，触发MPU error 271：

   ```text
   error from aicore error exception, core id is 0, error code = 271
   errorStr: The MPU address access is invalid
   ```

   `%p`打印地址值（不读取UB数据）是安全的。**正确做法**：AIC kernel中不直接读取UB数据值，将UB打印逻辑移到AIV kernel中。

7. **不可通过DMA将UB数据搬到GM再打印**：AIC (Cube核)上没有MTE3 DMA引擎（`copy_ubuf_to_gm`、`copy_ubuf_to_gm_align_v2`等intrinsic不支持cube target），`TStoreVec`（`OP_UB_COPY_OUT`）的`OpCoreType`为`AIV`，属于Vector核专用。在AIC kernel中调用这些接口会编译报错：

   ```text
   error: function type '...' of 'copy_ubuf_to_gm' does not support the given target feature
   ```

#### 常见问题

##### 1. 未看到打印输出

检查：ENABLE_AICORE_PRINT=1、已重新编译安装，已指定日志落盘路径，已设置日志级别为info级别（1），grep搜索文件正确。

##### 2. L1/L0C Print对齐WARNING

确保l1_staging / l0c_staging地址32B对齐，workspaceAddr本身已对齐。

##### 3. Overflow Warning

减少打印数量或增大PRINT_BUFFER_SIZE后重新编译。

##### 4. FP8/HiFloat8无法打印

当前平台不支持（检查 `SUPPORT_FP8_HF8_PRINT` 宏；仅 Ascend 950PR / `__NPU_ARCH__ == 3510` 时为 1）。

##### 5. AiCorePrintL1Tensor找不到接口定义

当前平台不支持（检查SUPPORT_L1_COPY宏）。

##### 6. ld.lld: error: undefined symbol

编译时出现`ld.lld: error: undefined symbol`链接错误，导致编译失败。

**原因**：`parallel_compile` 配置值大于 1 时，CodeGen 会并行编译多个子图；在此模式下，部分编译单元之间的符号依赖未正确处理，导致链接失败。

**解决方案**：修改 `framework/src/interface/configs/tile_fwk_config.json`，将 `parallel_compile` 设为 `1`（编译线程数为 1，即串行编译）。注意：该配置项表示**并行编译线程数**，而非布尔开关（`1` 表示单线程，`128` 等为多线程并行）。修改后重新运行即可解决。

```json
"parallel_compile": 1
```

##### 7. AIC kernel中调用AiCorePrintUbTensor编译报错

AIC (Cube核) kernel中使用`AiCorePrintUbTensor`时，编译器会触发`static_assert`：

```text
error: static assertion failed due to requirement '!std::is_same_v<float, float>':
  [AIC UB Print Error] AiCorePrintUbTensor is not supported on AIC (Cube) kernel.
  AIC Scalar Processor cannot scalar-load from UB address space.
  Please use AiCorePrintUbTensor in AIV (Vector) kernel instead,
  or use AiCorePrintGmTensor to print data that has been moved to GM.
```

**原因**：AIC (Cube核)的标量处理器(SP)没有到UB地址空间的物理通路，无法从UB标量读取数据。

**解决方案**：将`AiCorePrintUbTensor`调用移到AIV (Vector核) kernel中，或使用`AiCorePrintGmTensor`打印已搬到GM的数据。

##### 8. AIC kernel中使用AiCoreLogF打印UB数据值触发error 271

在AIC kernel的CCE文件中使用如下代码：

```cpp
AiCoreLogF(param->ctx, "ubTensor val=%f", ((__ubuf__ float*)ubTensor.GetAddr())[521]);
```

运行时触发aicore error：

```text
error from aicore error exception, core id is 0, error code = 271
errorStr: The MPU address access is invalid
```

**原因**：`((__ubuf__ float*)addr)[521]`会生成一条从UB地址空间的标量load指令，AIC SP不支持此操作。**注意**：AIC kernel中从UB地址空间标量读取无法在编译期被`static_assert`拦截（因为`AiCoreLogF`的变参模板在参数表达式求值后，`__ubuf__`属性已丢失），也无法在运行时拦截（MPU error是硬件trap，无软件恢复机制）。

**解决方案**：

- 将UB数据打印逻辑移到AIV kernel中
- AIC kernel中仅使用`%p`打印UB地址值（不读取数据），这是安全的
- 检查AIC kernel的CCE代码，删除所有对UB地址空间做`[]`下标访问的表达式

##### 9. AIC kernel中尝试TStoreVec / copy_ubuf_to_gm搬运UB数据编译报错

在AIC kernel中调用`TStoreVec`、`copy_ubuf_to_gm`、`copy_ubuf_to_gm_align_v2`等接口编译报错：

```text
error: function type 'void (__gm__ void *, __ubuf__ void *, ...)' of 'copy_ubuf_to_gm' does not support the given target feature
```

**原因**：Cube核上没有MTE3 DMA输出引擎，所有从UB源地址搬迁数据的intrinsic均不支持cube target。`TStoreVec`（`OP_UB_COPY_OUT`）的`OpCoreType`为`AIV`，是Vector核专用操作。

**解决方案**：此类操作只能在AIV (Vector核) kernel中使用，不要在AIC kernel中调用。需要打印UB数据时，在AIV中完成。

---

### 泳道图相关问题指导

<a id="output-目录产物说明"></a>

#### output目录产物说明

在`output/output_时间戳`目录下，泳道图相关文件通常包括：

- `merged_swimlane.json`：IDE展示用的综合泳道图文件。
- `machine_runtime_operator_trace*.json`：AI CPU/AI Core泳道图展示文件，可用于观察联合时序。
- `machine_trace_perf_data*.json`：Machine组件原始Profiling数据文件。
- `tilefwk_L1_prof_data_*.json`：Machine组件原始Profiling数据文件。
其中，`machine_trace_perf_data*.json`与`tilefwk_L1_prof_data_*.json`可用于判断底层数据采集是否成功（例如文件是否为空）；`merged_swimlane.json`与`machine_runtime_operator_trace*.json`主要用于IDE可视化展示。建议优先联系IDE对应负责人咨询解决。

<a id="IDE-参数含义解释"></a>

#### IDE参数含义解释

在生成和查看泳道图时，IDE工具中会显示多个性能参数和事件标签。以下是常见参数的含义说明：

**1. CTRL AICPU**

| 阶段 | 含义 | 打点位置 |
| --- | --- | --- |
| **DEV_TASK_BUILD** | Ctrl AICPU构建devTask的耗时，即stitch耗时统计 | 在stitch之后 |
| **Post-process** | Ctrl AICPU在构建完所有DevTask到退出的时间 | AICPU退出时 |
| **Total run time** | Ctrl AICPU从被拉起到退出时的总时间 | 整个流程启动到退出 |

**2. SCHED AICPU**

| 阶段 | 含义 | 打点位置 |
| --- | --- | --- |
| **ALLOC_THREAD_ID** | 线程分配，绑核耗时统计 | AllocThreadIdx之后 |
| **INIT** | Sched AICPU初始化耗时统计 | Sched arg参数初始化后（Sched init()之后） |
| **CORE_HAND_SHAKE** | Sched AICPU与AICore握手的耗时统计 | 握手之后 |
| **DEV_TASK_RCV** | Sched AICPU接收到Ctrl AICPU构建的devTask的耗时 | 从taskQue读取DevTask之后 |
| **Post-process** | Sched AICPU在执行完所有DevTask到退出的时间 | ExecuteTask之后到AICPU退出时 |
| **Total run time** | Sched AICPU从被拉起到退出时的总时间 | 整个流程启动到退出 |

**3. AICORE**

| 阶段 | 含义 | 打点位置 |
| --- | --- | --- |
| **End-to-End time** | AICore端到端实际执行时间 | 从最早开始执行ExecCoreFunctionKernel的AICore到最晚结束执行的AICore统计时间 |
| **Total run time** | AICore从被拉起到退出时的总时间 | 整个流程启动到退出 |

#### 常见异常排查

##### 1. 未生成泳道图文件

**现象**：算子运行正常，但`output`目录下未生成`tilefwk_L1_prof_data_*.json`文件。
**原因与排查**：通常是因为未启动性能数据采集功能。请检查代码中是否已正确将`runtime_debug_mode`设置为`1`。

##### 2. 泳道图文件为空（无任何数据）

**现象**：成功生成了`tilefwk_L1_prof_data_*.json`文件，但文件内容为空。
**原因与排查**：通常是Profiling功能未能成功使能。需要开启DEBUG日志打印进行进一步排查：

   - 按照上文说明打开DEBUG日志并指定日志落盘路径。
   - **Device侧排查**：检查日志文件`log/debug/device*/device*.log`。若包含`aicore profiling is opened, level is %d.`，表示成功使能；若包含`aicore profiling is closed..`，则表示未能成功使能，aicore没有开启泳道图性能数据采集。

##### 3. 泳道图中某些核首任务启动时间过长

**现象**：从泳道图看，部分核并没有前序任务依赖，但第一个任务的启动时间却很长。
**原因与排查**：这种情况通常是因为AICPU启动较慢，导致AICore接收任务的时间被整体延后。在泳道图中表现为首任务启动前存在等待AICPU启动的时间。

##### 4. ACL Graph模式下采集不到泳道图数据

**现象**：当算子运行在ACL Graph模式时，启动泳道图性能采集后，`output`目录下没有生成泳道图文件。
**原因与排查**：当前PyPTO尚未支持ACL Graph场景的泳道图性能数据采集。在ACL Graph模式中，执行流程分为Capture和Replay两个阶段，当前Capture阶段未开启Profiling，而是在Replay阶段开启性能采集；但Task的下发实际发生在Capture阶段，由于此时Profiling开关是关闭的，所以不会上报OP相关信息。目前请暂时规避该场景，后续版本将支持ACL Graph模式下的泳道图性能数据采集。

##### 5. Profiling泳道图数据与msprof采集的结果差距较大

**现象**：`msprof`采集到的AICore耗时远大于泳道图中的AICore端到端耗时，二者数据无法对齐。
**原因与排查**：`msprof`所采集到的AICore耗时不能真实代表AICore内部端到端的执行耗时，因为它实际上还包含了**AICore启动等待AICPU下发devTask的时间**，以及**AICore执行完任务后的退出时间**。为了获取更准确的时间，当前已实现对AICore端到端执行时间的打屏输出，可以在执行算子前设置环境变量`export DUMP_DEVICE_PERF=true`，即可在终端中直接获取当前准确的性能统计数据。
