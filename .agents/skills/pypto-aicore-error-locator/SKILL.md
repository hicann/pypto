---
name: pypto-aicore-error-locator
description: 定位测试案例中出现 aicore error 时的问题 CCE 文件和问题代码行。当用户说"aicore error"、"定位 aicore error 的原因"、"帮我定位 aicore error 报错"时使用此技能。
license: 完整条款见 LICENSE.txt
---

# AICore Error 定位器

此技能用于定位 PyPTO 测试中出现的 aicore error，通过系统化的排查流程，找出导致错误的 CCE 文件和具体代码行。

## 工作流程概述

1. 初始化默认信息
2. 排除 machine 框架调度问题
3. 启用追踪日志
4. 重新编译和安装
5. 清理日志并运行测试
6. 分析追踪日志并定位 CCE 文件
7. 二分查找定位问题代码行
8. 映射到前端源代码
9. 输出结果

---

## 步骤 1：初始化默认信息

使用以下默认值直接初始化：

- **pypto_path**: 当前工作目录（即 pypto 项目根目录）
- **device_log_path**: `{pypto_path}/device_log`
- **run_path**: `{pypto_path}`
- **test_cmd**: 从用户输入中提取（用户触发此技能时提供的测试命令或问题描述中包含的运行命令）

---
**⚠️ 重要提示**: 将 bash 运行命令超时时间设置为 1800000ms
## 步骤 2：排除 machine 框架调度问题

### 2.1 注释 CallSubFuncTask

使用脚本注释 `aicore_entry.h` 中的 CallSubFuncTask 部分：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/modify_callsubfunctask.py comment pypto_path
```

**脚本参数说明**:
- `comment`: 注释 CallSubFuncTask
- `pypto_path`: pypto 项目的根目录路径（绝对路径或相对路径）

### 2.2 编译安装

进入 `pypto_path`，重新编译 pypto 包并 pip 安装。

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

### 2.3 运行验证（在注释 CallSubFuncTask 的状态下）

进入 `run_path`，运行测试。

```bash
cd run_path && test_cmd
cd -
```

**⚠️ 重要提示**：
- **若没有 aicore error**: 说明问题在 kernel 代码中，而非 machine 调度框架，**请继续执行后续步骤！**
- **若有 aicore error**: 说明是 machine 调度框架的问题（CallSubFuncTask 相关），已找到问题原因，**停止执行后续步骤！**

### 2.4 取消注释 CallSubFuncTask

使用脚本取消注释 `aicore_entry.h` 中的 CallSubFuncTask 部分：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/modify_callsubfunctask.py uncomment pypto_path
```

**脚本参数说明**:
- `uncomment`: 取消注释 CallSubFuncTask
- `pypto_path`: pypto 项目的根目录路径（绝对路径或相对路径）



---

## 步骤 3：启用追踪日志

进入 `pypto_path`，修改以下配置：

- **配置文件**: 修改 `tile_fwk_config.json`
  - 设置 `"fixed_output_path"` 为 `true`
  - 设置 `"force_overwrite"` 为 `false`

- **头文件**: 修改 `aicore_print.h`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

- **工具头文件**: 修改 `device_switch.h`
  - 设置 `#define ENABLE_COMPILE_VERBOSE_LOG` 为 `1`

---

## 步骤 4：重新编译和安装

进入 `pypto_path`，重新编译 pypto 包并 pip 安装。

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

---

## 步骤 5：清理日志并运行测试

### 5.1 运行测试

进入 `run_path`，配置环境变量并运行测试。

```bash
rm -rf device_log_path/* && rm -rf run_path/kernel_aic* && cd run_path && export ASCEND_PROCESS_LOG_PATH=device_log_path && export ASCEND_GLOBAL_LOG_LEVEL=0 && test_cmd
cd -
```

**⚠️ 重要提示**:
- 运行测试的打屏日志中必须出现 aicore error，
- **如果未出现 aicore error，则不适用于该 SKILL，立即停止执行后续步骤！！！**


### 5.2 获取 program.json 路径

运行脚本获取最新的 program.json 路径：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/get_latest_program_json.py run_path/output
```

记录输出的 `program_json_path`，该路径将在步骤 8 中使用。

---

## 步骤 6：分析追踪日志并定位 CCE 文件

### 6.1 查找 trace 日志、分析缺失 leaf index 并定位问题 CCE 文件

```bash
python3 scripts/analyze_trace.py device_log_path run_path/kernel_aicore
```

**⚠️ 重要提示**: 若未定位到问题 CCE 文件，请说明原因，**停止执行后续步骤**

### 6.2 测试验证 CCE 文件

如果有多个问题 CCE 文件，需要分别测试每个文件，以确定哪个是问题文件。若只有一个问题 CCE 文件，测试验证该文件是否为问题文件：

```bash
python3 scripts/test_cce_file.py <cce_file> test_cmd run_path
```

**⚠️ 重要提示**:
- 若未定位到问题 CCE 文件，请说明原因，**停止执行后续步骤**
- 若打印的 error 中包含 `ld.lld: error: undefined` 关键字，则修改 `tile_fwk_config.json` 中的 `parallel_compile` 为 `1`，再从步骤 1 开始重新执行一遍

---

## 步骤 7：二分查找定位问题代码行

### 7.1 获取 ERROR_IN_T 的值（错误是否在 T 操作中）

```bash
python3 scripts/determine_error_scope.py <cce_file> test_cmd run_path
```

**⚠️ 重要提示**:
- `cce_file` 为步骤 6.2 的输出
- 若打印的 error 中包含 `ld.lld: error: undefined` 关键字，则修改 `tile_fwk_config.json` 中的 `parallel_compile` 为 `1`，再从步骤 1 开始重新执行一遍

### 7.2 获取二分查找初始范围

```bash
python3 scripts/get_commentable_range.py <cce_file> ERROR_IN_T
```

记录输出的 `LEFT` 和 `RIGHT` 值。

### 7.3 执行二分查找迭代

根据上一步的 `LEFT` 和 `RIGHT` 值，执行第一次迭代：

```bash
python3 scripts/binary_search_iteration.py <cce_file> test_cmd run_path <left> <right> ERROR_IN_T
```

记录输出的 `NEXT_LEFT` 和 `NEXT_RIGHT` 值。

**判断逻辑**:
- 如果 `NEXT_LEFT` 等于 `NEXT_RIGHT`，则已找到问题行（输出 `FOUND <problem_line>`）
- 否则，使用新的 `NEXT_LEFT` 和 `NEXT_RIGHT` 作为下一轮的 `left` 和 `right`，重复执行此步骤

**⚠️ 重要提示**:
- `cce_file` 为步骤 6.2 的输出
- 若未定位到问题代码行，请说明原因，**停止执行后续步骤**

---

## 步骤 8：映射到前端源代码

使用以下命令将 CCE 问题代码行映射到前端源代码：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/locate_source_line.py <cce_file> <program_json_path> <problem_line>
```

**参数说明**:
- `<cce_file>`: 步骤 6.2 输出的问题 CCE 文件路径
- `<program_json_path>`: 步骤 5.2 输出的 program.json 文件路径
- `<problem_line>`: 步骤 7 输出的问题代码行号

**输出说明**:
- 若匹配成功，将输出前端源代码文件路径和行号
- 若匹配失败，将说明原因（例如：框架自动生成代码、操作数不匹配等）

---

## 步骤 9：输出结果

输出以下信息：
- 找到的 CCE 文件路径
- 问题代码行号
- 问题代码内容
- 前端源代码文件路径（如果步骤 8 映射成功）
- 前端源代码行号（如果步骤 8 映射成功）
- 前端源代码内容（如果步骤 8 映射成功）

---

## 关键注意事项

1. **fixed 模式**: 确保 `fixed` 模式启用以保持输出路径不变
2. **路径规范**: 所有路径必须使用绝对路径
3. **默认路径**: 步骤 1 使用默认值初始化，无需收集用户输入
4. **停止条件**: 遇到不适用的情况或定位失败时，立即停止执行并说明原因
5. **并行编译问题**: 遇到 `ld.lld: error: undefined` 错误时，需要修改 `parallel_compile` 为 `1` 并从头重新执行
