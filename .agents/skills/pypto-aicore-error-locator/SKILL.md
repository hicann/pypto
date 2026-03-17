---
name: pypto-aicore-error-locator
description: 定位测试案例中出现 aicore error 时的问题 CCE 文件和问题代码行。当需要分析 aicore 错误并找到导致错误的 CCE 文件及具体代码行时使用此技能。
license: 完整条款见 LICENSE.txt
---

# AICore Error 定位器

此技能工作流程：

## 1. 收集必要信息（必须执行）

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息，严禁猜测或使用默认值。**

使用 `question` 工具收集以下信息：

- **pypto_path**: pypto 项目的根目录路径
- **device_log_path**: device log 的落盘路径（若不存在则需创建）
- **test_cmd**: 触发 aicore error 的测试命令
- **run_path**: 运行测试命令的目录路径

将收集的路径全部转换成绝对路径，收集到所有信息后才能继续后续步骤。

## 2. 启用追踪日志

根据 pypto_path，修改以下配置：

- **配置文件**: 修改 `tile_fwk_config.json`
  - 设置 `"fixed_output_path"` 为 `true`
  - 设置 `"force_overwrite"` 为 `false`

- **头文件**: 修改 `aicore_entry.h`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

- **工具头文件**: 修改 `device_switch.h`
  - 设置 `#define ENABLE_COMPILE_VERBOSE_LOG` 为 `1`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

## 3. 重新编译和安装

进入 pypto_path，重新编译 pypto 包并 pip 安装。

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

## 4. 清理日志运行测试

进入 run_path，配置环境变量并运行测试。

```bash
rm -rf device_log_path/* && rm -rf run_path/kernel_aic* && cd run_path && export ASCEND_PROCESS_LOG_PATH=device_log_path && export ASCEND_GLOBAL_LOG_LEVEL=0 && test_cmd
cd -
```
**重要**: 运行测试的打屏日志中必须出现aicore error，如果未出现，则不适用于该SKILL，请停止运行后续步骤

## 5. 分析追踪日志并定位 CCE 文件

### 5.1 查找 trace 日志、分析缺失 leaf index 并定位问题 CCE 文件

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/analyze_trace.py device_log_path run_path/kernel_aicore
```
**重要**: 若未定位到问题CCE文件，请说明原因，停止运行后续步骤

### 5.2 测试每个 CCE 文件（多个文件时执行）

如果有多个问题 CCE 文件，需要分别测试每个文件，若只有一个问题 CCE 文件，不需要执行此步骤：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/test_cce_file.py <cce_file> test_cmd run_path
```

**重要**: 若未定位到问题CCE文件，请说明原因，停止运行后续步骤
**重要**: 若打印的error中包含 `ld.lld: error: undefined`关键字，则修改tile_fwk_config.json中的parallel_compile为1，再从第一步开始重新执行一遍

## 6. 二分查找定位问题代码行

### 6.1 检查原始文件是否有 error

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/check_original_error.py <cce_file> test_cmd run_path
```
**重要**: cce_file为上一步的输出，若原始文件无 error，则不适用于该SKILL，停止运行后续步骤

### 6.2 注释所有行后是否有 error

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/check_all_commented_error.py <cce_file> test_cmd run_path
```
**重要**: cce_file为上一步的输出，若注释所有行后存在 error，则不适用于该SKILL，停止运行后续步骤
**重要**: 若打印的error中包含 `ld.lld: error: undefined`关键字，则修改tile_fwk_config.json中的parallel_compile为1，再从第一步开始重新执行一遍

### 6.3 获取二分查找初始范围

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/get_commentable_range.py <cce_file>
```

记录输出的 `LEFT` 和 `RIGHT` 值。

### 6.4 执行二分查找迭代

根据上一步的 LEFT 和 RIGHT 值，执行第一次迭代：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/binary_search_iteration.py <cce_file> test_cmd run_path <left> <right>
```

记录输出的 `NEXT_LEFT` 和 `NEXT_RIGHT` 值。

如果 `NEXT_LEFT` 等于 `NEXT_RIGHT`，则已找到问题行（输出 `FOUND <problem_line>`）。

否则，使用新的 `NEXT_LEFT` 和 `NEXT_RIGHT` 作为下一轮的 left 和 right，重复执行此步骤。

**重要**: cce_file为上一步的输出，若未定位到问题代码行，请说明原因，停止运行后续步骤

### 7. 输出结果
 	
输出找到的 CCE 文件路径、问题代码行和问题代码。

## 关键点

- 确保 `fixed` 模式启用以保持输出路径不变
