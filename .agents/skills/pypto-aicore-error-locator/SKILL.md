---
name: pypto-aicore-error-locator
description: 定位测试案例中出现 aicore error 时的问题 CCE 文件。当需要分析 aicore 错误并找到导致错误的 CCE 文件时使用此技能。
license: 完整条款见 LICENSE.txt
---

# AICore Error 定位器

此技能帮助定位测试案例中出现 aicore error 时的问题 CCE 文件。

## 工作流程

### 1. 收集必要信息

首先使用 `question` 工具向用户收集以下必要信息：

- **pypto目录路径**: 用户必须提供 pypto 项目的根目录路径
- **device log落盘路径**: 用户必须提供 device log 的落盘路径（若不存在则需创建）
- **运行命令**: 用户必须提供触发 aicore error 的测试命令
- **运行目录**: 用户必须提供运行测试命令的目录路径

示例问题配置：
```
question: 
  - header: "PyPTO配置"
    question: "请提供 pypto 目录的完整路径"
    options: []
  - header: "日志路径"
    question: "请提供 device log 落盘路径（不存在将自动创建）"
    options: []
  - header: "运行命令"
    question: "请提供触发 aicore error 的测试命令"
    options: []
  - header: "运行目录"
    question: "请提供运行测试命令的目录路径"
    options: []
```

### 2. 启用追踪日志

根据用户提供的 pypto 目录路径，修改以下配置以启用详细的追踪日志：

- **配置文件**: 搜索修改 `tile_fwk_config.json`
  - 设置 `"fixed_output_path"` 为 `true`
  - 设置 `"force_overwrite"` 为 `false`

- **头文件**: 搜索修改 `aicore_entry.h`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

- **工具头文件**: 搜索修改 `device_switch.h`
  - 设置 `#define ENABLE_COMPILE_VERBOSE_LOG` 为 `1`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

### 3. 重新编译和安装

进入用户提供的 pypto 目录路径，重新编译 pypto 包并pip安装。

### 4. 清理日志

清理日志目录和运行目录中的 kernel_aic* 文件夹：

- 清理 device log 落盘路径下的所有日志信息
- 在用户提供的运行目录下清理 `kernel_aic*` 文件夹

**重要**: 每次运行前必须清理，避免日志混淆。

### 5. 运行测试

在用户提供的运行目录中执行用户提供的测试命令。

**配置日志路径和设置日志级别**：
- 设置 device log 落盘路径：`export ASCEND_PROCESS_LOG_PATH=<用户提供的路径>`
- 若路径不存在则创建该目录
- 设置日志级别：`export ASCEND_GLOBAL_LOG_LEVEL=0`

**重要**: 一定要进入**运行目录**，同时**配置日志路径和设置日志级别**，再执行测试命令。
**重要**: 运行测试的打屏日志中必须出现aicore error，如果未出现，则不适用于该SKILL，请停止运行

### 6. 分析追踪日志

在 device log 落盘路径中搜索 "trace"和"LActStart"、"trace"和"LActFinish" 关键字：
**重要**: <log-file>在 device log 落盘路径中的debug文件夹下，且命名包含device关键字，同时后缀为log
**搜索 LActStart 事件**:
```bash
grep -rn "trace" <log-file> | grep "LActStart"
```

**搜索 LActFinish 事件**:
```bash
grep -rn "trace" <log-file> | grep "LActFinish"
```

### 7. 定位问题 CCE 文件

**对比日志**:
1. 从 LActStart 日志中提取所有 Uid
2. 从 LActFinish 日志中提取所有 Uid
3. 对比两者的Uid，找出在 LActStart 中存在而在 LActFinish 中缺失所有的 Uid，从Uid对应的LEvent提取出`<CCE_ID>`（括号中的最后一个值）
**重要**: 理论上，所有缺失的Uid对应的是同一个`<CCE_ID>`。只有当缺失多个Uid且这些Uid对应的`<CCE_ID>`值不同时，才可能需要输出多个`<CCE_ID>`对应的文件（如果存在）

**解析 LEvent 格式**:
LEvent 格式为 `#LEvent{LUid{0,0,0,<value>,<CCE_ID>},LActStart{<uid>}}`

**查找 CCE 文件**:
在 `kernel_aicore` 目录中查找包含 `_<CCE_ID>_` 后缀为cpp的文件。

### 8. 输出结果

输出找到的 CPP 文件路径。

## 关键点

- 确保 `fixed` 模式启用以保持输出路径不变
- 每次运行前清理日志以避免混淆
- 执行每条命令时，务必确保在用户提供的运行目录下执行
- 通过对比 LActStart 和 LActFinish 事件定位失败的 Uid
- CCE 文件名中包含对应 `_<CCE_ID>_` 的标识符
- 当缺失多个Uid且对应的`<CCE_ID>`值不同时，在 `kernel_aicore` 目录中查找时可能存在多个 `_<CCE_ID>_` 后缀为cpp的文件，一定不要遗漏
