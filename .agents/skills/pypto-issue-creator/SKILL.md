---
name: pypto-issue-creator
description: 基于会话上下文智能创建 GitCode Issue。支持5种类型：Bug Report（报错/异常/精度问题）、Feature Request（新功能/优化）、Documentation（文档问题）、Question（使用问题）、Task（开发任务）。触发词：创建issue、提交issue、反馈问题、报告bug、功能请求、文档问题。
---

# PyPTO Issue Creator

基于会话上下文，智能创建符合 PyPTO 和 CANN 社区规范的 GitCode Issue。

---

## 目录

1. [工作流程](#工作流程)
2. [Issue 类型识别](#issue-类型识别)
3. [去重检查](#去重检查)
4. [环境信息获取](#环境信息获取)
5. [Issue 内容生成](#issue-内容生成)
6. [脚本与参考文档](#脚本与参考文档)

---

## 工作流程

执行以下 9 阶段流程：

```
阶段1-3: 上下文理解与推断
    ↓
阶段4: 去重检查 (GitCode MCP)
    ↓
阶段5: 类型化验证 + 环境信息获取
    ├─ 5a: Bug Report → 自动验证 + 完整环境
    ├─ 5b: Feature Request → 文档/代码/PR/生态对比
    ├─ 5c: Question/Documentation → 文档验证
    └─ 5d: Task → 确认完整
    ↓
阶段6: 智能交互 (补充缺失信息)
    ↓
阶段7: Issue 内容生成
    ↓
阶段8: 最终确认
    ↓
阶段9: 创建 Issue
```

---

## Issue 类型识别

根据会话上下文自动判断类型：

| 类型 | 触发信号 | 必需信息 | 验证要求 |
|-----|---------|---------|---------|
| **Bug Report** | 报错、异常、失败、精度问题、性能问题 | 现象、复现步骤、期望行为 | 自动验证 + 完整环境 |
| **Feature Request** | 希望支持、新增功能、优化、扩展 | 功能描述、使用场景 | 文档/代码/PR/生态对比 |
| **Documentation** | 文档缺失、文档错误、文档改进 | 问题描述、影响范围 | 文档验证 |
| **Question** | 如何使用、为什么、不理解 | 问题描述 | 文档验证 |
| **Task** | 需要开发、计划、任务 | 任务描述、验收标准 | 确认完整 |

---

## 去重检查

**阶段4执行**，在生成 Issue 内容前必须检查：

### 1. GitCode 远程检查

```bash
# 使用 GitCode MCP 工具
gitcode_search_issues(query="repo:cann/pypto {关键词}")
```

### 2. 处理策略

- **发现重复**: 展示已有 Issue，询问用户是否继续
- **无重复**: 继续下一阶段

---

## 环境信息获取

**仅 Bug Report 类型需要获取完整环境信息**。

### 自动获取命令

| 信息 | 命令 |
|-----|------|
| CANN 版本 | `echo $ASCEND_HOME_PATH \| grep -oP 'cann-\\K[\\d.]+'` |
| PyPTO Commit | `COMMIT=$(git merge-base HEAD $(git remote -v \| grep 'gitcode.com/cann/pypto.git' \| head -1 \| cut -f1)/master 2>/dev/null \|\| git merge-base HEAD origin/master 2>/dev/null) && git log -1 --format='%H %ci' $COMMIT \|\| echo "Unknown"` |
| 服务器类型 | `lspci -n -D \| grep '19e5:d80[23]' \| sed 's/.*d80\\([23]\\).*/A\\1/' \| head -n1`|
| Python 版本 | `python --version` |
| 操作系统 | `cat /etc/os-release \| grep PRETTY_NAME` |

若上述命令无法正确获取环境信息，查询[environment-commands.md](references/environment-commands.md)获取进阶用法

### 获取策略

1. **优先从会话上下文推断** - 用户已提供则直接使用
2. **执行命令获取** - 自动运行命令
3. **请求用户提供** - 无法自动获取时询问

---

## Issue 内容生成

### 模板选择

根据类型加载对应模板（详见 [references/issue-templates.md](references/issue-templates.md)）：

- **Bug Report**: 现象 → 复现步骤 → 期望行为 → 环境信息 → 日志 → 代码
- **Feature Request**: 功能描述 → 使用场景 → 期望API → 动机 → 生态对比
- **Documentation**: 问题描述 → 影响范围 → 建议修改
- **Question**: 问题描述 → 上下文 → 尝试过的方法
- **Task**: 任务描述 → 背景 → 具体要求 → 验收标准

### 生态对比 (Feature Request 专用)

收集以下框架的相关实现：

- **PyTorch**: `https://pytorch.org/docs/`
- **Triton**: `https://triton-lang.org/`
- **cuTile**: [NVIDIA CUDA Tile](https://docs.nvidia.com/cuda/cutile-python/)
- **TileLang**: `https://github.com/tile-ai/tilelang`

### 标签映射

| 类型 | 默认标签 |
|-----|---------|
| Bug Report | `bug`, `needs-triage` |
| Feature Request | `enhancement`, `needs-discussion` |
| Documentation | `documentation` |
| Question | `question` |
| Task | `task`, `needs-assignment` |

---

## 脚本与参考文档

### references/

| 文件 | 内容 |
|-----|------|
| [issue-templates.md](references/issue-templates.md) | 5种Issue类型的完整模板 |
| [environment-commands.md](references/environment-commands.md) | 环境信息获取命令详解 |

---

## 执行检查清单

创建 Issue 前确认：

- [ ] 类型判断正确
- [ ] 去重检查完成
- [ ] 必需信息完整
- [ ] 环境信息已获取（Bug Report）
- [ ] 生态对比已完成（Feature Request）
- [ ] 内容符合模板规范
- [ ] 用户已确认
