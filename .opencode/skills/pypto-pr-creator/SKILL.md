---
name: pypto-pr-creator
description: "PyPTO 项目 PR 创建全流程指南。当需要为 cann/pypto 仓库创建 PR、编写 commit 信息、准备代码提交、检查 PR 规范时使用。覆盖：(1) 仓库发现与 fork 验证，(2) 用户确认检查点（强制阻塞确认），(3) 分支创建与 commit，(4) 通过 GitCode MCP 创建/更新 PR，(5) PR 创建后结构化报告。触发词：创建PR、提交PR、PR规范、commit message、pypto贡献、代码提交到pypto、更新PR。"
---

# PyPTO PR Creator

---

## ⚠️ 核心概念：跨 Fork PR

> **本 skill 的目标是创建从用户 fork 仓库到上游仓库 `cann/pypto` 的 PR**

```
┌─────────────────────┐      PR       ┌─────────────────────┐
│  <username>/pypto   │ ─────────────▶│     cann/pypto      │
│    (用户 fork)      │               │    (上游主仓库)      │
│  add-shared-skills  │               │       master        │
└─────────────────────┘               └─────────────────────┘
```

| 项目 | 值 |
|------|-----|
| **源仓库** | `<username>/pypto`（用户 fork） |
| **源分支** | `<branch_name>`（如 `feat/add-skill`） |
| **目标仓库** | `cann/pypto`（上游主仓库） |
| **目标分支** | `master` |
| **PR 链接格式** | `https://gitcode.com/cann/pypto/merge_requests/<pr_id>` |

> ⚠️ **常见错误**：在 fork 仓库内创建 PR（链接为 `gitcode.com/<username>/pypto/...`），这是**错误**的！
> ✅ **正确做法**：PR 必须指向 `cann/pypto`（链接为 `gitcode.com/cann/pypto/...`）

---

## 强制约束

| 规则 | 说明 |
|------|------|
| **跨 Fork PR** | PR 目标必须是 `cann/pypto`，不是用户 fork |
| **远程操作** | 所有远程操作必须通过 GitCode MCP，禁止直接使用 `GITCODE_TOKEN` |
| **Origin 配置** | `origin` 必须指向用户 fork（如 `<username>/pypto`），不能是 `cann/pypto` |
| **隐私保护** | ⚠️ **禁止打印 `GITCODE_TOKEN`，包括屏幕、日志、调试信息** |
| **用户确认** | 创建分支、commit、push、创建/更新 PR 前必须获得用户明确确认 |
| **文件路径** | 使用 `$PYPTO_REPO` 指代用户的 pypto 本地仓库根目录 |
| **Commit 信息** | 必须使用英文，不超过 10 行 |
### GitCode MCP 工具

- PR 操作：`gitcode_create_pull_request`, `gitcode_update_pull_request`, `gitcode_list_pull_requests`
- 仓库查询：`gitcode_get_repository`

未安装时按 `gitcode-mcp-install` skill 完成配置。

---

## 完整工作流（9 个阶段）

### 阶段 1：仓库发现与验证

**目标**：找到用户的 pypto fork 仓库，验证 origin 配置正确。

**执行逻辑**（按优先级）：

1. 检查当前工作目录是否是 pypto 仓库（`git rev-parse --is-inside-work-tree`）
2. 检查 origin：必须包含 `pypto` 且不能包含 `cann/pypto`
3. 若当前目录不符合，在工作区搜索（`find` 搜索 `.git` 目录，检查 remote）
4. 检测浅克隆：`git rev-parse --is-shallow-repository`（若 true，后续需 `git fetch --unshallow origin`）

### 阶段 2：Git 认证检查（关键）

> ⚠️ **Push 需要 Git 认证，必须检查并配置认证方式**

#### 2.1 检测当前认证状态

```bash
# 一键检测脚本
echo "=== Git 认证状态检测 ==="
echo "1. GITCODE_TOKEN 环境变量: $([ -n "$GITCODE_TOKEN" ] && echo '✅ 已设置' || echo '❌ 未设置')"
echo "2. SSH Key: $(ls ~/.ssh/*.pub 2>/dev/null | wc -l) 个公钥"
echo "3. credential.helper: $(git config --global credential.helper 2>/dev/null || echo '未配置')"
echo "4. .git-credentials: $([ -f ~/.git-credentials ] && echo '✅ 存在' || echo '❌ 不存在')"
echo "5. SSH Agent: $(ssh-add -l 2>/dev/null && echo '运行中' || echo '未运行')"
```

#### 2.2 认证方式对比

| 方式 | 安全性 | 持久性 | 配置难度 | 推荐场景 |
|------|--------|--------|----------|----------|
| **1. cache** | ⭐⭐⭐⭐⭐ 内存 | 临时（可设超时） | 简单 | 容器/临时环境 |
| **2. store** | ⭐ 明文 | 持久 | 简单 | 个人开发机 |
| **3. SSH Key** | ⭐⭐⭐⭐⭐ 加密 | 持久 | 中等 | 长期开发 |
| **4. GITCODE_TOKEN + URL** | ⭐⭐ 环境变量 | 临时 | 简单 | CI/CD |
| **5. libsecret** | ⭐⭐⭐⭐⭐ 系统加密 | 持久 | 中等 | Linux 桌面 |

#### 2.3 各方式配置方法

**方式 1: credential.helper cache（推荐容器环境）**
```bash
# 缓存 7 天
git config --global credential.helper 'cache --timeout=604800'
# 首次 push 输入用户名和 Token，后续自动使用缓存
```

**方式 2: credential.helper store**
```bash
# 配置
git config --global credential.helper store
# 首次 push 输入用户名和 Token，保存到 ~/.git-credentials
```

**方式 3: SSH Key（需在 GitCode 网站配置）**
```bash
# 生成 SSH Key
ssh-keygen -t ed25519 -C "your@email.com"

# 查看公钥（添加到 GitCode → Settings → SSH Keys）
cat ~/.ssh/id_ed25519.pub

# 使用 SSH URL
git remote set-url origin git@gitcode.com:<username>/pypto.git
```

**方式 4: GITCODE_TOKEN + URL 内嵌（CI/CD 推荐）**
```bash
# 设置环境变量
export GITCODE_TOKEN="your-token"

# 配置 URL
git remote set-url origin https://oauth2:${GITCODE_TOKEN}@gitcode.com/<username>/pypto.git
```

**方式 5: libsecret（Linux 桌面）**
```bash
# 安装依赖
sudo apt install libsecret-1-0 libsecret-1-dev libglib2.0-dev

# 编译
cd /usr/share/doc/git/contrib/credential/libsecret && sudo make

# 配置
git config --global credential.helper /usr/share/doc/git/contrib/credential/libsecret/git-credential-libsecret
```

#### 2.4 用户确认

向用户展示检测结果，询问使用哪种方式：
```
检测到当前认证状态：
- GITCODE_TOKEN: [已设置/未设置]
- SSH Key: [X 个/无]
- credential.helper: [cache/store/未配置]

请选择认证方式：
1. cache - 内存缓存，安全但临时
2. store - 明文存储，方便但有风险
3. SSH Key - 最安全，需预先配置
4. GITCODE_TOKEN - 环境变量，适合 CI/CD
5. libsecret - 系统加密，Linux 桌面推荐

请输入选择 (1-5):
```

> **在用户选择并配置认证之前，禁止执行 push 操作。**

| Origin 配置 | 判定 | 处理 |
|-------------|------|------|
| `<username>/pypto`（用户 fork） | ✅ | 直接使用 |
| `cann/pypto`（upstream） | ❌ | 需修复 origin：`git remote set-url origin https://gitcode.com/<username>/pypto.git`；或重新克隆 fork：`git clone https://gitcode.com/<username>/pypto.git` |
| 无 origin 或非 pypto | ❌ | 询问用户正确的仓库路径 |

**通过 GitCode MCP 验证 fork 关系**：

```python
result = gitcode_get_repository(owner="<username>", repo="pypto")
# 验证: result.parent.full_name == "cann/pypto"
```

### 阶段 3：用户确认（强制阻塞）

> **在获得用户明确确认之前，禁止执行任何 git 操作。**

向用户展示执行计划表（包含以下字段），等待确认：

- 本地仓库路径（`$PYPTO_REPO`）
- Fork 仓库（`<username>/pypto`）
- 本地分支名
- Commit 信息（`tag(scope): Summary` 格式）
- Push 目标（origin → 分支名）
- PR 目标（`cann/pypto` → `master`）
- PR 标题与 Body 预览

### 阶段 4：预检修复

用户确认后执行：

#### 3.1 基础检查

- 浅克隆修复：若 `git rev-parse --is-shallow-repository` 为 true → `git fetch --unshallow origin`
- 再次验证 origin 不是 `cann/pypto`

#### 3.2 Upstream 同步检查（关键）

> ⚠️ **分支落后于 upstream 是 PR 创建失败（pre-receive hook）的常见原因**

**步骤 1：添加 upstream remote（如不存在）**
```bash
# 检查是否已有 upstream
git remote -v | grep upstream

# 若无，添加 upstream
git remote add upstream https://gitcode.com/cann/pypto.git
```

**步骤 2：获取 upstream 最新状态**
```bash
git fetch upstream master
```

**步骤 3：检查分支是否落后**
```bash
# 检查当前分支落后于 upstream 多少个 commit
git log --oneline HEAD..upstream/master | wc -l

# 若输出 > 0，说明分支落后，需要 rebase
```

**步骤 4：rebase 到 upstream（如落后）**
```bash
# 确保 origin 配置了认证（避免 push 时失败）
git remote set-url origin https://oauth2:${GITCODE_TOKEN}@gitcode.com/<username>/pypto.git

# Rebase
git rebase upstream/master

# Force push 更新 fork 分支
git push -f origin <branch_name>
```
### 阶段 5：创建分支、Commit、Push

```bash
git -C "$PYPTO_REPO" checkout -b <branch_name>
git -C "$PYPTO_REPO" add <files>
git -C "$PYPTO_REPO" commit -m "tag(scope): Summary"
git -C "$PYPTO_REPO" push origin <branch_name>
```

> **认证说明**：push 依赖 git credential helper 或 `.gitconfig` 中已配置的凭据。

> **⚠️ Commit Message 格式（Pre-receive Hook 强制验证）**：`^(feat|fix|docs|style|refactor|perf|test)(.*): [A-Z].{10,200}` — Tag 必须是这 7 种之一，冒号后必须有空格，Summary 首字母必须大写且长度 10-200 字符。验证：`git log -1 --format="%s" | grep -E '^(feat|fix|docs|style|refactor|perf|test)(.*): [A-Z].{10,200}'`

#### Push 失败诊断与修复

若 push 报认证错误（401/403），按以下步骤排查：

**步骤 1：检查 http.extraheader 配置**

```bash
git config --local --get http.extraheader
# 如输出包含 "Authorization: Bearer ..." → 这是问题根源
```

**原因**：GitCode 不支持 Bearer token 认证，只支持 HTTP Basic Auth。若配置了 `http.extraheader=Authorization: Bearer <token>`，git 会强制使用 Bearer token 导致认证失败。

**步骤 2：删除错误的配置并修复认证**

```bash
# 删除错误的 Bearer token 配置
git config --local --unset http.extraheader

# 配置正确的 credential helper
git config --local credential.helper store

# 存储凭据（使用 HTTP Basic Auth）
git credential-store store << 'EOF'
protocol=https
host=gitcode.com
username=<your_username>
password=<your_token>
EOF
```

**步骤 3：调试认证问题（可选）**

```bash
# 查看实际发送的认证头
GIT_CURL_VERBOSE=1 git push origin <branch_name> 2>&1 | grep -i authorization
# 正确: Authorization: Basic <base64>
# 错误: Authorization: Bearer <token>
```
### 阶段 6：创建或更新 PR

#### 5.1 判断创建还是更新

```python
# 查询是否已有该分支的 open PR
prs = gitcode_list_pull_requests(owner="cann", repo="pypto")
# 筛选: state == "opened" 且 source_branch 匹配当前分支

# 如存在匹配 PR → 询问用户：更新现有 PR 还是创建新 PR
# 如不存在 → 创建新 PR
```

#### 5.2 创建新 PR

```python
gitcode_create_pull_request(
    owner="cann",                          # 上游仓库 owner（固定）
    repo="pypto",                          # 上游仓库名（固定）
    title="tag(scope): Summary",           # PR 标题
    head="<username>:<branch_name>",       # ⚠️ 必须是 "fork_owner:branch" 格式
    base="master",                         # 目标分支
    body="..."                             # PR 描述（禁止为空）
)
```

**关键细节**：
- `head` 格式为 `<username>:<branch_name>`（冒号分隔），不是纯分支名
- `owner`/`repo` 指向**上游仓库**（`cann/pypto`），不是 fork
- 所有参数名必须**小写**

> 完整参数说明和示例见 [references/pr-spec.md](references/pr-spec.md)。

#### 5.3 更新现有 PR

```python
gitcode_update_pull_request(
    owner="cann",
    repo="pypto",
    pull_number=<pr_number>,    # 目标 PR 编号
    title="新标题",              # 可选
    body="新描述",               # 可选
    state="open"                # 可选: "open" 或 "closed"
)
```

#### 5.4 MCP 失败后的 curl 兜底方案

> ⚠️ GitCode MCP 的 `create_pull_request` 可能返回 400 错误，即使参数正确。此时使用 curl 直接调用 API。

**兜底步骤**：

```bash
# 直接调用 GitCode API 创建 PR
curl -s -X POST "https://api.gitcode.com/api/v5/repos/cann/pypto/pulls" \
  -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "tag(scope): Summary",
    "head": "<username>:<branch_name>",
    "base": "master",
    "body": "PR 描述内容"
  }'
```

**成功响应**：返回 PR JSON，包含 `html_url` 字段

**失败响应**：
- `pre receive hook check failed` → 检查 commit message 格式和 upstream 同步
- `400 Unknown` → 检查参数格式

> 注意：此兜底方案仅在 MCP 工具失败时使用，正常情况优先使用 MCP。

### 阶段 7：PR 创建后报告

PR 操作成功后，向用户展示结构化报告，包含：

- **PR 链接**（取自返回值的 `html_url` 字段）
  - ⚠️ **必须验证**：链接格式为 `https://gitcode.com/cann/pypto/merge_requests/<pr_id>`
  - ❌ **错误格式**：`https://gitcode.com/<username>/pypto/merge_requests/<pr_id>`（这是 fork 内 PR）
- 操作类型（创建/更新）
- PR 标题、源分支 → 目标分支
- Commit hash 与 message
- 后续操作提示（发送 `compile` 触发 CI、等待 review）

**验证示例**：
```python
# 正确的 PR 链接
pr_url = result["html_url"]
assert "cann/pypto" in pr_url, f"PR 链接错误：{pr_url}，应为 cann/pypto"
```

### 阶段 8：CLA 检查（关键）

> ⚠️ **CLA (Contributor License Agreement) 检查是 PR 合并的前置条件**

#### 8.1 检查 CLA 状态

PR 创建/更新后，必须检查 CLA 是否通过：

```python
# 获取 PR 详情，检查 labels
pr = gitcode_get_pull_request(owner="cann", repo="pypto", pull_number=<pr_number>)

# 检查 CLA 标签
labels = pr.get("labels", [])
cla_labels = [l for l in labels if "cla" in l.get("name", "").lower()]

# 判断 CLA 状态
if any("cla/no" in l.get("name", "") for l in cla_labels):
    # CLA 未通过
    cla_passed = False
elif any("cla/yes" in l.get("name", "") or "cla/pass" in l.get("name", "") for l in cla_labels):
    # CLA 已通过
    cla_passed = True
else:
    # 无 CLA 标签，可能需要等待或手动触发
    cla_passed = None
```

#### 8.2 CLA 失败处理

若 CLA 检查失败（标签含 `cann-cla/no` 或类似），执行以下诊断和修复步骤：

**步骤 1：检查本地 Git 配置**

```bash
# 检查当前 commit 作者信息
git log -1 --format='Author: %an <%ae>'

# 检查全局配置
git config --global user.name
git config --global user.email
```

**步骤 2：验证邮箱与 GitCode 账户一致**

> ⚠️ **CLA 检查基于 commit 作者邮箱，必须与 GitCode 账户主邮箱一致**

常见失败原因：
- Git 本地配置的邮箱与 GitCode 账户邮箱不一致
- 使用了公司邮箱（如 `@hisilicon.com`）但 GitCode 账户未添加该邮箱
- Commit 由多人协作，存在不同作者的 commit

**步骤 3：修复邮箱配置**

```bash
# 方式 1: 修改全局配置（推荐）
git config --global user.name "your_username"
git config --global user.email "your_email@example.com"

# 方式 2: 修改最近一次 commit 的作者（已 push 需 force push）
git commit --amend --author="Your Name <your_email@example.com>"
git push -f origin <branch_name>

# 方式 3: 在 GitCode 添加邮箱
# 访问 GitCode → Settings → Emails → Add email
# 添加 commit 中使用的邮箱并验证
```

**步骤 4：重新触发 CLA 检查**

```bash
# 修改作者信息后，添加空 commit 触发重新检查
git commit --allow-empty -m "docs: trigger CLA check"
git push origin <branch_name>
```

#### 8.3 CLA 检查提示模板

```
📋 CLA 检查结果: ❌ 未通过

检测到 PR 标签包含 "cann-cla/no"，表示 CLA 验证失败。

请检查以下配置：
1. 本地 Git 邮箱: $(git config --global user.email)
2. GitCode 账户邮箱: 请在 GitCode → Settings → Emails 中确认

修复步骤：
- 确保两个邮箱一致，或在 GitCode 添加本地邮箱
- 修改后执行: git commit --amend --reset-author --no-edit && git push -f

参考文档: https://gitcode.com/help/user/cla
```

#### 8.4 用户确认

CLA 失败时，向用户展示诊断信息并询问：

```
检测到 CLA 检查未通过（标签: cann-cla/no）。

当前 Git 配置:
- user.name: <当前值>
- user.email: <当前值>

可能原因:
1. 邮箱与 GitCode 账户不一致
2. GitCode 账户未添加该邮箱

请选择操作:
1. 修改 Git 配置（输入正确的用户名和邮箱）
2. 在 GitCode 添加邮箱（手动操作）
3. 跳过，稍后处理
```

### 阶段 9：追加修改（可选）

若需修改已有 PR：在同一分支追加 commit 并 push，PR 自动更新。如需修改标题/描述，用 `gitcode_update_pull_request`。
---

## PR 标题与 Body 规范

详见 [references/pr-spec.md](references/pr-spec.md)。

### 速查

| 规则 | 说明 |
|------|------|
| 格式 | `tag(scope): Summary` |
| Tag | feat / fix / docs / style / refactor / test / perf |
| 语言 | **必须使用英文** |
| Summary | 英文、首字母大写、无句号、祈使语气 |
| Body | 禁止为空，描述动机+变更列表 |
| 长度 | **整个 commit message 不超过 10 行** |

### Commit 信息示例

| ✅ 好的示例 | ❌ 不好的示例 |
|-------------|---------------|
| `feat(skills): Add PR creator skill` | `feat(skills): 为 PyPTO 项目添加 PR 创建技能` |
| `fix(ops): Fix precision issue in softmax` | `docs(api): 更新文档` |
| `docs(api): Update tensor creation doc` | `feat: add feature`（无 scope） |

**要点**：
- **必须使用英文**（禁止中文）
- 整个 commit message 控制在 **10 行以内**
- 格式：`tag(scope): Summary` + Body（可选多行）


---

## 提交前检查清单

详见 [references/checklist.md](references/checklist.md)。

---

## 常见陷阱

|| 陷阱 | 解决方案 |
|------|---------|
| origin 指向 `cann/pypto` | `git remote set-url origin https://gitcode.com/<username>/pypto.git`，或重新克隆你的 fork |
| push 报 "shallow update not allowed" | `git fetch --unshallow origin` |
| push 报认证错误 401/403 | 检查并删除 `http.extraheader` Bearer 配置，使用 credential.helper store |
| git 强制使用 Bearer token | `git config --local --unset http.extraheader` 并重新配置 credential |
| PR 创建报 "pre receive hook check failed" | (1) commit message 不符合 `tag(scope): Summary`；(2) **分支落后于 upstream** |
| 分支落后于 upstream | `git fetch upstream master && git rebase upstream/master && git push -f origin <branch>` |
| PR 创建返回 400 | 检查 `head` 是否用了 `<username>:<branch_name>` 格式 |
| `head` 只有分支名 | 必须是 `<username>:<branch_name>` |
| `owner` 指向了 fork | `owner` 应为 `cann`（上游仓库） |
|| **PR 链接指向 fork** | PR 链接必须是 `cann/pypto/merge_requests/<id>`，不是 `<username>/pypto/...` |
|| MCP 创建 PR 返回 400 但参数正确 | 使用 curl 兜底方案（见阶段 5.4） |
| commit message 超过 10 行 | 精简内容，控制在 10 行以内 |
| commit message 使用中文 | **必须使用英文** |

| **无 Git 认证配置** | 按阶段 2 配置 credential.helper 或 SSH Key |
| **SSH Key 未添加到 GitCode** | 将公钥添加到 GitCode → Settings → SSH Keys |
| **Token 权限不足** | 创建 Token 时勾选 `repo`、`read:user` 权限 |
| **credential.helper 未生效** | 检查 `git config --global credential.helper` 输出 |
| **cache 超时失效** | 重新 push 输入凭证，或增大 `--timeout` |
|| **CLA 检查失败 (cann-cla/no)** | 检查 `git config user.email` 是否与 GitCode 账户邮箱一致，见阶段 8 |
|| **Git 邮箱与 GitCode 不一致** | 修改 `git config
--global user.email` 或在 GitCode → Settings → Emails 添加邮箱 |

## 相关文件

| 文件 | 路径 |
|------|------|
| 贡献指南 | `$PYPTO_REPO/CONTRIBUTION.md` |
| PR 模板 (EN) | `$PYPTO_REPO/.gitcode/PULL_REQUEST_TEMPLATE.md` |
| PR 模板 (CN) | `$PYPTO_REPO/.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` |
| 详细规范 | `$PYPTO_REPO/docs/contribute/pull-request.md` |
| 代码检查规则 | `$PYPTO_REPO/docs/contribute/code-check-rule.yaml` |

> `$PYPTO_REPO` = 用户的 pypto 本地仓库根目录
