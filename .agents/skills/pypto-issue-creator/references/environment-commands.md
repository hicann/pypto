# 环境信息获取命令

本文档定义了获取环境信息的标准命令。

---

## 快速参考

| 信息类型 | 命令 | 输出示例 |
|---------|------|---------|
| CANN 版本 | `echo $ASCEND_HOME_PATH \| grep -oP 'cann-\\K[\\d.]+'` | `8.5.0` |
| PyPTO Commit | `COMMIT=$(git merge-base HEAD $(git remote -v \| grep 'gitcode.com/cann/pypto.git' \| head -1 \| cut -f1)/master 2>/dev/null \|\| git merge-base HEAD origin/master 2>/dev/null) && git log -1 --format='%H %ci' $COMMIT \|\| echo "Unknown"` | `abc123... 2025-03-10 10:00:00 +0800` |
| 服务器类型 | `lspci -n -D \| grep '19e5:d80[23]' \| sed 's/.*d80\\([23]\\).*/A\\1/'` | `A3` |
| Python 版本 | `python --version` | `Python 3.10.12` |
| 操作系统 | `cat /etc/os-release \| grep PRETTY_NAME` | `PRETTY_NAME="Ubuntu 22.04.3 LTS"` |

---

## 详细说明

### CANN 版本

```bash
# 方法1：从环境变量获取
echo $ASCEND_HOME_PATH | grep -oP 'cann-\K[\d.]+'

# 方法2：从 npu-smi 获取
npu-smi info | grep Version

# 方法3：从 CANN 安装目录
ls -d /usr/local/Ascend/ascend-toolkit/* | grep -oP '\d+\.\d+\.\d+'
```

### PyPTO Commit

```bash
# 获取本地存在于 cann/pypto master 的最新commit
# 优先使用 cann/pypto 远程，回退到 origin/master
COMMIT=$(git merge-base HEAD $(git remote -v | grep 'gitcode.com/cann/pypto.git' | head -1 | cut -f1)/master 2>/dev/null || git merge-base HEAD origin/master 2>/dev/null) && git log -1 --format='%H %ci' $COMMIT || echo "Unknown"

# 仅获取短哈希
git rev-parse --short HEAD

# 获取分支信息
git branch --show-current
```

### 服务器类型

```bash
# A3 检测 (d803 = A3)
lspci -n -D | grep '19e5:d803' && echo "A3 detected"

# A2 检测 (d802 = A2)
lspci -n -D | grep '19e5:d802' && echo "A2 detected"

# 综合检测
lspci -n -D | grep '19e5:d80[23]' | sed 's/.*d80\([23]\).*/A\1/'
```

**设备ID说明**:
- `19e5:d802` → A2 服务器
- `19e5:d803` → A3 服务器

### Python 版本

```bash
# 完整版本
python --version

# 仅版本号
python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'

# 检查 Python 路径
which python
```

### 操作系统

```bash
# 发行版名称
cat /etc/os-release | grep PRETTY_NAME

# 内核版本
uname -r

# 完整系统信息
uname -a
```

---

## 批量获取脚本

```bash
#!/bin/bash
# 获取所有环境信息

echo "=== 环境信息 ==="
echo "CANN 版本: $(echo $ASCEND_HOME_PATH | grep -oP 'cann-\K[\d.]+')"
echo "PyPTO Commit: $(COMMIT=$(git merge-base HEAD $(git remote -v | grep 'gitcode.com/cann/pypto.git' | head -1 | cut -f1)/master 2>/dev/null || git merge-base HEAD origin/master 2>/dev/null) && git log -1 --format='%h %ci' $COMMIT 2>/dev/null || echo 'Unknown')"
echo "服务器类型: $(lspci -n -D 2>/dev/null | grep '19e5:d80[23]' | sed 's/.*d80\([23]\).*/A\1/' | head -1 || echo 'Unknown')"
echo "Python 版本: $(python --version 2>&1)"
echo "操作系统: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo 'Unknown')"
```

---

## 环境信息获取策略

1. **优先使用环境变量** - 最可靠的信息来源
2. **验证命令可用性** - 使用 `command -v` 检查命令是否存在
3. **优雅处理错误** - 命令失败时返回 "Unknown" 或 "N/A"
4. **合并相关命令** - 减少工具调用次数

---

## 在 Skill 中的使用

Agent 应按以下优先级获取环境信息：

1. **执行命令获取** - 使用上述命令获取
2. **请求用户提供** - 无法自动获取时询问

**重要**: 只有创建 Bug Report 类型 Issue 时才需要获取完整环境信息。其他类型 Issue 可根据需要选择性获取。
