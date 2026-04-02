---
name: pypto-host-stacktrace-analyzer
description: 分析 host 侧捕获异常后的堆栈信息，通过地址到源码行映射和符号解析，定位问题代码位置。支持 Python traceback、C++ stack trace 和混合堆栈的自动识别与分析。支持编译Debug版本PyPTO包并定位具体代码行。Triggers："堆栈分析"、"堆栈反汇编"、"分析堆栈信息"、"地址到源码行"、"stack trace"、"backtrace"
---

# 堆栈信息分析器

此技能用于分析 host 侧捕获异常后的堆栈信息，通过反汇编和符号解析，定位问题代码位置。

## 工作流程

1. 收集堆栈信息
2. 检查是否有调试符号
3. 如果没有调试符号，编译Debug版本PyPTO
4. 使用综合分析器自动分析
5. 输出完整报告（包含源码行号）

---

## 步骤 1：收集堆栈信息

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息，严禁猜测或使用默认值。**

使用 `question` 工具收集以下信息：

- **stack_text**: 堆栈信息文本（直接粘贴或文件路径）
- **stack_file**: 堆栈信息文件路径（可选，如果提供了文本则不需要）
- **stack_source**: 堆栈来源的测试用例文件路径（可选，如果编译Debug版本后需要重新运行）

**⚠️ 重要**：如果 `stack_source` 是测试用例文件，编译Debug版本后需要重新运行该用例以获取新的堆栈信息。

将收集的路径全部转换成绝对路径，收集到所有信息后才能继续后续步骤。

---

## 步骤 2：检查调试符号

运行调试符号检查脚本：

```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py --check-debug
```

**输出**：
- ✓ 二进制文件包含调试符号
- ✗ 二进制文件不包含调试符号

---

## 步骤 3：编译Debug版本PyPTO（如果需要）

如果步骤2显示没有调试符号，则需要编译Debug版本：

```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/build_debug_pypto.py
```

**编译选项**：
- `-p, --pypto-root`: PyPTO项目根目录（默认：当前目录）
- `-t, --timeout`: 编译超时时间（秒，默认：600）
- `--skip-check`: 跳过前提条件检查

**编译完成后会自动**：
- 找到编译生成的wheel文件
- 显示编译信息

---

## 步骤 4：安装Debug版本

编译完成后，安装Debug版本：

```bash
pip install build_out/pypto*.whl --force-reinstall --no-deps
```

---

## 步骤 5：重新运行测试用例并收集新堆栈信息 ⭐ 新增

**⚠️ 重要：如果用户在步骤1中提供了 `stack_source`（测试用例文件路径），则需要执行此步骤**

如果步骤1中用户提供了测试用例文件路径，则需要重新运行该用例以获取新的堆栈信息：

```bash
# 假设用户提供的是测试用例文件 /path/to/test_case.py
python3 /path/to/test_case.py --run_mode sim 2>&1 | tee /tmp/new_stack_trace.log
```

**然后从新输出的日志中提取堆栈信息**：

```bash
# 提取 C++ stack trace
grep -A 50 "libtile_fwk_interface.so" /tmp/new_stack_trace.log > /tmp/new_cpp_stack.txt

# 或提取 Python traceback
grep -A 100 "Traceback" /tmp/new_stack_trace.log > /tmp/new_python_stack.txt

# 或提取完整堆栈（包含错误信息）
grep -A 100 "Run pass failed" /tmp/new_stack_trace.log > /tmp/new_stack_trace.txt
```

**⚠️ 注意事项**：
- 使用与原始堆栈信息相同的运行参数（如 `--run_mode sim`）
- 将新堆栈信息保存到文件中，供后续分析使用
- 确保新堆栈信息包含完整的错误信息

---

## 步骤 6：使用综合分析器自动分析

运行综合分析脚本，自动完成所有分析步骤：

```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/comprehensive_analyzer.py <stack_file> -f
```

**综合分析器会自动完成以下步骤**：

1. **提取错误信息**
   - 错误码 (Errcode)
   - 错误位置 (file, line, func)
   - 错误消息

2. **解析 Python traceback**
   - 自动识别 Python traceback 格式
   - 提取文件、行号、函数名
   - 标记错误触发点

3. **解析 C++ stack trace**
   - 自动识别 C++ stack trace 格式
   - 支持 `libtile.so(function+offset) [address]` 格式
   - 支持 `#0 0xaddress in function at file.c:line` 格式
   - 标记错误发生点

4. **自动查找二进制文件**
   - 从堆栈信息中提取二进制文件名
   - 在多个路径中搜索：
     - 当前目录
     - PATH 环境变量
     - PyPTO 安装路径
     - 常见库路径 (/usr/lib, /usr/local/lib)
   - 验证二进制文件有效性

5. **符号反混淆**
   - 使用 `c++filt` 工具反混淆 C++ 符号
   - 显示原始符号和反混淆后的符号

6. **源码行定位** ⭐ 新功能
   - 使用 `addr2line` 工具定位地址到源码行
   - 显示函数名、文件名、行号
   - 仅在Debug版本中可用

---

## 步骤 6：输出完整报告

综合分析器会自动生成包含以下内容的完整报告：

### 错误信息
- 错误码
- 错误位置
- 错误消息

### Python Traceback
- 总帧数
- 每帧的详细信息（文件、行号、函数）
- 错误触发点标记

### C++ Stack Trace
- 总帧数
- 每帧的详细信息（二进制、符号、偏移、地址）
- 符号反混淆结果
- **源码行号** ⭐ 新功能
- 错误发生点标记

### 二进制文件
- 找到的二进制文件路径

---

## 高级功能

### 单独使用各个脚本

#### 1. 编译Debug版本PyPTO ⭐ 新功能
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/build_debug_pypto.py
```

#### 2. 源码定位 ⭐ 新功能
```bash
# 检查是否有调试符号
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py --check-debug

# 定位地址
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py -a 0x19d600a

# 定位符号
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py -s "npu::tile_fwk::Pad(long, long)"
```

#### 3. 解析堆栈信息
```bash
# 从文本解析
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/parse_stack.py "stack text"

# 从文件解析
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/parse_stack.py /path/to/stack.log -f

# 输出 JSON 格式
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/parse_stack.py /path/to/stack.log -f -j
```

#### 4. 提取错误信息
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/extract_error_info.py /path/to/stack.log -f
```

#### 5. 自动搜索二进制文件
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/auto_find_binary.py binary_name
```

#### 6. 地址到源码行映射
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/resolve_address.py /path/to/binary 0xaddress1 0xaddress2
```

#### 7. 符号解析
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/resolve_symbol.py /path/to/binary 0xaddress1 0xaddress2
```

#### 8. 符号反混淆
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/demangle_symbols.py _ZN3npu8tile_fwk11HostMachine17CompileThreadFuncEv
```

---

## 支持的堆栈格式

### Python traceback
```
Traceback (most recent call last):
  File "script.py", line 10, in <module>
    function()
  File "script.py", line 5, in function
    raise Exception("Error")
```

### C++/C stack trace (gdb)
```
#0  0x00007f1234567890 in function_name (args) at file.c:123
#1  0x00007f1234567890 in caller_function (args) at file.c:100
```

### PyPTO 格式
```
[0] 0x7f1234567890 in function_name at file.py:123
[1] 0x7f1234567890 in caller_function at file.py:100
```

### PyPTO C++ 格式（新增支持）⭐
```
libtile_fwk_interface.so(npu::tile_fwk::HostMachine::CompileFunction(npu::tile_fwk::Function*) const+0x744) [0xfffef3d2c3bc]
libtile_fwk_interface.so(npu::tile_fwk::HostMachine::Compile(npu::tile_fwk::MachineTask*) const+0x25c) [0xfffef3d2f21c]
```

### 通用格式
```
0x7f1234567890 function_name+0x1234 (/path/to/binary)
0x7f1234567890 caller_function+0x5678 (/path/to/binary)
```

---

## 关键注意事项

1. **工具依赖**: 确保 `addr2line`/`llvm-addr2line`、`objdump`/`llvm-objdump` 和 `c++filt` 工具可用
2. **路径规范**: 所有路径必须使用绝对路径
3. **信息收集**: 第一步必须通过 `question` 工具收集信息，严禁猜测
4. **自动分析**: 推荐使用综合分析器，它会自动完成所有分析步骤
5. **混合堆栈**: 综合分析器可以同时处理 Python 和 C++ 混合堆栈
6. **符号反混淆**: 自动使用 `c++filt` 反混淆 C++ 符号
7. **Debug版本**: ⭐ 新功能 - 源码行定位需要Debug版本的PyPTO包
8. **编译时间**: ⭐ 新功能 - Debug版本编译可能需要较长时间（建议10-15分钟）

---

## 常见问题处理

### 问题 1：未找到二进制文件
**原因**: 二进制文件不在默认搜索路径中
**解决方法**:
- 综合分析器会自动在多个路径中搜索
- 如果仍未找到，可以手动指定搜索路径
- 确认二进制文件名正确

### 问题 2：地址解析失败
**原因**: 地址是运行时地址，不是编译时地址
**解决方法**:
- 对于 PyPTO C++ 堆栈，使用偏移量而不是绝对地址
- 综合分析器会自动处理这种情况

### 问题 3：符号解析失败
**原因**: 二进制文件未包含符号表
**解决方法**:
- 确认二进制文件未经过 strip
- 使用包含调试信息的二进制文件
- 综合分析器会自动使用符号反混淆

### 问题 4：工具未找到
**原因**: 系统未安装必需的工具
**解决方法**:
- 安装 binutils: `sudo apt-get install binutils`
- 或安装 LLVM 工具: `sudo apt-get install llvm`
- c++filt 通常包含在 binutils 中

### 问题 5：无法定位源码行 ⭐ 新问题
**原因**: 二进制文件不包含调试信息
**解决方法**:
- 编译Debug版本PyPTO: `python3 build_ci.py -f python3 --build_type=Debug`
- 安装Debug版本: `pip install build_out/pypto*.whl --force-reinstall --no-deps`
- 重新运行分析

### 问题 6：Debug版本编译失败 ⭐ 新问题
**原因**: 编译环境或依赖问题
**解决方法**:
- 检查cmake是否安装: `which cmake`
- 检查Python版本: `python3 --version`
- 查看编译日志: `tail -100 /tmp/pypto_debug_build.log`
- 增加编译超时时间: `-t 1800` (30分钟)

---

## 使用示例

### 示例 1：分析PyPTO编译错误（包含源码行定位）⭐ 推荐
```bash
# 步骤 1：检查调试符号
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py --check-debug

# 步骤 2：如果没有调试符号，编译Debug版本
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/build_debug_pypto.py

# 步骤 3：安装Debug版本
pip install build_out/pypto*.whl --force-reinstall --no-deps

# 步骤 4：分析堆栈（现在会显示源码行号）
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/comprehensive_analyzer.py <stack_file> -f
```

### 示例 2：分析测试用例堆栈（包含重新运行）⭐ 新增
```bash
# 步骤 1：检查调试符号
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py --check-debug

# 步骤 2：如果没有调试符号，编译Debug版本
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/build_debug_pypto.py

# 步骤 3：安装Debug版本
pip install build_out/pypto*.whl --force-reinstall --no-deps

# 步骤 4：重新运行测试用例（⭐ 新增）
python3 /path/to/test_case.py --run_mode sim 2>&1 | tee /tmp/new_stack_trace.log

# 步骤 5：从新输出中提取堆栈信息
grep -A 50 "Run pass failed" /tmp/new_stack_trace.log > /tmp/new_stack_trace.txt

# 步骤 6：分析新堆栈信息（现在会显示源码行号）
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/comprehensive_analyzer.py /tmp/new_stack_trace.txt -f
```

### 示例 2：分析Python traceback
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/comprehensive_analyzer.py /tmp/python_traceback.txt -f
```

### 示例 3：分析C++ stack trace
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/comprehensive_analyzer.py /tmp/cpp_stack.txt -f
```

### 示例 4：仅定位特定地址 ⭐ 新功能
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py -a 0x19d600a
```

### 示例 5：仅定位特定符号 ⭐ 新功能
```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py -s "npu::tile_fwk::Pad(long, long)"
```

---

## 优化说明

相比初始版本，优化后的 skill 具有以下改进：

1. **新增 PyPTO C++ 格式支持**
   - 可以解析 `libtile.so(function+offset) [address]` 格式
   - 这是 PyPTO 常见的 C++ 堆栈格式

2. **综合分析器**
   - 一次性完成所有分析步骤
   - 自动处理 Python 和 C++ 混合堆栈
   - 自动查找二进制文件
   - 自动符号反混淆

3. **错误信息提取**
   - 自动提取 Errcode
   - 自动提取错误位置
   - 自动提取错误消息

4. **符号反混淆**
   - 使用 `c++filt` 自动反混淆 C++ 符号
   - 显示原始符号和反混淆后的符号

5. **改进的输出格式**
   - 更清晰的报告结构
   - 标记关键帧（错误发生点/触发点）
   - 显示二进制文件路径

6. **JSON 输出支持**
   - 所有脚本都支持 JSON 输出
   - 便于程序化处理

7. **编译Debug版本PyPTO** ⭐ 新功能
   - 自动编译Debug版本
   - 检查编译前提条件
   - 支持自定义超时时间

8. **源码行定位** ⭐ 新功能
   - 使用 addr2line 定位地址到源码行
   - 支持地址定位和符号定位
   - 仅在Debug版本中可用

---

## 新增功能总结 ⭐

| 功能 | 说明 | 脚本 |
|------|------|------|
| 编译Debug版本PyPTO | 自动编译Debug版本Py | build_debug_pypto.py |
| 源码行定位 | 定位地址/符号到源码行 | locate_source_code.py |
| 调试符号检查 | 检查是否有调试符号 | locate_source_code.py --check-debug |

---

## 完整工作流程示例 ⭐

```bash
# 假设有一个浮点异常堆栈文件 float_exception.md

# 步骤 1：检查调试符号
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py --check-debug
# 输出：✗ 二进制文件不包含调试符号

# 步骤 2：编译Debug版本（可能需要10-15分钟）
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/build_debug_pypto.py
# 输出：✓ Debug版本编译成功

# 步骤 3：安装Debug版本
pip install build_out/pypto*.whl --force-reinstall --no-deps
# 输出：Successfully installed pypto-0.2.1

# 步骤 4：分析堆栈（现在会显示源码行号）
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/comprehensive_analyzer.py float_exception.md -f
# 输出：
#   帧 #0 ⚠️ 错误发生点
#   - 函数: npu::tile_fwk::Pad(long, long)
#   - 源码: /path/to/pad_local_buffer.cpp:49  ← 新增！
#   - 问题代码: return (dim + padValue - 1) / padValue * padValue;
```

---

**Skill 优化完成！** 🎉

现在的 `pypto-stack-trace-analyzer` skill 已经能够：
- ✅ 分析各种格式的堆栈信息
- ✅ 自动编译Debug版本PyPTO
- ✅ 定位到具体的源码行号
- ✅ 提供完整的错误分析报告
