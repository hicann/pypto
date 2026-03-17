# MACHINE 组件错误码

- **范围**：F7-F8XXXX
- 本文档说明 MACHINE 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义与使用说明

相关错误码的统一定义，参见 `framework/src/machine/utils/machine_error.h` 文件。
---

## 排查建议

### 怀疑和MACHINE内存处理有关的精度问题

1. **检查输入初始化**：
保证输入/输出初始化

2. **检查 Tensor 连续性**：
部分 MACHINE 相关算子或接口要求输入/输出在指定内存布局下**连续**（`tensor.is_contiguous()` 为 True）。非连续 tensor（如某些 view、transpose、slice 结果）可能导致精度异常或运行错误。
eg：reshape/view、交换维度、索引/切片后的 tensor 可能非连续；若前序是 COPY_IN 或尾轴 reduce，需在前端或调用侧保证传入的 tensor 在对应格式下连续。

3. **扩大 workspace 大小**:
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
（1）打开VERBOSE日志
`framework/src/machine/utils/device_switch.h`
```cpp
#define ENABLE_COMPILE_VERBOSE_LOG 1
```

（2）打开DEBUG日志，指定日志落盘路径
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=./my_log
```

（3）重新编pypto whl包并安装

（4）启动性能数据采集功能，保证生成 dyn_topo.txt

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

如无异常，提示device task 无内存重叠；
如存在异常，提示内存重叠的 device task 以及 leaf function。

注：
（1）如果报错 emory reuse must happen for full match. 则两个需要内存复用的 rawtensor 范围不一致；
（2）如果报错 memory reuse must happen for same dimension. 则两个内存复用的 rawtensor 的 shape 不一致；
上述两种情况非内存重叠，脚本内存检查依赖不会发生上述情况 ，因此脚本会断言，直接提示日志信息错误。

6. **复杂特性排除**：
关闭unroll_list、合轴特性、配置submit_before_loop=True使loop串行执行、确定valid_shape配置正确性、+0.0等，缩小定位范围


### F70006 HANDSHAKE_TIMEOUT

1. **确认设备与驱动**：NPU 设备可用、驱动正常，`npu-smi info` 无异常。
2. **确认资源与负载**：当前进程/容器内 NPU 占用是否过高，是否存在多进程争用同一设备。
3. **确认超时配置**：若存在握手/同步超时配置项，检查是否过短或与环境不符。
4. **查日志上下文**：结合同线程前后日志（如 “Schedule run init succ” 之后、AbnormalStop 相关）确认是首次握手失败还是运行中异常。

**关联 Skill**：[pypto-environment-setup](../../.opencode/skills/pypto-environment-setup/SKILL.md)（环境与 NPU 设备诊断、`npu-smi`、驱动与编译运行）