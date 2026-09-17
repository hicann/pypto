# 功能调试

PyPTO Pro Kernel出现功能问题时，先判断问题发生在编译阶段还是执行阶段，再选择对应的定位手段。对于设备侧AIC Error，可采集异常现场并使用离线复现工具定位到源码行。

## 编译与执行流程

被`@pypto_pro.language.jit()`装饰的Kernel在首次调用时依次完成参数绑定、前端解析与校验、代码生成、编译和Kernel下发。定位问题时，应先根据报错位置和调用栈判断失败环节：

- **前端解析与编译阶段**：参数绑定、Python前端解析或IR校验失败时，优先查看异常中的用户源码位置和错误描述，检查接口参数、Shape、数据类型、内存空间和编译期约束；代码生成或工具链编译失败时，还需结合编译日志和生成代码定位。
- **下发与执行阶段**：Kernel下发是异步操作，应在启动后同步Stream，使设备侧异常在当前调用点暴露，并保留完整的报错信息和运行日志。

```python
kernel[None, block_dim](*args)
torch.npu.synchronize()
```

出现异常时，可结合报错信息中的错误码快速判断问题所属阶段和组件，并选择相应的定位手段。错误码的分类、含义和处理建议将在后续版本补充。

## 常用定位手段

### 使用pypto_pro.language.printf打印信息

[`pypto_pro.language.printf`](../../../../api/pro_api/Utils-API/debugging/printf.md)用于在Kernel中打印Core编号、循环变量、偏移、Shape和分支标志等标量信息，可辅助确认多核切分、循环边界和运行时参数是否符合预期。多核场景建议通过`pypto_pro.language.get_block_idx()`限制打印的Core，避免日志大量交错。该接口会引入运行时开销，仅用于功能调试，不应用于性能测试。

### 检测内存问题

Kernel的内存访问错误（如GM访问越界、Tile访问越界、Tile内存区间重叠、mutex未正确配对）通常不直接报错，而是表现为结果错乱或执行挂起。可使用[内存检测工具](sanitizer.md)进行定位：在核函数装饰器上设置`@pl.jit(sanitizer=True)`即可自动覆盖Kernel内的全部访问点，检测结果写入报告文件。内存检测会产生额外的运行时开销，仅用于功能调试。

### 定位精度问题

当Kernel能够正常执行但结果与参考实现不一致时，可使用[pypto-pro-precision-debug Skill](https://gitcode.com/cann/pypto-gym/blob/master/cannbot-skills/ops/pypto-pro-precision-debug/SKILL.md)进行定位。该Skill提供较为详细的精度定位流程，并汇总了精度调试中的常见问题及处理方法，建议优先参考。

### 查看编译产物

JIT编译产物默认位于当前工作目录下的`build`目录；设置`ASCEND_WORK_PATH`后，位于`${ASCEND_WORK_PATH}/PYPTO_PRO/build`目录。可重点查看以下文件：

| 文件 | 用途 |
| --- | --- |
| `kernel.cpp` | 编译器生成的设备侧Kernel源码，可用于检查参数类型、控制流、地址计算和数据搬运。 |
| `pipeline_generated.py` | 启用[自动CV并行流水](../advanced_programming/auto_parallel_pipeline.md)后生成的自动核间流水排布代码，可用于检查stage排布、preload和核间同步。 |

以上文件均为编译器生成的诊断产物。定位到问题后，应修改原始Python Kernel并重新编译，不要直接修改生成文件。

## AIC Error信息采集

Kernel在设备上触发越界读写、死锁超时等AIC Error后，可通过异常dump和离线复现工具还原现场，并将Error PC定位到Kernel源码行。

### 采集前配置

执行用例前设置一个可写的工作目录：

```bash
export ASCEND_WORK_PATH=./wk
```

采集前请确保未设置`NPU_COLLECT_PATH`；如果已经设置，可执行`unset NPU_COLLECT_PATH`将其取消，否则无法生成离线复现所需的数据文件。建议在独立子进程中运行问题用例，以确保异常数据完整生成。

异常发生后，相关文件保存在以下目录：

```text
${ASCEND_WORK_PATH}/extra-info/data-dump/<device_id>/
```

主要包括：

| 产物 | 用途 |
| --- | --- |
| dump数据文件 | 保存输入、输出和workspace Tensor的Shape、数据类型及原始数据。 |
| `<kernel>_launch_args.json` | 保存Kernel名称、`block_dim`以及完整的指针和标量启动参数。 |
| `<kernel>_call_kernel.so` | 带调试信息的Kernel执行体，用于离线复现和源码行定位。 |
| `call_kernel_<digest>.so` | 原始JIT产物副本，用于比对和高保真复现。 |
| `kernel.cpp`、`call_kernel.cpp`和相关头文件 | 异常Kernel对应的源码副本。 |

### 离线复现与定位

使用以下命令解析dump并复现异常：

```bash
python tools/scripts/debug_aicore_error_pro_repro.py \
    -p <dump_dir> \
    [-d <device_id>] \
    [-out <output_dir>] \
    [-t <seconds>]
```

其中，`-p`指向`extra-info/data-dump/<device_id>`目录；`-d`用于覆盖从目录名解析出的Device ID；`-out`用于指定报告输出目录；`-t`用于设置单次复现的超时时间，默认值为600秒。

工具会恢复Tensor数据和Kernel启动参数，生成并执行单算子复现脚本，再从本次运行的plog中提取Error PC。主要输出包括：

- `test_single_op.py`：可独立执行的单算子复现脚本。
- `reproduction_report.txt`：复现结果、出错Core、Error PC、符号、源码行、inline调用链和源码上下文。
- `reproduction_report--singlecommit.txt`：超时类故障的singlecommit诊断结果。仅超时类故障生成。

对于超时类故障，工具会自动在singlecommit模式下重跑：若singlecommit下执行成功，说明问题与核间并发或同步时序有关；若仍然失败，则应结合报告中的卡死Core和源码位置继续排查。

若报告提示`Error PC not found in plog`，应检查CANN是否生成了symbol locator日志及日志目录配置。若报告提示源码行无法解析，应检查`ASCEND_HOME_PATH`和BiSheng工具链是否可用。旧dump缺少`<kernel>_launch_args.json`时，工具会将`block_dim`回退为`1`，依赖多核并发的问题可能无法按原现场复现。
