# JIT编译

使用`@pypto_pro.language.jit()`声明的Kernel在首次启动时完成解析、代码生成和编译，不需要单独执行编译命令。JIT适合Kernel开发、功能验证和性能调试。

## JIT编译流程

首次启动Kernel时依次执行以下步骤：

1. 绑定Kernel实参和启动配置，确定动态参数与编译期特化信息。
2. 解析Kernel函数体，生成PyPTO IR。
3. 对IR执行Pass优化和校验。
4. 生成Device代码和Host侧Launcher。
5. 编译并加载产物，然后向指定Stream下发Kernel。

编译和执行都由一次Kernel调用触发。Kernel启动相对于Host异步，但首次调用会先等待当前编译实例生成完成。

## 触发JIT编译

Kernel首次通过`kernel[stream, block_dim](...)`启动时触发JIT编译。以下以已定义的`add_kernel`为例，展示首次启动和同一编译签名下的复用：

```python
import torch
import torch_npu


x = torch.rand(64, 64, device="npu:0", dtype=torch.float16)
y = torch.rand_like(x)
out = torch.empty_like(x)

# 首次启动：编译并执行。
add_kernel[None, 1](x, y, out)
torch.npu.synchronize()

# 编译签名相同时复用当前进程中的编译结果。
add_kernel[None, 1](x, y, out)
```

Kernel的定义和启动语法参考[Kernel核函数创建](../kernel_function.md)。

## 编译签名与复用

同一Kernel对象按照编译签名区分编译实例。以下信息可能产生不同实例：

| 信息 | 对编译实例的影响 |
| --- | --- |
| Tensor固定维度 | 声明为固定值的维度必须匹配；不同静态签名使用不同实例。 |
| `pypto_pro.language.STATIC`维度 | 运行时取值参与特化，值变化时生成新实例。 |
| `pypto_pro.language.DYNAMIC`维度 | 维度值不参与特化，值变化时复用实例。 |
| TilingKey | 每个合法Key对应一个专用实例。 |
| datatype | 每组数据类型组合对应一个专用实例。 |
| 编译目标 | 目标在Kernel对象创建时确定；不同目标使用不同的Kernel对象。 |

TilingData字段是运行时数据，字段值变化不会单独产生编译实例。静态与动态shape的声明方式参考[Tensor创建和操作](../tensor_creation_and_operations.md)，TilingData和TilingKey的区别参考[Tiling结果传输](../tiling/tiling_result_transfer.md)。

`stream`和`block_dim`只影响本次启动，不参与编译签名；调整Stream或逻辑Block数不会因此生成新的编译实例。

`pypto.options(...)`中的编译配置也不参与编译签名。配置只在某个签名首次编译时读取；命中当前Kernel对象的已有编译实例后，改变配置不会触发重新编译。需要让新配置生效时，应创建新的Kernel对象或重新启动Python进程。

JIT复用范围限于当前Python进程。重新启动进程后会重新执行生成和编译流程；`build`目录中的文件用于加载和调试，不作为跨进程持久化缓存。

## jit装饰器配置

`@pypto_pro.language.jit()`配置当前Kernel的编译行为：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `auto_mutex` | 是否根据TileGroup的mutex元数据自动处理可识别的数据依赖。 | `True` |
| `compile_timeout` | 当前Kernel的编译超时时间，单位为秒。 | `None` |
| `name` | 自定义Kernel名称，用于区分编译产物。 | `None` |
| `tiling_key` | 绑定TilingKey Schema。 | `None` |
| `datatype` | 声明参与数据类型特化的Kernel参数。 | `None` |
| `pipeline` | 配置自动流水变换。 | `None` |
| `arch` | 指定编译目标；通常省略并由运行环境自动确定。 | `None` |

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True, compile_timeout=1200, name="add_kernel")
def add_kernel(x, y, out):
    ...
```

TilingKey和datatype的定义及启动参数位置参考[Kernel核函数创建](../kernel_function.md#使用tilingkey和datatype)。自动流水配置参考[自动并行流水](../../advanced_programming/auto_parallel_pipeline.md)。

## 编译配置

Host、Pass、CodeGen、验证和调试等共享配置通过`pypto.options(...)`作用于当前编译作用域：

```python
import pypto


with pypto.options(
    host_options={"compile_timeout": 1200},
    pass_options={"enable_slice": False},
):
    add_kernel[None, 1](x, y, out)
```

也可以分别使用`pypto.set_host_options()`、`pypto.set_pass_options()`、`pypto.set_codegen_options()`、`pypto.set_verify_options()`和`pypto.set_debug_options()`。主要配置分类如下：

| 分类 | `pypto.options(...)`参数 | 用途 |
| --- | --- | --- |
| Host编译控制 | `host_options` | 编译阶段、编译监控和超时。 |
| Pass控制 | `pass_options` | 流水、Buffer复用、调度和切分等Pass选项。 |
| CodeGen控制 | `codegen_options` | 代码生成、PMU和VF相关选项。 |
| Pass验证 | `verify_options` | Pass结果校验和中间Tensor保存。 |
| 调试 | `debug_options` | 编译、运行时和Pass图调试。 |
| 运行时 | `runtime_options` | 调度、Workspace和运行模式。 |
| 算子行为 | `operation_options` | 算子级行为配置。 |
| Tile与矩阵规格 | `vec_tile_shapes`、`cube_tile_shapes`等 | 设置当前编译作用域使用的Tile和矩阵规格。 |

配置项的类型和取值范围参考[`pypto.set_host_options`](../../../../../api/tensor_api/config/pypto-set_host_options.md)、[`pypto.set_pass_options`](../../../../../api/tensor_api/config/pypto-set_pass_options.md)、[`pypto.set_codegen_options`](../../../../../api/tensor_api/config/pypto-set_codegen_options.md)、[`pypto.set_verify_options`](../../../../../api/tensor_api/config/pypto-set_verify_options.md)和[`pypto.set_debug_options`](../../../../../api/tensor_api/config/pypto-set_debug_options.md)。这些配置字典传给`pypto.options(...)`，不能作为`pypto_pro.language.jit`的参数。

### 编译超时配置

`compile_timeout`按照以下优先级确定：

1. `@pypto_pro.language.jit(compile_timeout=...)`中显式设置的值。
2. 当前`pypto.options()`作用域中的`host_options["compile_timeout"]`。
3. 框架默认值600秒。

编译监控的开关、总耗时阈值和阶段阈值通过`host_options`配置。

## Kernel下发与同步

编译完成后，JIT通过Host侧Launcher将Kernel提交到指定Stream。`kernel[None, block_dim](...)`使用当前Stream，`kernel[stream, block_dim](...)`使用显式Stream。

Kernel下发是异步操作。在读取输出、检查精度或统计完整执行时间前，同步相应Stream：

```python
import torch
import torch_npu


stream = torch.npu.Stream()
add_kernel[stream, num_cores](x, y, out)
stream.synchronize()
```

Stream和`block_dim`的完整说明参考[Kernel核函数创建](../kernel_function.md#调用kernel)。

## 编译产物

未设置`ASCEND_WORK_PATH`时，JIT产物以`./build/`为根目录；设置后，以`${ASCEND_WORK_PATH}/PYPTO_PRO/build/`为根目录。Kernel目录名称以`{kernel_name}__{arch}`开头，并可能包含静态签名、Device和Rank等后缀；datatype和TilingKey实例还会分别使用`dt_{hash}`和`tk_{packed}`（未使用TilingKey时为`tk_none`）子目录。主要文件包括：

| 文件 | 作用 |
| --- | --- |
| `kernel.cpp` | CodeGen生成的Device侧源码。 |
| `call_kernel.cpp` | Host侧Launcher源码，负责参数打包和Kernel下发。 |
| `call_kernel_{hash}.so` | 编译后的Launcher共享库。 |
| `*_tiling.h` | 使用TilingData时生成的结构体头文件。 |

编译失败时，优先结合错误日志和`kernel.cpp`定位解析、代码生成或工具链问题。编译成功后，中间源码默认保留在产物目录中，可用于核对生成代码。
