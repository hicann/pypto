# 异步执行

当开发者完成一个Kernel函数的编写后，通过`@pl.jit()`装饰器完成JIT编译，并将Kernel提交到NPU上执行。PyPTO Pro的Kernel编程模型采用SPMD（Single Program Multiple Data），多个AI Core执行同一份Kernel程序，并通过核索引处理不同的数据分片。Kernel下发相对于Host异步执行。

完成Kernel加载和运行的主要流程分以下几步：

1. **环境初始化**：配置CANN环境变量，设置设备ID。
2. **Kernel函数定义**：通过`@pl.jit()`装饰器定义Kernel函数，配置Tile、数据搬运和计算逻辑。
3. **数据准备**：通过PyTorch创建输入、输出张量，并将张量放到目标NPU设备。
4. **异步执行**：调用Kernel函数，PyPTO Pro自动完成JIT编译和NPU任务下发。相同编译签名的后续调用复用已编译结果；编译签名发生变化时，可能触发新的编译。
5. **结果获取**：Host在读取结果、进行精度比较或计时时，应先调用`torch.npu.synchronize()`，或者同步本次下发所使用的Stream。

在这个过程中，PyPTO Pro框架自动处理编译缓存和Kernel下发。本节说明SPMD执行模型、Stream选择和同步方式。

## SPMD执行模型

PyPTO Pro的Kernel编程模型采用SPMD（Single Program Multiple Data），多个AI Core执行同一份Kernel程序，并处理不同的数据分片。

- `pl.get_block_num()`获取本次Kernel启动的核数。
- `pl.get_block_idx()`获取当前核的索引。
- Kernel应根据核索引显式划分任务；框架不会自动把循环迭代分配给不同的核。

SPMD执行模型的优势包括：

- **显式并行**：开发者可以根据数据规模选择分块方式和启动核数。
- **独立分片**：各AI Core处理不同数据分片，减少不必要的跨核同步。
- **统一程序**：所有核复用同一份Kernel逻辑，通过核索引区分任务。

## 架构指定

通过`@pl.jit(arch=...)`参数可以指定目标NPU架构，框架会自动适配对应的硬件特性。架构指定的详细用法请参考[JIT编译](operator_compilation/JIT_compilation.md#架构指定)。

## Stream选择与同步

不指定Stream时，Kernel下发到PyTorch NPU当前Stream。方括号启动语法为`kernel[stream, block_dim](...)`，其中`None`表示使用当前Stream，`block_dim`表示启动核数。

```python
# 使用当前Stream，启动num_cores个AI Core。
kernel[None, num_cores](input_tensor, output_tensor)

# Kernel下发相对于Host异步；Host读取结果前需要同步。
torch.npu.synchronize()
```

也可以创建并显式传入Stream：

```python
stream = torch.npu.Stream()
kernel[stream, num_cores](input_tensor, output_tensor)

# 只等待该Stream上的任务完成。
stream.synchronize()
```

同一Stream内的任务按下发顺序执行；使用不同Stream时，若任务之间存在数据依赖，需要通过PyTorch NPU的Stream同步机制建立依赖。
