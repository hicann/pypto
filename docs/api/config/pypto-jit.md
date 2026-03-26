# pypto.jit

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

jit 函数是一个用于装饰和优化动态函数的工具，它通过即时编译技术将 Python 函数转换为高效的计算图。jit 函数可以接受一个函数以及多个可选的配置参数，包括 codegen\_options、host\_options、pass\_options 和 runtime\_options。当 jit 被调用时，它会自动管理函数的编译和执行过程。在首次运行时，jit 会编译函数并生成执行计划，后续调用会复用编译结果以提高性能。此外，jit 还支持缓存机制，根据输入输出Tensor的形状判断是否需要重新编译，从而优化计算效率。

## 函数原型

```python
def jit(dyn_func=None,
        *,
        codegen_options=None,
        host_options=None,
        pass_options=None,
        runtime_options=None)
```

## 参数说明

### API 参数说明

| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| dyn_func          | 输入      | jit修饰的函数，需要传入pypto.Tensor作为平铺参数，用于后续构建计算图。 |
| codegen_options   | 输入      | 类型为dict[str, any]，用于设置codegen配置项，配置项参数见[参数说明](pypto-set_codegen_options.md) |
| host_options      | 输入      | 类型为dict[str, any]，用于设置host配置项，配置项参数见[参数说明](pypto-set_host_options.md) |
| pass_options      | 输入      | 类型为dict[str, any]，用于设置Pass配置项，配置项参数见[参数说明](pypto-set_pass_options.md) |
| runtime_options   | 输入      | 类型为dict[str, any]，用于设置runtime配置项，配置项参数见[runtime_options 参数说明](#runtime_options_detail) |

### runtime_options 参数说明 <a id="runtime_options_detail"></a>

| 参数名                         | 说明                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| device_sched_mode               | 含义：设置计算子图的调度模式 <br> 说明：0：代表默认调度模式，ready子图放入共享队列，各个调度线程抢占子图进行发送，子图获取发送遵循先入先出； <br> 1：代表L2cache亲和调度模式，选择最新依赖ready的子图优先下发，达到复用L2cache的效果； <br> 2：公平调度模式，aicpu上多线程调度管理多个aicore的时候，下发子图会尽量控制在多线程间的公平性，此模式会带来额外的调度管理开销； <br> 3：代表同时开启L2cache亲和调度模式以及公平调度模式； <br> 类型：int <br> 取值范围：0 或 1 或 2 或 3 <br> 默认值：0 <br> 影响pass范围：NA |
| stitch_function_max_num        | 含义：machine运行时ctrlflow aicpu里控制每次提交给schedule aicpu处理的最大device task的计算任务量 <br> 说明：设置的值代表每一个stitch task里处理的最大loop个数，该数值越大，通常stitch batch内并行度越高，相应的workspace内存使用也越大。<br> 注意：此项配置会替代掉stitch_function_inner_memory、stitch_function_outcast_memory和stitch_function_num_initial三项。 <br> 类型：int <br> 取值范围:1 ~ 1024 <br> 默认值：128 <br> 影响pass范围：NA |
| run_mode                       | 含义：设置计算子图的执行设备 <br> 说明：<br> 0：表示在NPU上执行 <br> 1：表示在模拟器上执行 <br> 类型：int <br> 取值范围：0或者1 <br> 默认值：根据是否设置cann的环境变量来决定。如果设置了环境变量，则在NPU上执行；否则在模拟器上执行 <br> 影响pass范围：NA |
| valid_shape_optimize            | 含义：动态shape场景，validshape编译优化选项，打开该选项后，动态轴的Loop循环中，主块（shape与validshape相等）采用静态shape编译，尾块采用动态shape编译 <br> 说明：<br> 0：默认值，表示关闭validshape编译优化选项，所有Loop循环均采用动态shape进行编译 <br> 1：表示打开validshape编译优化选项 <br> 类型：int <br> 取值范围：0或者1 <br> 默认值：0 <br> 影响pass范围：NA |
| ready_on_host_tensors           | 含义：标记在Host端准备好的Kernel入口函数的输入tensor名称列表，格式为["tensor1", "tensor2", ...]。<br> 说明：如果算子的计算逻辑对某输入tensor有值依赖(即获取了tensor的值)，且此tensor的device数据在Host端已提前准备好，那么cpu的控制流可以提前发射以提升性能。<br> 类型：list of string <br> 默认值：空列表 <br> 影响pass范围：NA |

## 返回值说明

无

## 约束说明

修饰的函数传入的计算参数需为pypto.Tensor类型。

## 调用示例

无参数装饰

```python
@pypto.jit
def func(tensor1, tensor2, tensor3):
...
```

带配置装饰

```python
@pypto.jit(
    codegen_options={"support_dynamic_aligned": True}
)
def func(tensor1, tensor2, tensor3):
...
```

