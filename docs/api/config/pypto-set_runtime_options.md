# pypto.set\_runtime\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

设置runtime的选项。

## 函数原型

```python
set_runtime_options(*,
                    device_sched_mode : int = None,
                    stitch_function_inner_memory: int = None,
                    stitch_function_outcast_memory : int = None,
                    stitch_function_num_initial : int = None,
                    stitch_function_num_step : int = None,
                    stitch_function_size : int = None,
                    stitch_cfgcache_size: int = None,
                    run_mode : int = None,
                    ) -> None
```

## 参数说明


| 参数名                         | 输入/输出 | 说明                                                         |
| ------------------------------ | --------- | ------------------------------------------------------------ |
| device_sched_mode              | 输入      | 含义：设置计算子图的调度模式 <br> 说明：0：代表默认调度模式，ready子图放入共享队列，各个调度线程抢占子图进行发送，子图获取发送遵循先入先出； <br> 1：代表L2cache亲和调度模式，选择最新依赖ready的子图优先下发，达到复用L2cache的效果； <br> 2：公平调度模式，aicpu上多线程调度管理多个aicore的时候，下发子图会尽量控制在多线程间的公平性，此模式会带来额外的调度管理开销； <br> 3：代表同时开启L2cache亲和调度模式以及公平调度模式； <br> 类型：int <br> 取值范围：0 或 1 或 2 或 3 <br> 默认值：0 <br> 影响pass范围：NA |
| stitch_function_inner_memory   | 输入      | 含义：控制root function中间计算结果的内存池大小的参数，内存池大小为max_root_nonoutcast_workspace（单个rootfunction的最大非outcast内存） *STITCH_FUNCTION_INNER_MEMORY。 <br> 说明：该数值越小，root function间越容易因workspace重叠导致互相产生依赖，导致无法并行；反之，该数值越大，通常stitch batch内并行度越高。 <br> 类型：int <br> 取值范围：1~2147483647，当前版本最大有效值是 128 * max_unroll_times，max_unroll_times 是算子代码里最大多分档档位。 <br> 默认值：10 <br> 影响pass范围：NA |
| stitch_function_outcast_memory | 输入      | 含义：控制stitch构建的devicetask中间计算结果（devitask内部rootfunction的outcast）的内存池大小的参数，内存池大小为maxOutcastWorkspace*stitch_function_outcast_memory <br> 说明：设置的值代表该workspace允许将多少loop的计算图动态的stitch到一起并行下发处理，设置的值越大代表评估使用的workspace内存越大 <br> 类型：int <br> 取值范围:1 ~ 2147483647，当前版本最大有效值是 128 * max_unroll_times，max_unroll_times 是算子代码里最大多分档档位。 <br> 默认值：50 <br> 影响pass范围：NA |
| stitch_function_num_initial    | 输入      | 含义：machine运行时ctrlflow aicpu里控制首个提交给schedule aicpu处理的device task的计算任务量 <br> 说明：设置的值代表第一个stitch task里处理的loop个数，通过此值来控制device machine启动头开销的大小，让ctrlflow aicpu和schedule aicpu计算尽快overlap起来 <br> 类型：int <br> 取值范围:1 ~ 128 <br> 默认值：128 <br> 影响pass范围：NA |
| stitch_function_num_step       | 输入      | 含义：machine运行时ctrlflow aicpu里控制非首次device task的计算任务量 <br> 说明：为了后续stitch task处理计算量平滑增加，可以通过设置此配置项进行控制。如设置为n，则每次stitch task里处理的loop次数分别base+n， base+2n 。。。 <br> 类型：int <br> 取值范围:0 ~ 128 <br> 默认值：0 <br> 影响pass范围：NA |
| stitch_function_size           | 输入      | 含义：machine运行时ctrlflow aicpu里控制stitch生成的device task处理最大Callop计算量 <br> 说明：为了保障stitch task处理单次loop时的性能，需通过设置该配置项进行控制，该配置项设置的过大会带来额外的性能和内存开销，需根据算子最大Callop数量调整该配置项。若Callop数量超过该配置会报错提示：ASSERT FAILED：CallOpSize&lt;=CallOpmaxSize."loopFunction:&lt;function name&gt; ,CallopSize:&lt;当前Callop数量&gt;，CallOpmaxSize：&lt;配置项大小&gt;" <br> 类型：int <br> 取值范围:1 ~ 65535 <br> 默认值：20000 <br> 影响pass范围：NA |
| stitch_cfgcache_size           | 输入      | 含义：指定生成控制流缓存的大小，单位是字节 <br>说明：如果该值是0，则表示不使能控制流缓存。由于控制流缓存是按照任务大小来缓存，如果设置比较小，例如小于一个任务，那么无法缓存。<br>类型：int<br>取值范围：0~100000000<br>默认值：0<br>影响pass范围：NA |
| run_mode                       | 输入      | 含义：设置计算子图的执行设备 <br> 说明：<br> 0：表示在NPU上执行 <br> 1：表示在模拟器上执行 <br> 类型：int <br> 取值范围：0或者1 <br> 默认值：根据是否设置cann的环境变量来决定。如果设置了环境变量，则在NPU上执行；否则在模拟器上执行 <br> 影响pass范围：NA |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

无。

## 调用示例

```python
pypto.set_runtime_options(device_sched_mode=1,
                          stitch_function_inner_memory=128,
                          stitch_function_outcast_memory=128,
                          stitch_function_num_initial=128,
                          stitch_function_num_step=20)
@pypto.jit(
 	     runtime_options={
 	     "stitch_function_inner_memory": 128,
 	     "stitch_function_outcast_memory": 128,
 	     "stitch_function_num_initial": 128,
 	     "device_sched_mode": 1
 	     }
)
```

