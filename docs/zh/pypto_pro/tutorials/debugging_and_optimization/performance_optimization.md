# 性能调优

PyPTO Pro的性能调优对象是直接运行在AI Core上的Kernel。调优时应先建立可复现的性能基线，再通过Profiling确定瓶颈，最后针对多核切分、数据搬运、片上内存、计算流水和编译期特化逐项优化。

性能优化必须以结果正确为前提。每次修改Tile Shape、流水、同步或多核分片后，都应先完成精度回归，再比较性能变化。

## 调优流程

建议按照以下顺序开展调优：

1. 固定目标Shape、数据类型、TilingKey、`block_dim`、输入数据和执行Stream。
2. 完成JIT预热，建立稳定、可重复的Kernel耗时基线。
3. 使用Profiling数据判断瓶颈属于计算、访存、流水等待、多核负载不均还是启动开销。
4. 每次只调整一类关键因素，并记录修改前后的精度、耗时和资源占用。
5. 对全部目标Shape、数据类型和TilingKey回归，防止局部优化造成其他场景退化。
6. 在算子实际调用链路中复测端到端性能。

## 建立性能基线

### 排除JIT编译开销

Kernel首次调用可能包含JIT编译、Host侧共享库生成和加载开销，不能计入Kernel执行时间。不同静态Shape、TilingKey或数据类型可能对应不同的编译实例，测量每种配置前均应单独预热。

### 正确处理异步执行

Kernel启动是异步操作。计时前后必须执行同步，否则测得的可能只是Host侧任务下发时间。以下代码给出基本测量方式：

```python
import time
import torch


warmup = 10
repeat = 100

# 输入分配、初始化和参考结果计算放在计时区间之外
for _ in range(warmup):
    kernel[None, block_dim](*args)
torch.npu.synchronize()

torch.npu.synchronize()
start = time.perf_counter()
for _ in range(repeat):
    kernel[None, block_dim](*args)
torch.npu.synchronize()
elapsed = time.perf_counter() - start

avg_call_us = elapsed * 1e6 / repeat
print(f"average JIT call time: {avg_call_us:.3f} us")
```

为了获得可比较的结果，应满足以下条件：

- 输入分配、随机数生成、Host到Device拷贝和结果校验不放入Kernel计时区间。
- 测试期间保持Shape、数据类型、TilingKey、`block_dim`和Stream不变。
- 预热次数应足以完成编译并使运行状态稳定。
- 重复多轮测量并报告中位数或稳定区间，不以单次结果判断优化效果。
- 调优版本和基线版本使用相同的同步位置及测量方法。
- 测量前移除`pl.printf`、`pl.dump_data`、`pl.pto_assert`和`pl.trap`等调试代码。

上述方式测量的是连续下发场景下JIT调用链路的稳态平均耗时，包含Host侧逐次下发开销，不等同于Profiling结果中的纯Device Kernel执行时间。需要测量单次同步调用延迟时，应在每次调用后同步，并将同步方式和固定开销纳入基线。算子通过二进制包和aclnn接口交付时，还应在实际aclnn调用路径中测量端到端耗时，以覆盖Host侧Tiling、参数校验、Kernel选择和任务下发开销。

## 使用Profiling定位瓶颈

可以使用NPU Profiling工具采集运行数据。例如，对可独立运行的测试脚本执行：

```bash
msprof python3 test_kernel.py
```

采集前应先单独运行脚本完成正确性验证，并保证脚本包含预热和足够次数的稳定执行。分析时重点关注Kernel耗时、AI Core利用率、各执行流水耗时、内存搬运量、流水等待和多核执行差异。

PyPTO Pro不包含AI CPU侧的Execute Graph调度，其优化对象是单个SPMD Kernel。因此，面向Execute Graph的图执行泳道、图节点耗时以及图级调试环境变量不适用于PyPTO Pro。需要分析端到端调用时，应结合Host侧时间和Kernel时间分别判断开销来源。

## 判断性能瓶颈

| Profiling现象 | 可能原因 | 优化方向 |
|---|---|---|
| 计算流水利用率高且持续繁忙 | Kernel接近计算受限 | 提高Cube或Vector指令效率，减少重复计算，选择合适的数据类型和计算布局。 |
| 数据搬运耗时高、计算流水空闲 | GM或片上搬运受限 | 提高连续访问比例，增大有效搬运粒度，复用片上数据，减少重复加载和写回。 |
| 搬运和计算先后串行 | 流水重叠不足 | 使用TileGroup和多缓冲，使下一块数据搬运与当前块计算重叠。 |
| 流水存在较长等待 | 同步过多或依赖关系不合理 | 检查mutex、Tile复用和手工同步，删除不必要的屏障，同时保证真实依赖不被破坏。 |
| 不同Core耗时差异明显 | 多核负载不均或尾块集中 | 调整分片方式，将Tile跨步或均匀分配给各Core。 |
| Core数量增加但性能无提升 | 单Core工作量过小、核数超过可并行任务数或访存带宽饱和 | 根据Tile总数和硬件资源重新选择`block_dim`。 |
| 小Shape中Kernel时间占比低 | Host下发或固定启动开销占主导 | 减少不必要的多次Kernel启动，在实际调用链路评估端到端收益。 |

## 多核切分优化

PyPTO Pro通过`pl.get_block_idx()`和`pl.get_block_num()`进行多核分片。合理的多核策略应同时满足完整覆盖、无重复写和负载均衡。

- `block_dim`不应超过可独立执行的任务块数量，否则会产生空闲Core。
- 每个Core的工作量应尽量接近，避免将全部尾块或耗时较高的分支集中到少数Core。
- 规则二维Tile可先线性编号，再按Core编号进行跨步分配，以减小尾部负载差异。
- 输出区域应由唯一Core写入；需要跨Core归约时，应使用明确且受支持的同步与归约方案。
- 对小Shape和大Shape分别测试核数，单一`block_dim`不一定适合全部Shape。

调试阶段使用`block_dim=1`有助于验证逻辑，但性能测试必须恢复目标核数。

## Tile与片上内存优化

Tile Shape决定一次计算的数据量、片上内存占用、循环次数、尾块比例和搬运效率。选择Tile时需要综合考虑：

- Vec/UB、Mat/L1、Left/L0A、Right/L0B和Acc/L0C的容量限制。
- 数据类型和Tile Shape共同决定的实际字节数。
- 数据搬运和计算指令的对齐要求。
- 尾块比例以及有效数据之外的补齐开销。
- 同一份输入数据在片上的复用次数。
- 多缓冲后总内存占用的倍增。

增大Tile可以减少循环和指令下发次数，但也会增加片上内存占用，可能降低双缓冲可行性或减少并行度。缩小Tile可以降低单块资源占用，但会增加循环、搬运和尾块处理开销。应通过实测选择目标场景的平衡点。

GM访问应尽量连续、对齐并合并为较大的有效搬运。对于会被多次使用的数据，应在容量允许时保留在片上，避免在每个计算步骤中重复从GM加载。中间结果能够在片上直接消费时，应避免不必要的GM写回和重新加载。

## 流水与多缓冲优化

`pl.make_tile_group`可为同一逻辑数据声明多块轮转Tile，通过`current()`取得当前Tile、通过`next()`轮转到下一块。配合`@pl.jit(auto_mutex=True)`，编译器根据Tile的mutex信息插入核内流水同步，可用于构建双缓冲或N缓冲。

优化时应重点检查：

- 加载下一块数据能否与当前块的Vector或Cube计算重叠。
- 当前结果写回能否与后续计算重叠。
- 每个轮转Tile是否使用独立且不冲突的mutex编号。
- Tile调用`next()`的次数和位置是否与生产、消费顺序一致。
- 是否存在不必要的手工`sync_src`、`sync_dst`或其他屏障。
- 流水末尾是否正确排空，最后一个结果是否完成写回。

缓冲数量越多，片上内存占用越大，管理开销也可能增加。双缓冲不能掩盖全部延迟时才考虑更多缓冲，并通过Profiling验证收益。

## Cube与Vector协同优化

矩阵类Kernel使用`pl.section_cube()`描述Cube任务，向量类处理使用`pl.section_vector()`描述Vector任务。混合Kernel应尽量让Cube计算、Vector前后处理和DMA搬运并行，同时避免不必要的数据格式转换和跨存储层往返。

- Cube计算应检查M、N、K方向的Tile Shape、左右矩阵布局、转置方式以及L0A/L0B装载格式。
- 归约长度较大时，应在Acc/L0C中完成分块累加，再按需要转换并写回。
- Vector前后处理应尽量与Cube流水重叠，避免形成全局串行阶段。
- 多个连续的细粒度Vector操作产生明显额外开销时，可在确认瓶颈后使用Vector Function表达寄存器级计算，减少不必要的中间Tile读写。
- 调整Cube与Vector协同时必须重新检查mutex和真实数据依赖，不能通过删除同步换取错误的表面性能。

## TilingKey与编译期特化

TilingKey适合表达数量有限、执行路径差异明显且能够带来性能收益的编译期条件。例如，对齐路径与非对齐路径、不同算法模式或有限的数据布局可以分别生成专用Kernel，从而消除热循环中的无效分支。

不应将取值范围很大的运行时Shape直接展开为TilingKey。过多Key会增加二进制数量、编译时间和缓存占用，也会提高测试与交付复杂度。只有当特化收益经过测量且Key集合可控时，才应新增TilingKey。

常用Shape可使用规整、无分支的主路径，尾块和少见场景使用独立分支处理。热循环中应尽量减少取决于运行时数据的分支，但不能省略必要的边界检查。

## 结果验证与交付检查

完成调优后，应执行以下检查：

- 全部目标Shape、数据类型、布局、TilingKey和`block_dim`均通过精度测试。
- 覆盖最小Shape、非对齐Shape、尾块、空闲Core和最大资源占用等边界场景。
- 在相同测量条件下比较基线与优化版本，并保存多轮结果。
- Profiling数据能够解释性能变化，不以偶然波动作为优化结论。
- 已移除调试接口和仅用于定位问题的额外同步。
- JIT直接调用和实际aclnn调用路径均完成性能验证。
- 二进制大小、TilingKey数量和首次编译时间仍处于可接受范围。
