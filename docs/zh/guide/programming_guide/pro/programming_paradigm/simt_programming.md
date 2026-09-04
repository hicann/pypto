# SIMT编程模型

SIMT（Single Instruction Multiple Threads）以线程为基本编程单元。开发者定义一份SIMT函数，由多个线程执行相同的函数体，并通过线程索引处理不同的数据。

PyPTO Pro支持在Ascend 950PR/Ascend 950DT的AIV上使用SIMT。SIMT与Tile API、Reg API面向不同的计算特点，可以在同一个Kernel中组合使用。

## 适用场景

SIMT允许每个线程独立计算数据地址和执行条件分支，适合以下场景：

- 数据访问离散，难以组织为规则的连续Tile计算；
- 控制流复杂，不同数据元素可能执行不同分支；
- 需要使用线程索引表达逐元素处理逻辑；
- 多个线程可能更新同一位置，需要使用原子操作处理并发访问。

对于规则、连续且计算密集的批量数据处理，优先考虑Tile API或Reg API；对于不规则访存和分支较多的计算，可以使用SIMT简化索引和控制逻辑。

## 执行模型

### 线程层次

SIMT使用Grid、Thread Block和Thread组织并行任务。Warp是硬件对线程进行调度和执行的分组，不需要在启动时单独配置。

| 层次 | 说明 | PyPTO Pro中的表示 |
|---|---|---|
| Grid | 由多个尺寸相同的Thread Block组成 | grid_dim()返回Grid尺寸，block_idx()返回当前Thread Block索引 |
| Thread Block | 由一至三维线程组成，块内线程执行同一份SIMT入口函数 | block_dim()返回线程块尺寸，每个线程块最多包含2048个线程 |
| Thread | SIMT编程的基本单元，每个线程拥有独立的寄存器和栈空间 | thread_idx()返回线程在线程块内的三维坐标 |

在PyPTO Pro中，Grid不通过pl.simt.launch单独配置。每个执行到pl.simt.launch的Vector逻辑Block启动一个SIMT Thread Block，因此grid_dim().x由当前Vector执行域的逻辑Block数量决定，grid_dim().y和grid_dim().z当前固定为1。threads只配置单个Thread Block的尺寸，不配置Grid大小。

### Warp执行

一个Thread Block中的线程由硬件按照线性顺序划分为Warp，每个Warp包含32个线程。同一Warp中的线程执行相同的指令，但可以根据条件进入不同分支。

当同一Warp中的线程进入不同分支时，硬件需要分别执行各分支路径，这种情况称为分支发散。分支发散不会改变计算结果，但可能降低执行效率。线程块的线程数不要求必须是32的整数倍；从执行效率考虑，通常建议优先使用32的整数倍，并尽量让同一Warp中的线程执行相同路径。

## 线程组织与索引

### 配置线程块

pl.simt.launch通过threads配置一至三维线程块，未指定的维度补为1：

| threads写法 | block_dim() | 线程总数 |
|---|---|---|
| threads=256 | (256, 1, 1) | 256 |
| threads=(16, 16) | (16, 16, 1) | 256 |
| threads=(8, 4, 8) | (8, 4, 8) | 256 |

threads的每个维度必须是取值范围为[1, 2048]的编译期整数，各维乘积不能超过2048，也不能超过入口函数声明的max_threads。max_threads表示入口函数允许的最大线程数，实际线程块尺寸仍由threads决定。

### 获取执行上下文

| 接口 | 含义 | 当前取值范围 |
|---|---|---|
| pl.simt.thread_idx() | 当前线程在线程块内的三维坐标 | 各分量从0开始，小于block_dim()对应分量 |
| pl.simt.block_dim() | 当前线程块在X、Y、Z三个维度的线程数 | 由threads确定 |
| pl.simt.block_idx() | 当前线程块在Grid中的索引 | X维从0开始，Y维和Z维为0 |
| pl.simt.grid_dim() | Grid在X、Y、Z三个维度的线程块数量 | X维为Vector逻辑Block数量，Y维和Z维为1 |
| pl.simt.linear_thread_idx() | 当前线程在线程块内的一维编号 | 从0开始，小于线程块的线程总数 |

三维线程坐标按照X维优先的顺序展开：

$$
\text{local\_idx} = \text{threadIdx.x}
+ \text{threadIdx.y} \times \text{blockDim.x}
+ \text{threadIdx.z} \times \text{blockDim.x} \times \text{blockDim.y}
$$

一维线程块中，linear_thread_idx()与thread_idx().x相同。linear_thread_idx()只在线程块内唯一；需要计算当前Grid内的一维线程编号时，可以结合block_idx().x：

$$
\text{threads\_per\_block} = \text{blockDim.x} \times \text{blockDim.y} \times \text{blockDim.z}
$$

$$
\text{global\_idx} = \text{blockIdx.x} \times \text{threads\_per\_block} + \text{local\_idx}
$$

## 定义和启动SIMT函数

### SIMT函数类型

@pl.simt.function将Python函数标记为SIMT函数。根据是否设置max_threads，SIMT函数分为入口函数和辅助函数：

| 类型 | 定义方式 | 调用方式 | 返回值 |
|---|---|---|---|
| 入口函数 | @pl.simt.function(max_threads=N) | 由外层Kernel通过pl.simt.launch启动 | 无返回值 |
| 辅助函数 | @pl.simt.function | 由入口函数或其他辅助函数直接调用 | 无返回值或一个Scalar |

入口函数对应一个可以启动的线程块。辅助函数用于复用逐线程计算逻辑，调用辅助函数不会创建新的线程块。入口函数不能作为普通SIMT辅助函数调用，辅助函数也不能作为pl.simt.launch的启动目标。

SIMT函数的参数和返回值类型标注是可选的。未提供标注时，参数类型根据调用点实参推导；提供标注时，标注用于校验类型兼容性。函数参数必须是必选位置参数，不支持默认参数、可变参数和仅关键字参数。辅助函数不支持递归调用。

### 启动入口函数

在外层JIT Kernel的pl.section_vector作用域中调用pl.simt.launch：

```python
pl.simt.launch(
    simt_func,
    threads=256,
    args=(dst, src, count),
)
```

callee必须是设置了max_threads的SIMT入口函数，args中的实参数量、顺序和类型必须与入口函数形参一致。SIMT函数内部不能嵌套调用pl.simt.launch。

pl.simt.launch在Vector流水上执行。使用auto_mutex并传入由make_tile_group创建且配置了mutex_ids的Tile时，框架会处理对应的流水同步；未被auto_mutex覆盖的跨流水依赖，需要在外层Kernel中处理。

## 数据访问

SIMT函数通过参数接收数据，每个线程使用自己的索引访问对应元素：

| 数据类型 | 用途 | 访问方式 |
|---|---|---|
| Scalar | 标量参数、索引和条件计算 | 直接参与标量表达式 |
| Tensor | 访问Global Memory中的数据 | 通过完整索引取得或更新一个Scalar元素 |
| Vec Tile | 访问Vector核UB中的数据 | 通过二维索引取得或更新一个Scalar元素 |

传入SIMT函数的Tensor需要使用ND Layout；Tile需要是二维ND Vec Tile。Tensor和Tile必须以完整变量传给pl.simt.launch，元素下标表达式、Slice和Tile Subview不能作为launch实参。具体限制请参见[simt.launch](../../../../api/pro_api/SIMT-API/execution/launch.md)。

## 编程示例

下面的示例使用四个Vector逻辑Block处理1024个元素。每个Vector逻辑Block启动一个包含256个线程的SIMT Thread Block，每个线程根据全局线程编号处理一个元素。辅助函数affine复用逐线程计算逻辑，不会产生新的线程层次。

```python
import pypto_pro.language as pl


@pl.simt.function
def affine(
    value: pl.DT_FP32,
    scale: pl.DT_FP32,
    delta: pl.DT_FP32,
) -> pl.DT_FP32:
    return value * scale + delta


@pl.simt.function(max_threads=256)
def transform(
    dst: pl.Tensor[[1, 1024], pl.DT_FP32],
    src: pl.Tensor[[1, 1024], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    delta: pl.DT_FP32,
):
    block = pl.simt.block_dim()
    threads_per_block = block.x * block.y * block.z
    index = pl.simt.block_idx().x * threads_per_block + pl.simt.linear_thread_idx()
    if index < count:
        dst[0, index] = affine(src[0, index], scale, delta)


@pl.jit()
def transform_kernel(
    src: pl.Tensor[[1, 1024], pl.DT_FP32],
    dst: pl.Tensor[[1, 1024], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    delta: pl.DT_FP32,
):
    with pl.section_vector():
        pl.simt.launch(
            transform,
            threads=256,
            args=(dst, src, count, scale, delta),
        )
```

Host端使用transform_kernel\[None, 4\](...)启动该Kernel时，当前Vector执行域包含四个逻辑Block。SIMT函数中grid_dim()为(4, 1, 1)，block_idx().x的取值范围为[0, 4)，四个Thread Block合计处理1024个线程任务。

## 编程建议

- 优先使用一维线程块；仅当数据天然具有二维或三维索引时，再使用多维threads配置；
- 线程块大小通常选择32的整数倍，减少最后一个Warp中的空闲线程；
- 让相邻线程尽量访问连续的数据元素，以提高访存效率；
- 尽量减少同一Warp内的分支发散；
- 数据规模不一定是总线程数的整数倍时，在访问数据前进行边界判断；
- 规则的批量向量计算优先使用Tile API或Reg API，SIMT用于表达不规则访存和复杂控制逻辑。

## 相关文档

- [Add算子（SIMT）快速入门](../../../quick_start/pro/SIMT/Add_operator.md)
- [SIMT API](../../../../api/index.md)
- [simt.launch](../../../../api/pro_api/SIMT-API/execution/launch.md)
- [SIMT执行](../../../../api/index.md)
