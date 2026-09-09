# 多核Tiling切分

多核Tiling将一个Tensor的计算任务划分为多个Tile任务，并把这些任务分配给多个逻辑AI Core。Kernel的每个逻辑Block执行同一份程序，通过Block索引处理不同的数据区域。

**图1 多核SPMD任务映射**

![多核SPMD任务映射](../../../../figures/pro/pro_multicore_spmd_mapping.png "多核SPMD任务映射")

## 逻辑Block与执行域

启动Kernel时通过`block_dim`指定逻辑Block数。Kernel内使用以下接口获取实际执行域：

| 接口 | 含义 |
| --- | --- |
| `pypto_pro.language.get_block_idx()` | 当前逻辑Block的索引。 |
| `pypto_pro.language.get_block_num()` | 当前Kernel实际使用的逻辑Block数。 |
| `pypto_pro.language.get_subblock_idx()` | 混合Kernel中，当前Vector子Block的索引。 |
| `pypto_pro.language.get_subblock_num()` | 每个Cube Block对应的Vector子Block数。 |

`block_dim`是启动时申请的逻辑Block数，`pypto_pro.language.get_block_num()`返回运行时实际生效的值。多核循环必须使用该接口计算步长，避免限核后遗漏任务。

不同Kernel类型的执行域如下：

| Kernel类型 | `get_block_idx()` | 工作单元数 |
| --- | --- | --- |
| Vector | AIV的逻辑Block索引 | `get_block_num()` |
| Cube | AIC的逻辑Block索引 | `get_block_num()` |
| Cube与Vector混合 | Cube侧为AIC索引；Vector侧已展平为AIV索引 | Cube侧为`get_block_num()`；Vector侧为`get_block_num() * get_subblock_num()` |

混合Kernel中，可以根据当前执行单元分别组织Cube和Vector代码：

```python
import pypto_pro.language as pl


with pl.section_cube():
    cube_idx = pl.get_block_idx()
    cube_num = pl.get_block_num()
    # 分配Cube任务

with pl.section_vector():
    vector_idx = pl.get_block_idx()
    vector_num = pl.get_block_num() * pl.get_subblock_num()
    # 分配Vector任务
```

Vector侧的`pypto_pro.language.get_block_idx()`已经包含子Block偏移，无需再次使用`pypto_pro.language.get_subblock_idx()`展平索引。

## 使用跨步循环进行多核切分

设Tile任务总数为`total_tiles`，逻辑Block `core_id`从`core_id`开始，每次跨过全部工作单元数：

```python
import pypto_pro.language as pl


core_id = pl.get_block_idx()
core_num = pl.get_block_num()

for tile_idx in pl.range(core_id, total_tiles, core_num):
    # 处理第 tile_idx 个Tile任务
    ...
```

这种跨步切分不要求任务数能够被核数整除。当`total_tiles < core_num`时，索引超出任务范围的逻辑Block不会进入循环。

**图2 跨步多核切分**

![跨步多核切分](../../../../figures/pro/pro_multicore_strided_partition.png "跨步多核切分")

### 二维任务的切分

二维Tile网格可以先展平，再按一维索引分配：

```python
import pypto_pro.language as pl


m_tiles = (m + tile_m - 1) // tile_m
n_tiles = (n + tile_n - 1) // tile_n
total_tiles = m_tiles * n_tiles

core_id = pl.get_block_idx()
core_num = pl.get_block_num()

for tile_idx in pl.range(core_id, total_tiles, core_num):
    tile_m_idx = tile_idx // n_tiles
    tile_n_idx = tile_idx % n_tiles
    ...
```

也可以先将外层维度分配给不同逻辑Block，再在每个Block内遍历内层维度：

```python
import pypto_pro.language as pl


for tile_m_idx in pl.range(core_id, m_tiles, core_num):
    for tile_n_idx in pl.range(0, n_tiles, 1):
        ...
```

**图3 展平切分与二维切分**

![展平切分与二维切分](../../../../figures/pro/pro_multicore_flat_vs_2d.png "展平切分与二维切分")

两种方式的选择取决于任务形状和数据访问方式：

- 展平切分的任务粒度更细，通常更容易均衡各核负载。
- 按外层维度切分便于同一逻辑Block复用一行或一列数据，但外层Tile数较少时并行度会受到限制。

## 在启动时设置逻辑Block数（block_dim）

Kernel启动格式为：

```python
kernel[stream, block_dim](...)
```

`stream`为执行流，使用当前流时可传入`None`；`block_dim`为正整数。设置`block_dim`时应同时考虑可用核数和有效任务数：

```python
from pypto_pro.runtime.platform import get_platform_info

platform_info = get_platform_info()

# Vector Kernel
block_dim = min(platform_info.vector_core_num, total_tiles)

# Cube Kernel
block_dim = min(platform_info.cube_core_num, total_tiles)
```

混合Kernel以Cube Block为启动单位，并同时受Cube核数和对应Vector核数约束：

```python
block_dim = min(
    platform_info.cube_core_num,
    platform_info.vector_core_num // 2,
    total_cube_tasks,
)
```

不要仅为了占满所有核而增大`block_dim`。当任务数较少或单核工作量过小时，更多逻辑Block不会增加有效并行度，还会增加调度开销。

### Device、Stream与作用域限核

运行时可以在Device、Stream或代码作用域上限制可用核数。更具体的限制覆盖更宽泛的限制：

```python
import torch
import torch_npu

device = torch.npu.current_device()
torch.npu.set_device_limit(device, cube_num=8, vector_num=16)

stream = torch.npu.Stream()
torch.npu.set_stream_limit(stream, cube_num=4, vector_num=6)

kernel[stream, 32](...)

with torch.npu.npugraph_ex.scope.limit_core_num(2, 4, stream=stream):
    kernel[stream, 32](...)

torch.npu.reset_stream_limit(stream)
```

限核可能使实际逻辑Block数小于启动时传入的`block_dim`。Kernel内始终以`pypto_pro.language.get_block_num()`返回的值作为跨步循环的步长。使用限核接口时还需要注意：

- Stream限核配置覆盖Device限核配置；Stream未配置时继承Device限制。
- `set_device_limit`用于设置Device默认值，需要反复调整时使用Stream限核接口。
- `limit_core_num`作用域退出后恢复该Stream原来的限制。
- 图捕获按照捕获时的Stream限制确定Block数；修改限制后需要重新捕获，已有图的`replay()`不会重新查询核数。

## 多核Tiling设计

### 负载均衡

每个Tile任务的计算量接近时，跨步切分通常可以获得较均衡的负载。任务计算量差异较大时，可以调整展平顺序，使重任务分散到不同逻辑Block；也可以拆分粒度过大的任务，减少单个长任务造成的尾部等待。

设计切分方式时重点检查：

- Tile任务总数是否足以覆盖计划使用的逻辑Block。
- 每个逻辑Block分到的任务数和计算量是否接近。
- 相邻任务是否复用数据，以及展平顺序是否破坏这种复用。
- 切分后的Tile形状是否满足数据搬运和计算接口的约束。

Tile大小、Buffer分配和Tile计算分别参考[Tile创建和操作](../tile_creation_and_operations.md)和[Tile计算](../vector_computation/tile_computation.md)。Host侧生成的运行时Tiling参数参考[Tiling结果传输](tiling_result_transfer.md)。

### 矩阵乘任务映射示例

矩阵乘可将输出矩阵划分为`m_tiles * n_tiles`个任务，每个任务计算一个输出Tile：

```python
import pypto_pro.language as pl


m_tiles = (m + tile_m - 1) // tile_m
n_tiles = (n + tile_n - 1) // tile_n
total_tiles = m_tiles * n_tiles

core_id = pl.get_block_idx()
core_num = pl.get_block_num()

for task_idx in pl.range(core_id, total_tiles, core_num):
    m_idx = task_idx // n_tiles
    n_idx = task_idx % n_tiles

    # 沿K方向累加输出Tile
    for k_idx in pl.range(0, k_tiles, 1):
        ...
```

该映射把输出Tile作为独立任务，避免多个逻辑Block同时写同一输出区域。若算法需要跨核归约，应另外设计中间结果和归约阶段，不能假定不同逻辑Block之间存在隐式同步。

<a id="常见问题"></a>

## 尾块处理

尾块场景需要向上取整计算Tile数以覆盖全部任务，并根据数据搬运和计算接口的对齐要求选择Tile尺寸；有效形状、填充和计算方法参考[Tile计算中的尾块处理](../vector_computation/tile_computation.md#尾块处理)。

## 使用限制与建议

- 不要把启动时传入的`block_dim`当作实际核数，Kernel内使用`pypto_pro.language.get_block_num()`。
- 混合Kernel的Vector工作单元数需要乘以`pypto_pro.language.get_subblock_num()`；纯Vector Kernel不需要。
- `block_dim`不能替代Tiling设计。任务粒度过大时负载不均，粒度过小时调度和重复搬运开销会增大。
- 多核切分只负责分配任务，不提供逻辑Block之间的隐式同步。
- 运行时shape、循环边界等参数通过TilingData传递；有限的编译期模式通过TilingKey选择，参考[Tiling结果传输](tiling_result_transfer.md)。
