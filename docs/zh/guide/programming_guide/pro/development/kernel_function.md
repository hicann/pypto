# Tile核函数

Tile核函数（Kernel Function）是在NPU设备侧执行的Python函数。它由Host端代码调用，PyPTO Pro框架自动将其编译为硬件指令，并调度到AI Core上执行。每个Kernel函数通过显式的Tile定义、数据搬运和同步控制，精确管理片上计算流程。

## 核函数的定义

定义Tile核函数时需要遵循以下规则：

### 使用JIT装饰器

必须使用`@pypto_pro.language.jit()`装饰器标识该函数为Kernel函数，PyPTO Pro框架会将其编译为NPU可执行的二进制。

### 参数类型标注

Tensor输入输出通常使用[`pypto_pro.language.Tensor`](../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md)标注，并指定张量的形状和数据类型：

```python
@pypto_pro.language.jit()
def my_kernel(x: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16],
              y: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16],
              out: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16]):
    ...
```

根据数据和参数的组织方式，Kernel还支持以下参数类型：

| 参数类型 | 典型用途 |
| --- | --- |
| `pypto_pro.language.Tensor[[shape], dtype]` | shape和数据类型可在签名中确定的Tensor |
| [`pypto_pro.language.Ptr[dtype]`](../../../../api/pro_api/SIMD-API/basic_data_structures/Ptr.md) | 裸指针输入输出；常与TilingData配合重建动态shape的Tensor视图 |
| `pypto_pro.language.DT_*` | 运行时标量参数，例如`pypto_pro.language.DT_INT64`、`pypto_pro.language.DT_FP32` |
| TilingData类 | 传递shape、循环边界、算子选择器等结构化运行时参数 |

下面的示例通过`pypto_pro.language.Ptr`和TilingData重建动态shape的Tensor视图。当前JIT要求
TilingData位于Kernel形参和启动实参的末尾：

```python
from dataclasses import dataclass

@dataclass
class AddTiling:
    m: int
    n: int

@pypto_pro.language.jit(auto_mutex=True)
def dynamic_kernel(
    x: pypto_pro.language.Ptr[pypto_pro.language.DT_FP16],
    out: pypto_pro.language.Ptr[pypto_pro.language.DT_FP16],
    scale: pypto_pro.language.DT_FP32,
    tiling: AddTiling,
):
    tensor_x = pypto_pro.language.make_tensor(x, [tiling.m, tiling.n])
    tensor_out = pypto_pro.language.make_tensor(out, [tiling.m, tiling.n])
    ...
```

核函数不支持返回值，计算结果通过与`pypto_pro.language.Tensor`或`pypto_pro.language.Ptr`输出参数对应的缓冲区写回。

### Tile定义与分配

核函数内部使用[`pypto_pro.language.TileType`](../../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md)定义Tile类型，并通过[`pypto_pro.language.make_tile_group`](../../../../api/pro_api/SIMD-API/operation/resource_management/make_tile_group.md)等接口分配片上内存：

```python
tt = pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP16, target_memory=pypto_pro.language.MemorySpace.Vec)
tile_x = pypto_pro.language.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
tile_y = pypto_pro.language.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
tile_out = pypto_pro.language.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2])
```

### 流水段与同步

计算逻辑需要放在[`pypto_pro.language.section_vector()`](../../../../api/pro_api/SIMD-API/operation/controlflow/section_vector.md)上下文中，开启`auto_mutex=True`后，搬运与计算间的流水同步由框架按Tile的mutex自动插入：

```python
with pypto_pro.language.section_vector():
    cur_x = tile_x.current()
    cur_y = tile_y.current()
    cur_out = tile_out.current()
    pypto_pro.language.load(cur_x, x, [0, 0])
    pypto_pro.language.load(cur_y, y, [0, 0])
    pypto_pro.language.add(cur_out, cur_x, cur_y)
    pypto_pro.language.store(out, cur_out, [0, 0])
```

### 其他规则

- 运行时标量形参使用`pypto_pro.language.DT_*`类型标注，Host侧传入对应的Python标量值。
- 使用TilingData时，必须将其放在Kernel形参和启动实参的末尾。
- 可以使用`auto_mutex=True`参数启用自动互斥锁插入。

## 核函数的调用

PyPTO Pro中核函数通过方括号启动语法发起，方括号内指定Stream和`block_dim`（逻辑Block数）：

```python
import os
# 准备输入数据
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
a = torch.rand(64, 64, device=device, dtype=torch.float16)
b = torch.rand(64, 64, device=device, dtype=torch.float16)
out = torch.empty(64, 64, device=device, dtype=torch.float16)

# 方括号启动语法：[stream, block_dim]
# None表示默认Stream，num_cores指定逻辑Block数
add_kernel[None, num_cores](a, b, out)
torch.npu.synchronize()
```

Kernel执行域决定启动类型：

- 仅包含`pypto_pro.language.section_vector()`的Kernel编译并启动为Vector（AIV）Kernel，`block_dim`表示启动的AIV逻辑核数。
- 仅包含`pypto_pro.language.section_cube()`的Kernel编译并启动为Cube（AIC）Kernel，`block_dim`表示启动的AIC逻辑核数。
- 同时包含`pypto_pro.language.section_vector()`和`pypto_pro.language.section_cube()`的Kernel编译并启动为混合Kernel，`block_dim`表示AIC与AIV的配对执行组数，而不是AIV总数。具体工作单元数取决于AIC:AIV比例。

PyPTO Pro JIT不会自动截断超过平台核数上限的`block_dim`。Host侧应根据Kernel类型和
[`get_platform_info()`](tile_based_python_programming/multi_core_partitioning_and_Tiling.md#在启动时设置逻辑block数block_dim)
返回的核数上限计算`block_dim`。

也可以省略方括号直接调用，此时使用默认`block_dim=1`：

```python
add_kernel(a, b, out)
torch.npu.synchronize()
```

使用`tiling_key`或`datatype`特化的Kernel必须通过方括号语法提供对应的特化字典，不能省略方括号直接调用。

核函数的调用是异步的。首次调用时触发JIT编译；在同一Python进程中，同一Kernel对象以相同编译签名再次调用时复用编译结果。重新启动Python进程后会重新执行生成与编译流程。

### stream的含义与设置

`stream`指定Kernel下发的NPU执行流。传入`None`表示使用当前Stream：

```python
kernel[None, num_cores](x, out)
```

也可以显式传入`torch.npu.Stream`，并仅同步该Stream：

```python
stream = torch.npu.Stream()
kernel[stream, num_cores](x, out)
stream.synchronize()
```

### blockDim的含义与设置

`block_dim`为启动时传入的基础逻辑Block数，必须是正整数。它在不同Kernel执行模式下的含义如下：

| Kernel执行模式 | `block_dim`的含义 | 实际工作单元数 | 建议使用的平台上限 |
|:---|:---|:---|:---|
| 仅Cube | AIC逻辑核数 | AIC：`block_dim` | `cube_core_num` |
| 仅Vector | AIV逻辑核数 | AIV：`block_dim` | `vector_core_num` |
| AIC:AIV为1:2的混合Kernel | AIC/AIV配对执行组数 | AIC：`block_dim`；AIV：`2 * block_dim` | `core_num` |

Kernel中的`pypto_pro.language.get_block_num()`始终返回启动时传入的`block_dim`。在1:2混合Kernel的Vector段中，`pypto_pro.language.get_subblock_num()`返回2，`pypto_pro.language.get_block_idx()`返回已经按两个AIV subblock展平后的全局逻辑索引，范围为`[0, 2 * block_dim)`；如果需要区分同一执行组内的两个AIV，可使用`pypto_pro.language.get_subblock_idx()`获取0或1。

JIT只校验`block_dim`的类型和正值，不会按平台核数上限自动截断。Host侧应根据Kernel执行模式、平台核数和任务分片数计算合法值。详细计算方式参见[多核切分与Tiling](tile_based_python_programming/multi_core_partitioning_and_Tiling.md#在启动时设置逻辑block数block_dim)。

## Tiling参数化

`@pypto_pro.language.jit()`支持通过`tiling_key`参数实现Tiling参数化，在启动时通过字典选择不同的Kernel实例化，使每种模式各编一份专用Kernel（消除死分支、拿到最优指令）：

```python
from pypto_pro.runtime.tilingkey import TilingKeyField

class MyTilingKey:
    NeedAttnMask = TilingKeyField(bits=1, values=[0, 1])

@pypto_pro.language.jit(tiling_key=MyTilingKey)
def my_kernel(x: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16], out: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16]):
    ...

# 启动时通过字典选择实例化
my_kernel[None, num_cores, {"NeedAttnMask": 1}](x, out)
my_kernel[None, num_cores, {"NeedAttnMask": 0}](x, out)
```

`tiling_key`的完整说明（字段定义、`is_valid`校验、与TilingData的组合、运行时标志与TilingKey的选型对照表）请参考[多核切分与Tiling](tile_based_python_programming/multi_core_partitioning_and_Tiling.md#tiling-key--编译期特化而非运行时传值)。

## JIT配置选项

`@pypto_pro.language.jit()`装饰器支持以下配置选项：

| 选项 | 说明 | 默认值 |
|:---|:---|:---|
| arch | 目标架构，当前可选“a5”；None为自动检测当前受支持设备的架构 | None |
| auto_mutex | 是否启用自动互斥锁插入 | True |
| compile_timeout | 编译超时时间（秒）；显式设置时使用该值，传入或保持`None`时先读取当前PyPTO配置作用域，作用域也未配置时使用600秒 | None（有效默认值为600秒） |
| name | 自定义Kernel名称，用于构建产物路径隔离 | None |
| tiling_key | Tiling键类型，用于Tiling参数化 | None |
| pipeline | PipelineConfig，用于自动预取流水变换 | None |
| datatype | 数据类型特化，用于同一Kernel支持多种数据类型 | None |

```python
@pypto_pro.language.jit(arch="a5", auto_mutex=True, compile_timeout=200)
def my_kernel(x: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16],
              out: pypto_pro.language.Tensor[[64, 64], pypto_pro.language.DT_FP16]):
    ...
```

> [!NOTE]说明
> Kernel特有的选项通过`@pypto_pro.language.jit()`配置；Host、Pass、CodeGen、验证和调试等共享编译配置通过`pypto.options(...)`配置。完整说明参见[JIT编译](compilation_and_execution/JIT_compilation.md#编译配置)。
