# Kernel核函数创建

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

核函数内部使用[`pypto_pro.language.TileType`](../../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md)定义Tile类型，并通过[`pypto_pro.language.make_tile_group`](../../../../api/pro_api/SIMD-API/resource_management/make_tile_group.md)等接口分配片上内存：

```python
tt = pypto_pro.language.TileType(shape=[64, 64], dtype=pypto_pro.language.DT_FP16, target_memory=pypto_pro.language.MemorySpace.Vec)
tile_x = pypto_pro.language.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
tile_y = pypto_pro.language.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
tile_out = pypto_pro.language.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2])
```

### 流水段与同步

计算逻辑需要放在[`pypto_pro.language.section_vector()`](../../../../api/pro_api/SIMD-API/controlflow/section_vector.md)上下文中，开启`auto_mutex=True`后，搬运与计算间的流水同步由框架按Tile的mutex自动插入：

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

PyPTO Pro JIT将`block_dim`作为请求上限，每次启动按实际Stream的torch_npu核数限制计算Block数。
Stream未单独配置时继承Device限制，再回退到硬件核数。Kernel应使用`get_block_num()`分配全部任务，
避免减少工作核后遗漏Tile。详见[多核Tiling切分](tiling/multi_core_tiling.md#devicestream与作用域限核)。

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

`block_dim`为Host请求的逻辑Block数上限，必须是正整数。JIT按Stream的有效资源限制计算
实际启动值`block_num`，Kernel通过`pypto_pro.language.get_block_num()`读取该值。
它在不同Kernel执行模式下的含义如下：

| Kernel执行模式 | `block_dim`请求的工作单元 | 实际工作单元数 | Stream资源上限 |
|:---|:---|:---|:---|
| 仅Cube | AIC逻辑核数 | AIC：`block_num` | `cube_core_num` |
| 仅Vector | AIV逻辑核数 | AIV：`block_num` | `vector_core_num` |
| AIC:AIV为1:2的混合Kernel | AIC/AIV执行组数 | AIC：`block_num`；AIV：`2 * block_num` | 见下文混合Kernel上限 |

`block_num`可能小于Host请求的`block_dim`，数据切分应使用实际Block数。在AIC:AIV为1:2的混合Kernel的Vector段中，
`pypto_pro.language.get_subblock_num()`返回2，`pypto_pro.language.get_block_idx()`返回已经按两个AIV
subblock展平后的全局逻辑索引，范围为`[0, 2 * block_num)`；如果需要区分同一执行组内的两个AIV，
可使用`pypto_pro.language.get_subblock_idx()`获取0或1。

例如，Host请求`block_dim=12`，Stream限制为4个Cube Core和6个Vector Core时，混合Kernel实际启动
`block_num=3`个执行组，包含3个AIC和6个AIV。

JIT校验`block_dim`的类型和正值，并按Stream的有效资源限制减少实际启动值。
仅Cube、仅Vector和AIC:AIV为1:2的混合Kernel的上限分别为`cube_core_num`、`vector_core_num`和
`min(cube_core_num, vector_core_num // 2)`。详细计算方式参见
[多核Tiling切分](tiling/multi_core_tiling.md#在启动时设置逻辑block数block_dim)。

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

`tiling_key`的完整说明（字段定义、`is_valid`校验、与TilingData的组合、运行时标志与TilingKey的选型对照表）请参考[TilingKey](tiling/tiling_result_transfer.md#tilingkey)。

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
