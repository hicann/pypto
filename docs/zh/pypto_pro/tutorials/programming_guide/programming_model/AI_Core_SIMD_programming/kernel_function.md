# Tile核函数

Tile核函数（Kernel Function）是在NPU设备侧执行的Python函数。它由Host端代码调用，PyPTO Pro框架自动将其编译为硬件指令，并调度到AI Core上执行。每个Kernel函数通过显式的Tile定义、数据搬运和同步控制，精确管理片上计算流程。

## 核函数的定义

定义Tile核函数时需要遵循以下规则：

### 使用JIT装饰器

必须使用`@pl.jit()`装饰器标识该函数为Kernel函数，PyPTO Pro框架会将其编译为NPU可执行的二进制。

### 参数类型标注

核函数的输入输出参数需要使用`pl.Tensor`进行类型标注，指定张量的形状和数据类型：

```python
@pl.jit()
def my_kernel(x: pl.Tensor[[64, 64], pl.DT_FP16],
              y: pl.Tensor[[64, 64], pl.DT_FP16],
              out: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...
```

核函数不支持返回值，计算结果通过输出参数（`pl.Tensor`）配合`pl.store`写回。

### Tile定义与分配

核函数内部需要定义Tile类型并分配片上内存：

```python
tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
tile_y = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2])
```

### 流水段与同步

计算逻辑需要放在`pl.section_vector()`上下文中，开启`auto_mutex=True`后，搬运与计算间的流水同步由框架按tile的mutex自动插入：

```python
with pl.section_vector():
    cur_x = tile_x.current()
    cur_y = tile_y.current()
    cur_out = tile_out.current()
    pl.load(cur_x, x, [0, 0])
    pl.load(cur_y, y, [0, 0])
    pl.add(cur_out, cur_x, cur_y)
    pl.store(out, cur_out, [0, 0])
```

### 其他规则

- 核函数支持接受Python的`int`、`bool`等标量参数
- 可以使用`auto_mutex=True`参数启用自动互斥锁插入

## 核函数的调用

PyPTO Pro中核函数通过 **bracket-launch** 语法发起，方括号内指定stream和block_dim（启动核数）：

```python
# 准备输入数据
a = torch.rand(64, 64, device="npu:0", dtype=torch.float16)
b = torch.rand(64, 64, device="npu:0", dtype=torch.float16)
out = torch.empty(64, 64, device="npu:0", dtype=torch.float16)

# bracket-launch：[stream, block_dim]
# None 表示默认 stream，num_cores 指定启动的核数
add_kernel[None, num_cores](a, b, out)
torch.npu.synchronize()
```

也可以省略方括号直接调用，此时以单核模式执行：

```python
add_kernel(a, b, out)
torch.npu.synchronize()
```

核函数的调用是异步的。首次调用时触发JIT编译，后续调用直接执行缓存的二进制文件，无需重复编译。

## Tiling参数化

`@pl.jit()`支持通过`tiling_key`参数实现Tiling参数化，在launch时通过字典选择不同的Kernel实例化，使每种模式各编一份专用Kernel（消除死分支、拿到最优指令）：

```python
from pypto_pro.runtime.tilingkey import TilingKeyField

class MyTilingKey:
    NeedAttnMask = TilingKeyField(bits=1, values=[0, 1])

@pl.jit(tiling_key=MyTilingKey)
def my_kernel(x: pl.Tensor[[64, 64], pl.DT_FP16], out: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...

# launch时通过字典选择实例化
my_kernel[None, num_cores, {"NeedAttnMask": 1}](x, out)
my_kernel[None, num_cores, {"NeedAttnMask": 0}](x, out)
```

`tiling_key`的完整说明（字段定义、`is_valid`校验、与TilingData的组合、运行时flag与tiling_key的选型对照表）请参考[多核切分与Tiling](tile_based_python_programming/multi_core_partitioning_and_Tiling.md#tiling-key--编译期特化而非运行时传值)。

## JIT配置选项

`@pl.jit()`装饰器支持以下配置选项：

| 选项 | 说明 | 默认值 |
|:---|:---|:---|
| arch | 目标架构，当前可选"a5"；None为自动检测当前受支持设备的架构 | None |
| auto_mutex | 是否启用自动互斥锁插入 | False |
| enable_print_debug | 是否启用设备侧调试打印 | None |
| timeout | 编译超时时间（秒） | 60 |
| name | 自定义Kernel名称，用于构建产物路径隔离 | None |
| tiling_key | Tiling键类型，用于Tiling参数化 | None |
| pipeline | PipelineConfig，用于自动预取流水变换 | None |
| datatype | 数据类型特化，用于同一Kernel支持多种数据类型 | None |

```python
@pl.jit(arch="a5", auto_mutex=True, timeout=200)
def my_kernel(x: pl.Tensor[[64, 64], pl.DT_FP16],
              out: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...
```

> [!NOTE]说明
> 建议优先使用JIT入参配置各类选项，避免在计算函数内部出现与数据流和计算不相关的代码。
