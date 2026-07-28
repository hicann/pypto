# AiCore算子JIT编译基本用法

当开发者完成Kernel函数的编写后，通过`@pl.jit()`装饰器即可实现即时编译（JIT），无需手动执行编译命令。

## JIT编译流程

JIT编译的流程如下：

1. 首次调用被`@pl.jit()`装饰的函数时，PyPTO Pro解析函数体中的Tile定义、数据搬运和计算操作，构建计算图
2. 编译器对计算图进行多轮Pass优化，包括算子融合、调度优化、内存重用等
3. 编译器通过CodeGen生成针对NPU的优化代码，并缓存为二进制文件
4. 后续调用时直接加载缓存的二进制文件在NPU上执行，无需重复编译

## 基本用法

```python
import pypto_pro.language as pl
import torch
import torch_npu

@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])

# 首次调用触发JIT编译
device = "npu:0"
torch.npu.set_device(device)
a = torch.rand(64, 64, device=device, dtype=torch.float16)
b = torch.rand(64, 64, device=device, dtype=torch.float16)
out = torch.empty(64, 64, device=device, dtype=torch.float16)

add_kernel(a, b, out)        # 编译 + 执行
torch.npu.synchronize()

add_kernel(a, b, out)        # 仅执行，无编译开销
```

## 架构指定

通过`arch`参数可以指定目标NPU架构。当前可显式指定`"a5"`，也可以省略该参数以自动检测架构：

```python
# 指定A5架构
@pl.jit(arch="a5")
def kernel_a5(x, out):
    ...

# 自动检测（默认）
@pl.jit()
def kernel_auto(x, out):
    ...
```

| arch值 | 对应产品 |
|:---|:---|
| "a5" | Ascend 950PR/Ascend 950DT |
| None | 自动检测当前受支持设备的架构 |

## 编译产物

JIT编译完成后，编译产物默认缓存在`./build/<kernel_name>__<arch>/`目录下（`<arch>`为目标架构，如`a5`）。每个Kernel在该目录下拥有独立的子目录，若使用了TilingKey，则每个具体key further对应`tk_<packed>/`子目录。主要产物包括：

- **kernel.cpp**：CodeGen生成的Device侧C++源码，包含Kernel的计算逻辑实现。
- **call_kernel.cpp / call_kernel.so**：Host侧Launcher源码及其编译后的共享库，负责参数打包和Kernel下发。
- **tiling头文件**（`*_tiling.h`）：当Kernel包含TilingData参数时生成，描述tiling结构体的C布局。

> [!NOTE]说明
> 默认情况下，编译成功后`kernel.cpp`和`call_kernel.cpp`等中间源文件会保留在产物目录中，便于调试。若需查看Device侧生成的代码，可直接阅读`kernel.cpp`。
