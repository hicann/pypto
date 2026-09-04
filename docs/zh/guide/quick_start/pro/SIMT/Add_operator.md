# Add算子快速入门

本示例是一个入门实践，基于PyPTO Pro SIMT实现Add算子，帮助您快速上手。它完整呈现了Kernel函数定义、SIMT函数定义、Tile配置、数据搬运、计算及运行的全流程，助您建立整体认知。开始前，请先参考[环境准备](../../../../install/prepare_environment.md)完成基础环境搭建。

## Add算子

**功能介绍**：Add算子的数学表达式为$dst_i = src_i + delta$，计算逻辑为每个SIMT线程根据线程编号处理一个元素。

## 算子设计

| 模块 | 说明 |
|:---|:---|
| Kernel函数定义 | 通过@pl.jit声明JIT编译目标 |
| SIMT函数定义 | 通过@pl.simt.function定义SIMT入口函数，并通过max_threads设置单个线程块的最大线程数 |
| Tile定义 | 使用[pl.TileType](../../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md)定义片上Tile的形状、数据类型和目标内存空间 |
| Tile分配 | 使用[pl.make_tile](../../../../api/pro_api/SIMD-API/operation/resource_management/make_tile.md)分配片上内存 |
| 数据搬入 | 通过[pl.load](../../../../api/pro_api/SIMD-API/operation/memory_data_movement/load.md)将GM数据搬入UB Tile |
| 数据计算 | 通过[pl.simt.launch](../../../../api/pro_api/SIMT-API/execution/launch.md)启动一维线程块，每个线程通过[pl.simt.thread_idx](../../../../api/pro_api/SIMT-API/execution/thread_idx.md)获取线程编号并更新一个Tile元素 |
| 流水同步 | 通过pl.system.sync_src和pl.system.sync_dst描述MTE2、SIMT Vector流水和MTE3之间的数据依赖 |
| 数据搬出 | 通过[pl.store](../../../../api/pro_api/SIMD-API/operation/memory_data_movement/store.md)将UB Tile结果写回GM |

## 算子代码实现

```python
import os

import pypto_pro.language as pl
import torch
import torch_npu


THREADS = 256
TILE_BYTES = THREADS * 4


@pl.simt.function(max_threads=THREADS)
def add_delta(data: pl.Tile[[1, THREADS], pl.DT_FP32], delta: pl.DT_FP32) -> None:
    tid = pl.simt.thread_idx().x
    data[0, tid] = data[0, tid] + delta


@pl.jit(arch="a5")
def simt_add_kernel(
    src: pl.Tensor[[1, THREADS], pl.DT_FP32],
    dst: pl.Tensor[[1, THREADS], pl.DT_FP32],
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(
        shape=[1, THREADS],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    data = pl.make_tile(tile_type, addr=0x0000, size=TILE_BYTES)

    with pl.section_vector():
        pl.load(data, src, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        pl.simt.launch(add_delta, threads=THREADS, args=(data, delta))

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(dst, data, [0, 0])


# Host端调用
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)

delta = 2.5
src = torch.arange(THREADS, dtype=torch.float32, device=device).reshape(1, THREADS)
dst = torch.empty_like(src)

simt_add_kernel(src, dst, delta)
torch.npu.synchronize()

torch.testing.assert_close(dst, src + delta, rtol=0, atol=0)
print("SIMT kernel passed!")
```

> [!NOTE]说明
>
> - pl.Tensor[[1, THREADS], pl.DT_FP32]中的[1, THREADS]为张量形状，pl.DT_FP32为数据类型。
> - pl.TileType的target_memory=pl.MemorySpace.Vec表示Tile分配在Vector核的UB上。
> - @pl.simt.function(max_threads=THREADS)定义可由外层Kernel启动的SIMT入口函数，本例实际启动的线程数与max_threads均为256。
> - pl.simt.thread_idx().x返回当前线程在线程块X维的编号，取值范围为[0, 256)，每个线程访问Tile中的一个元素。
> - pl.load通过MTE2流水将输入搬入UB，pl.simt.launch在SIMT Vector流水上更新Tile，pl.store通过MTE3流水将结果搬回GM。
> - 不同流水之间存在数据依赖，因此需要成对调用pl.system.sync_src和pl.system.sync_dst显式同步。
> - 如需进一步了解PyPTO Pro的SIMT编程模型，请参阅[SIMT编程模型](../../../programming_guide/pro/programming_paradigm/simt_programming.md)。
