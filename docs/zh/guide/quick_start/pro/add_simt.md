# Add算子快速入门（SIMT）

## 任务与目标

本示例是一个入门实践，基于PyPTO Pro SIMT实现Add算子，帮助您快速上手。它完整呈现了Kernel函数定义、SIMT函数定义、Tile配置、数据搬运、计算及运行的全流程，助您建立整体认知。开始前，请先参考[环境准备](../../../install/prepare_environment.md)完成基础环境搭建。

## 算子设计规格

**表1** Add算子设计规格

| name | shape | data type | format |
| :---: | :---: | :-------: | :----: |
| input_src | [1,256] | float32 | ND |
| delta | - | float32 | Scalar |
| output | [1,256] | float32 | ND |

- 数学表达式

  给定输入张量***Src***和标量***delta***，逐元素相加得到输出张量***Dst***：
  $$
  Dst_i = Src_i + delta
  \qquad \text{for } 0 \le i < 256
  $$

- 使用的主要接口

  基础搬运接口：[pypto_pro.language.load](../../../api/pro_api/SIMD-API/memory_data_movement/load.md)、[pypto_pro.language.store](../../../api/pro_api/SIMD-API/memory_data_movement/store.md)

  SIMT执行接口：[SIMT索引调用](../../programming_guide/pro/development/vector_computation/simt_computation.md)、[pypto_pro.language.simt.thread_idx](../../../api/pro_api/SIMT-API/execution/thread_idx.md)

  资源管理接口：[pypto_pro.language.TileType](../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md)、[pypto_pro.language.make_tile](../../../api/pro_api/SIMD-API/resource_management/make_tile.md)


## 导入PyPTO Pro模块

在开始实现Add算子之前，需要导入PyPTO Pro、PyTorch和torch_npu模块，并配置待使用的NPU设备以及线程数。

```python
import os

import pypto_pro.language as pl
import torch
import torch_npu

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

THREADS = 256
TILE_BYTES = THREADS * 4
```

## 核心代码逻辑

```python
@pl.vector_function(mode="simt", max_threads=THREADS)
def add_delta(data, delta: pl.DT_FP32):
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

        add_delta[THREADS](data, delta)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(dst, data, [0, 0])
```

> [!NOTE]说明
>
> - pl.Tensor[[1, THREADS], pl.DT_FP32]中的[1, THREADS]为张量形状，pl.DT_FP32为数据类型。
> - pl.TileType的target_memory=pl.MemorySpace.Vec表示Tile分配在Vector核的UB上。
> - @pl.vector_function(mode="simt", max_threads=THREADS)定义可由外层Kernel启动的SIMT入口函数，本例实际启动的线程数与max_threads均为256。
> - pl.simt.thread_idx().x返回当前线程在线程块X维的编号，取值范围为[0, 256)，每个线程访问Tile中的一个元素。
> - pl.load通过MTE2流水将输入搬入UB，`add_delta[THREADS](data, delta)`在SIMT Vector流水上更新Tile，pl.store通过MTE3流水将结果搬回GM。
> - 不同流水之间存在数据依赖，因此需要成对调用pl.system.sync_src和pl.system.sync_dst显式同步。
> - 如需进一步了解PyPTO Pro的SIMT编程模型，请参阅[SIMT编程范式](../../programming_guide/pro/programming_paradigm/SIMT/programming_paradigm.md)。

## 测试用例

测试用例使用PyTorch Tensor准备输入，通过PyPTO Pro Kernel完成计算，并与PyTorch逐元素加法的结果进行比较。

```python
def test_simt_add_kernel():
    torch.npu.set_device(ST_DEVICE)

    delta = 2.5
    src = torch.arange(
        THREADS,
        dtype=torch.float32,
        device=ST_DEVICE,
    ).reshape(1, THREADS)
    dst = torch.empty_like(src)

    simt_add_kernel(src, dst, delta)
    torch.npu.synchronize()

    torch.testing.assert_close(dst, src + delta, rtol=0, atol=0)
```

## 编译与执行

将上述代码按顺序保存为`simt_add_example.py`，在已安装PyPTO Pro的环境中运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pytest -q simt_add_example.py::test_simt_add_kernel
```

用例执行成功后，pytest显示`1 passed`。
