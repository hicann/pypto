# Add算子快速入门（SIMD）

## 任务与目标

本节将详细介绍如何使用PyPTO Pro框架实现一个简单的Add算子，并通过测试用例验证其正确性。通过本节的学习，您将了解如何使用PyPTO Pro的API构建自定义Add算子。

本示例对两个形状均为[64, 64]的FP16张量执行逐元素加法。开始前，请先参考[环境准备](../../../install/prepare_environment.md)完成基础环境搭建。

## 算子设计规格

**表1** Add算子设计规格

| name | shape | data type | format |
| :---: | :---: | :-------: | :----: |
| input_x | [64,64] | float16 | ND |
| input_y | [64,64] | float16 | ND |
| output | [64,64] | float16 | ND |

- 数学表达式

  给定输入张量***X***和***Y***，逐元素相加得到输出张量***Z***：
  $$
  Z_{i,j} = X_{i,j} + Y_{i,j}
  \qquad \text{for } 0 \le i < 64, \; 0 \le j < 64
  $$

- 使用的主要接口

  基础搬运接口：[pypto_pro.language.load](../../../api/pro_api/SIMD-API/memory_data_movement/load.md)、[pypto_pro.language.store](../../../api/pro_api/SIMD-API/memory_data_movement/store.md)

  基础计算接口：[pypto_pro.language.add](../../../api/pro_api/SIMD-API/tile_computation/elementwise/add.md)

  资源管理接口：[pypto_pro.language.TileType](../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md)、[pypto_pro.language.make_tile_group](../../../api/pro_api/SIMD-API/resource_management/make_tile_group.md)

## 导入PyPTO Pro模块

在开始实现Add算子之前，需要导入PyPTO Pro、PyTorch和torch_npu模块，并配置待使用的NPU设备。

```python
import os

import pypto_pro.language as pl
import torch
import torch_npu

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
```

## 核心代码逻辑

```python
@pl.jit(auto_mutex=True)
def add_kernel(a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16],
               out: pl.Tensor[[64, 64], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

> [!NOTE]说明
>
> - pl.Tensor[[64, 64], pl.DT_FP16]中的[64, 64]为张量形状，pl.DT_FP16为数据类型。
> - pl.TileType的target_memory=pl.MemorySpace.Vec表示Tile分配在Vector核的UB上。
> - pl.make_tile_group通过mutex_ids分配缓冲，框架在auto_mutex=True时自动插入同步，开发者无需手写sync_src/sync_dst。
> - tile_group.current()获取当前可用缓冲。
> - pl.section_vector()标记后续代码在Vector流水单元上执行。
> - AI处理器对FP16和BF16有原生硬件加速，建议在算子开发中优先考虑这些数据类型。
> - 如需进一步了解PyPTO Pro的SIMD编程模型，请参阅[SIMD编程范式](../../programming_guide/pro/programming_paradigm/SIMD/programming_paradigm.md)。

## 测试用例

测试用例使用PyTorch Tensor准备输入，通过PyPTO Pro Kernel完成计算，并与PyTorch逐元素加法的结果进行比较。

```python
def test_add_kernel():
    torch.npu.set_device(ST_DEVICE)
    torch.manual_seed(0)

    a = torch.rand(64, 64, device=ST_DEVICE, dtype=torch.float16)
    b = torch.rand(64, 64, device=ST_DEVICE, dtype=torch.float16)
    out = torch.empty(64, 64, device=ST_DEVICE, dtype=torch.float16)

    add_kernel(a, b, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
```

## 编译与执行

将上述代码按顺序保存为`add_example.py`，在已安装PyPTO Pro的环境中运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pytest -q add_example.py::test_add_kernel
```

用例执行成功后，pytest显示`1 passed`。
