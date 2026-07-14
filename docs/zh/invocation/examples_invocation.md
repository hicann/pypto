# 样例运行

## 运行环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
cd examples/00_hello_world
python3 hello_world.py --run_mode=npu
```

更多示例请参考`examples/`目录下的示例代码。

## 结果查看

该基础样例运行成功后，在`${work_path}/output/`目录下生成编译和运行产物，相关产物包括[计算图](../tutorials/appendix/glossary.md)和[泳道图](../tutorials/appendix/glossary.md)，计算图和泳道图可通过PyPTO配套的ToolKit插件，在VS-CODE中查看并与代码关联，相关ToolKit使用请参考[快速入门-查看计算图](../tutorials/introduction/quick_start.md#查看计算图)、[快速入门-查看泳道图](../tutorials/introduction/quick_start.md#查看泳道图)

## 快速开始

以下是一个简单的PyPTO使用示例:

```python
import pypto
import torch
import torch_npu

shape = (1, 4, 1, 64)

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_kernel(
    x: pypto.Tensor([...], pypto.DT_FP32),
    y: pypto.Tensor([...], pypto.DT_FP32),
    out: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out[:] = x + y

if __name__ == "__main__":
    torch.npu.set_device(0)
    device = "npu:0"

    x = torch.rand(shape, dtype=torch.float32, device=device)
    y = torch.rand(shape, dtype=torch.float32, device=device)
    output = torch.empty(shape, dtype=torch.float32, device=device)

    # 执行计算并查看结果
    add_kernel(x, y, output)
    print(f"Output shape: {output.shape}")
```

- 可以直接通过查看输出张量的值查看运行结果

完整样例请参考：[hello_world.py](../../../examples/00_hello_world/hello_world.py)。
