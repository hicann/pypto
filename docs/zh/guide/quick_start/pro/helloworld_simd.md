# HelloWorld（SIMD）

## 任务与目标

本节将详细介绍如何使用PyPTO Pro框架实现一个简单的HelloWorld算子，并通过Host端直接调用验证其运行结果。通过本节的学习，您将了解Kernel函数定义、JIT编译、Device端打印和Host端调用的基本流程。

本示例在NPU上打印`Hello World!!!`，并将输出张量的第一个元素设置为1。开始前请参考[环境准备](../../../install/prepare_environment.md)完成基础环境搭建。

## 算子设计规格

**表1** HelloWorld算子设计规格

| name | shape | data type | format |
| :---: | :---: | :-------: | :----: |
| output | [1] | int32 | ND |

- 功能表达式

  HelloWorld Kernel在Device端打印指定字符串，并写入输出张量：
  $$
  output_0 = 1
  $$

- 使用的主要接口

  调试接口：[pypto_pro.language.printf](../../../api/pro_api/Utils-API/debugging/printf.md)

  控制流接口：[pypto_pro.language.section_vector](../../../api/pro_api/SIMD-API/controlflow/section_vector.md)

  Kernel定义接口：[pypto_pro.language.jit](../../programming_guide/pro/development/compilation_and_execution/JIT_compilation.md)

## 导入PyPTO Pro模块

Host端通过PyTorch张量准备输入输出数据，直接调用Kernel函数完成计算。

```python
import os
import torch
import torch_npu
import pypto_pro.language as pl
```

## 核心代码逻辑

HelloWorld Kernel通过[@pypto_pro.language.jit()](../../programming_guide/pro/development/compilation_and_execution/JIT_compilation.md)装饰器定义Kernel函数，使用[pypto_pro.language.printf](../../../api/pro_api/Utils-API/debugging/printf.md)在Device端打印字符串。[pypto_pro.language.section_vector()](../../../api/pro_api/SIMD-API/controlflow/section_vector.md)用于声明该段代码在Vector核上执行，其中pypto_pro.language.printf由Scalar流水执行。

```python
@pl.jit()
def hello_world_kernel(out: pl.Tensor[[1], pl.DT_INT32]):
    with pl.section_vector():
        pl.printf("Hello World!!!\n")
        pl.setval(out, 0, 1)

# Host端调用

device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)

out = torch.zeros(1, device=device, dtype=torch.int32)

hello_world_kernel(out)
torch.npu.synchronize()

print(f"kernel finished, out[0] = {out[0].item()}")
```

> [!NOTE]说明
>
> - PyPTO Pro的Kernel函数通过@pypto_pro.language.jit()装饰器标记为JIT编译目标。首次调用时触发编译；在同一Python进程中，同一Kernel对象以相同编译签名再次调用时复用编译结果。
> - pypto_pro.language.printf通过设备侧打印机制输出，具体查看位置由运行环境的CANN日志配置决定。printf仅用于调试，生产环境应移除。

## 编译与执行

将上述代码按顺序保存为`hello_world.py`。如果当前Shell尚未加载Ascend Toolkit环境变量，请先执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

环境准备完成后，直接运行：

```bash
python3 hello_world.py
```

运行成功后，Host端输出`kernel finished, out[0] = 1`。设备侧的`Hello World!!!`请按当前部署环境的CANN日志配置查看。

> [!NOTE]说明
>
> 如需进一步了解PyPTO Pro的SIMD编程模型，请参阅[SIMD编程范式](../../programming_guide/pro/programming_paradigm/SIMD/programming_paradigm.md)。
