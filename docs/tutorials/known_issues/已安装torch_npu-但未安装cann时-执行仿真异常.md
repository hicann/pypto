# 已安装torch\_npu，但未安装cann时，执行仿真异常

## 问题现象描述

在仿真环境中执行算子时，出现失败，报错信息如下。

```txt
ImportError: libhccl.so: cannot open shared object file: No such file or directory
```

## 问题原因

当程序启动时，torch（版本\>2.5）会自动加载所有名为“torch.backends”的扩展（例如 torch npu）。如果环境中已安装了torch\_npu但未安装CANN，由于找不到依赖项，将会引发异常。

## 处理步骤

在执行算子之前，添加以下环境变量，可以避免上述异常。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

