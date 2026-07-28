# 算子编译

本节介绍PyPTO Pro算子的编译方法。PyPTO Pro支持两种编译方式：

- **JIT编译**：即时编译，在首次调用Kernel函数时自动触发，适合开发调试阶段
- **离线二进制编译**：提前编译生成二进制文件，适合生产部署场景

```{toctree}
:maxdepth: 1
:titlesonly:

JIT_compilation
offline_binary_compilation
```
