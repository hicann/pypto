# PyPTO 编程入门介绍样例 (Introduction)

本样例展示了 PyPTO 的编程入门介绍，适合初学者快速上手。

## 总览介绍

本样例涵盖了 PyPTO 编程中的基本语法介绍，包括：
- Tensor创建， 使用及计算。
- jit 编译与执行
- 循环与数据切分
- 条件与分支

## 代码结构

- **`add_direct.py`**: 入门示例1。
- **`add_scalar.py`**: 入门示例2。
- **`add_scalar_loop.py`**: 循环示例。
- **`add_scalar_loop_dyn_axis.py`**: 动态Shape示例。
- **`add_scalar_loop_multi_jit.py`**: 多jit示例。
- **`add_scalar_loop_dyn_axis_static_cond.py`**: 静态分支示例。
- **`add_scalar_loop_dyn_axis_dyn_cond.py`**: 动态分支示例。
- **`add_scalar_loop_dyn_axis_dyn_loop_cond.py`**: 动态分支示例2。
- **`add_scalar_loop_view_assemble.py`**: view/assemble静态分支示例。

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
```

### 执行脚本

```bash
# 运行所有示例
python3 add_direct.py
```

## 注意事项
- **环境**: 确保 `torch_npu` 已正确安装并能识别到昇腾显卡。
