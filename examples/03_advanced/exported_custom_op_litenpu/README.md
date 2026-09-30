# 自定义算子导出与部署样例

端到端样例：将 pypto kernel 作为 `torch.ops` 自定义算子来编写，导出为 ONNX 制品，并构建成可部署的
`libcust_opapi.so`。样例侧只负责编写与运行 —— 导出会话、发现逻辑与 `.so` 构建都来自
[`tools/exported_custom_op_litenpu`](../../../tools/exported_custom_op_litenpu)（在 `<repo root>/tools` 下，以
`exported_custom_op_litenpu` 包导入）。

## 目录结构

- `onnx/` —— ONNX 导出样例（当前一个：`add`），每个样例一个目录。每个样例的文件分工、命令行参数与编写规则见 [`onnx/README.md`](onnx/README.md)。
- 系统测试位于 [`python/tests/st/examples/exported_custom_op_litenpu/`](../../../python/tests/st/examples/exported_custom_op_litenpu)：`test_build_so_loads.py` 验证构建出的 `.so` 是真实可加载的，`test_demo_sweep.py` 遍历每个样例的导出与构建阶段。

🔴 **一个进程只跑一个样例。** 样例的各模块使用的是普通顶层名字（`kernel`、`op`、`model`），依靠把样例自身目录加入 `sys.path` 来访问。因此同一解释器中的两个样例会共享这些名字 —— 先导入的那个胜出，后一个会静默复用它、从而根本不注册自己的算子。样例还会占用进程全局的 `torch.ops.pypto.*` 限定名，其中若干在不同样例间重复。请让每个样例在各自独立的进程中运行。

## 运行样例

```bash
python3 onnx/add/export_demo.py /tmp/add.onnx         # 导出，从不运行
python3 ../../../tools/exported_custom_op_litenpu/deploy/build_so_and_setup.py /tmp/add.onnx # 构建并放置 .so
python3 onnx/add/run_demo.py                          # 在 cpu 上运行，从不导出
```

## 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0          # 从 `npu-smi info` 中选一个空闲设备号，供 --device=npu 使用
```

在裁剪版 CANN 环境上构建 `.so` 还需要把 GE 外部头文件加入包含路径：

```bash
export PYPTO_EXTRA_INCLUDE_DIRS=<ge>/inc/graph_metadef/external
```
