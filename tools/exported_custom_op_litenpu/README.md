# exported_custom_op_litenpu/

导出样例所导入的仓库工具库（`from exported_custom_op_litenpu.<sub>.…`）。它不属于 `pypto` 包，而是放在
仓库的 `tools/` 下；样例把 `<repo>/tools` 加入 `sys.path` 来导入它，并各自发起自己的导出调用。

| 子目录 | 作用 |
|---|---|
| `common/` | pypto 自定义算子节点发现 —— `discovery.find_pypto_nodes` / `load_model`。 |
| `export/` | ONNX 导出会话 —— `onnx_export_session` / `resolve_pypto_export_opset` / `finalize_pypto_onnx_nodes`。 |
| `deploy/` | 构建并放置 `.so` —— `build_so_from_model` / `setup_onnx_custom_op_so`，以及 `deploy/vendors/` 下的代码生成与随附 C++。 |
| `run/` | 共享的运行 CLI，被每个带设备阶梯的 `run_demo.py` 导入 —— `add_run_args` / `validate_args` / `scenario_label` / `print_outputs`。 |

## 测试

单元测试位于 `python/tests/ut/tools/exported_custom_op_litenpu/`，无需 NPU 即可运行：

```bash
pytest python/tests/ut/tools/exported_custom_op_litenpu -q
```

其中带 `cpp_codegen` 标记的编译测试还需要：pybind11 头文件、Python 开发头文件与库（`python3-config --cflags --ldflags --embed`；部分发行版需安装 `python3-dev`），以及一个 C++17 编译器（`g++`、`clang++`，或环境变量 `CXX`）。所需前置条件缺失时，相应测试会自动跳过。

## 第三方

nlohmann/json 3.11.3（MIT，<https://github.com/nlohmann/json>）以纯头文件方式被生成的算子 executor 与 onnx 插件的 ParseParam 编译单元使用：读取 kernel 的 `*_aiv.json` 启动侧车文件（`blockDim` / `kernelName` / `kernelBin` / `workspaceSize`），并解析 ONNX 属性 JSON。`deploy/codegen._resolve_nlohmann_include_dir()` 负责解析出一个目录 `D`，使 `D/nlohmann/json.hpp` 为文件 —— 先查已安装的 pypto 的 `lib/framework/3rd/include`，其次是 `PYPTO_THIRD_PARTY_PATH` 或仓库的 `third_party_path/` —— 随后 `deploy/build.py` 将其前置到编译 `-I` 路径。找不到包含根属于致命错误：构建会以 `nlohmann/json.hpp not found` 中止。由于该根目录被前置，其下随附的任何头文件都会暴露给算子构建，并可能遮蔽同名的 CANN/GE 头文件；pypto 核心构建本身也基于该根编译，因此其内容是已知兼容的。
