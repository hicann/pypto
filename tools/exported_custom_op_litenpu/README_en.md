# exported_custom_op_litenpu/

The repo tool library the export demos import (`from exported_custom_op_litenpu.<sub>.…`) by putting `<repo>/tools`
on `sys.path`. Not part of the `pypto` package — it lives under `tools/`, and each demo owns its own export call.

| Subfolder | Role |
|---|---|
| `common/` | Pypto custom-op node discovery — `discovery.find_pypto_nodes` / `load_model`. |
| `export/` | The ONNX export session — `onnx_export_session` / `resolve_pypto_export_opset` / `finalize_pypto_onnx_nodes`. |
| `deploy/` | Build + place the `.so` — `build_so_from_model` / `setup_onnx_custom_op_so`, plus the codegen and bundled C++ under `deploy/vendors/`. |
| `run/` | The shared run CLI imported by every `run_demo.py` with a device ladder — `add_run_args` / `validate_args` / `scenario_label` / `print_outputs`. |

## Tests

Unit tests live in `python/tests/ut/tools/exported_custom_op_litenpu/`, and run without an NPU:

```bash
pytest python/tests/ut/tools/exported_custom_op_litenpu -q
```

The `cpp_codegen`-marked compile tests there additionally need pybind11 headers, the Python development headers and library (`python3-config --cflags --ldflags --embed`; on some distros install `python3-dev`), and a C++17 compiler (`g++`, `clang++`, or `CXX`). Each skips itself when its own prerequisite is missing.

## Third-party

nlohmann/json 3.11.3 (MIT, <https://github.com/nlohmann/json>) is consumed header-only by the generated op executor and the onnx-plugin ParseParam translation unit: it reads the kernel's `*_aiv.json` launch sidecar (`blockDim` / `kernelName` / `kernelBin` / `workspaceSize`) and parses the ONNX attribute JSON. `deploy/codegen._resolve_nlohmann_include_dir()` resolves an include root `D` such that `D/nlohmann/json.hpp` is a file — installed pypto's `lib/framework/3rd/include` first, then `PYPTO_THIRD_PARTY_PATH` or the repo's `third_party_path/` — and `deploy/build.py` prepends it to the compile `-I` path. A missing root is fatal: the build aborts with `nlohmann/json.hpp not found`. Because that root is prepended, any header vendored under it is exposed to the op build and could shadow a same-named CANN/GE header; the pypto core build already compiles against the same root, so its contents are known-compatible.
