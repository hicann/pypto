# Custom-Op Export and Deployment Demos

An end-to-end demo: author a pypto kernel as a `torch.ops` custom op, export it as an ONNX artifact, and
build it into a deployable `libcust_opapi.so`. The demo side only authors and runs — the export session,
the discovery logic and the `.so` build all come from
[`tools/exported_custom_op_litenpu`](../../../tools/exported_custom_op_litenpu) (under `<repo root>/tools`, imported as
the `exported_custom_op_litenpu` package).

## Layout

- `onnx/` — ONNX export demos (one so far: `add`), one folder per demo. The per-demo file breakdown, the CLI flags and the authoring rules are in [`onnx/README_en.md`](onnx/README_en.md).
- The system tests live in [`python/tests/st/examples/exported_custom_op_litenpu/`](../../../python/tests/st/examples/exported_custom_op_litenpu): `test_build_so_loads.py` proves the built `.so` is really loadable, and `test_demo_sweep.py` sweeps every demo's export and build stages.

🔴 **One demo per process.** A demo's modules are plain top-level names (`kernel`, `op`, `model`), reached
by putting the demo's own folder on `sys.path`. Two demos in one interpreter would therefore share those
names — the first one imported wins, and the second silently reuses it and never registers its op. Demos
also claim process-global `torch.ops.pypto.*` qualnames, several of which repeat across demos. Run each
demo in its own process.

## Running the demo

```bash
python3 onnx/add/export_demo.py /tmp/add.onnx         # export only, never runs
python3 ../../../tools/exported_custom_op_litenpu/deploy/build_so_and_setup.py /tmp/add.onnx # build + place the .so
python3 onnx/add/run_demo.py                          # run on cpu, never exports
```

## Environment setup

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0          # pick a free device id from `npu-smi info` for --device=npu
```

Building the `.so` on a trimmed CANN environment also needs the GE external headers on the include path:

```bash
export PYPTO_EXTRA_INCLUDE_DIRS=<ge>/inc/graph_metadef/external
```
