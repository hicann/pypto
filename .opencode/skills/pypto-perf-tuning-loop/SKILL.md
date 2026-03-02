---
name: pypto-perf-tuning-loop
description: Iterative performance tuning workflow for PyPTO operators on Ascend NPU with profiling, parameter sweeps, and stop criteria. Use when users ask for PyPTO performance optimization, tile size tuning, before/after benchmarking, or repeated on-board tuning loops.
---

# PyPTO Performance Tuning Loop

## Purpose

Tune a PyPTO operator with a repeatable on-board loop:
1. establish baseline
2. profile bottlenecks
3. sweep parameters (small to large)
4. stop when performance no longer improves
5. deliver a before/after report with tuning direction

This follows PyPTO debug/perf guidance (`调试调优 -> 性能调优 / Matmul高性能编程`).

## Auto-Trigger Terms

Apply this skill when requests include:
- PyPTO 性能调优 / 上板调优 / benchmark
- tile size / set_cube_tile_shapes / set_vec_tile_shapes
- 泳道图 / bubble_analysis / perfetto
- 调优前后对比 / 提升多少 / 找方向

## TOP Tuning Principles

1. **Hotspot-first**: only tune kernels that dominate end-to-end time.
2. **Single-variable changes**: change one knob per round (keep others fixed).
3. **Same workload, same environment**: prompt/shape/device/warmup/rounds must match.
4. **Use averages, not one shot**: each candidate needs multiple rounds.
5. **Stop early by rule**: if no meaningful gain, stop and lock best config.
6. **Correctness guardrail**: performance numbers are valid only when output is correct.

## Workflow

Copy and track:

```text
Task Progress:
- [ ] Step 1: Define metric and fixed benchmark setup
- [ ] Step 2: Collect baseline (N rounds)
- [ ] Step 3: Enable perf trace and profile bottleneck
- [ ] Step 4: Run parameter sweep loop (small -> large)
- [ ] Step 5: Apply stop criteria and choose best config
- [ ] Step 6: Re-validate correctness and report before/after
```

## Step 1: Fixed Benchmark Setup

Lock these before tuning:
- device id (for example `TILE_FWK_DEVICE_ID=0`)
- model/script/shape bucket
- prompt and generation length (or fixed operator input tensor)
- warmup rounds and measured rounds
- metric (`tok/s`, latency, or kernel us)

Recommended:
- warmup: 1~3 rounds
- measure: >= 5 rounds

## Step 2: Baseline

Run baseline first and save:
- per-round values
- mean / std / min / max
- command line and commit hash (if available)

Output baseline table:

```markdown
| Config | Round values | Mean | Std |
|--------|--------------|------|-----|
| baseline | [...] | ... | ... |
```

## Step 3: Profile Bottleneck

Temporarily enable perf collection in `@pypto.frontend.jit`:

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": mode},
    debug_options={"runtime_debug_mode": 1}
)
```

Run on board, then analyze:
- `merged_swimlane.json`
- `machine_runtime_operator_trace.json`
- `bubble_analysis.log`

Use analyzer if available:

```bash
python3 .opencode/skills/pypto-operator-perf-autotune/scripts/analyze_performance.py output/output_<timestamp>
```

After analysis, restore debug options to normal.

## Step 4: Parameter Sweep Loop (Small -> Large)

Tune order (default):
1. `set_cube_tile_shapes`
2. `set_vec_tile_shapes`
3. loop tile (for dynamic axis, e.g. `bs_tile`)

Rule: only one parameter family changes at a time.

Example sweep strategy:
- cube sizes: 32 -> 64 -> 96 -> 128
- vec tiles: small -> medium -> large
- each candidate runs fixed rounds with same workload

## Step 5: Stop Criteria

Stop current sweep when either condition is met:

1. **No-improve streak**: 2 consecutive candidates improve < `min_gain`
2. **Regression threshold**: candidate is slower than current best by > `regress_tol`

Default thresholds:
- `min_gain = 1.0%`
- `regress_tol = 1.5%`

When stopped:
- lock best candidate
- move to next parameter family or finish

## Step 6: Confirm Direction and Report

Direction rules:
- if larger tile keeps improving -> direction is “increase”
- if medium beats both ends -> direction is “center optimum”
- if smaller wins -> direction is “decrease”

Mandatory report format:

```markdown
## Tuning Summary
- Metric: [tok/s or latency]
- Baseline: [mean ± std]
- Best: [mean ± std]
- Gain: [absolute + %]
- Winning config: [tile settings]
- Direction: [increase / decrease / center optimum]
- Stop reason: [no-improve streak / regression threshold]

## Evidence
- Rounds per config: [N]
- Key profile signals: [AIC/AIV utilization, bubble, control overhead]
- Correctness check: [pass/fail + method]
```

## Guardrails

- Do not compare numbers across different prompts/shapes.
- Do not keep `runtime_debug_mode=1` in final production path.
- If output correctness fails, discard that candidate regardless of speed.

## Additional Resources

- For concrete usage examples, see [examples.md](examples.md).
