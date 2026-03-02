# Examples: PyPTO Performance Tuning Loop

## Example 1: SDPA Tile Sweep

**User request**
> 给 SDPA 做性能调优，跑前后对比

**Expected execution**
1. fix benchmark setup (`max-new-tokens`, prompt, device)
2. baseline 5 rounds
3. sweep cube tile: 32 -> 64 -> 96 -> 128
4. each candidate run 5 rounds
5. stop after 2 consecutive gains < 1%
6. output before/after and tuning direction

**Expected output format**

```markdown
## Tuning Summary
- Baseline: 22.80 ± 0.27 tok/s
- Best: 23.42 ± 0.19 tok/s
- Gain: +0.62 tok/s (+2.72%)
- Winning config: cube=[64,64], vec=[1,8,16,128]
- Direction: increase then plateau
- Stop reason: no-improve streak (2 rounds)
```

## Example 2: Dynamic Axis Loop Tile

**User request**
> dynamic bs 算子怎么调 bs_tile？

**Expected execution**
1. keep cube/vec fixed
2. sweep `bs_tile`: 2 -> 4 -> 8 -> 16
3. compare mean latency of 5 rounds
4. pick best and verify correctness

**Expected decision**
- if `bs_tile=8` best and `16` regresses >1.5%, stop
- direction: medium tile is optimal (center optimum)

## Example 3: End-to-End + Kernel Evidence

**User request**
> 除了 tok/s，还要说明为什么快了

**Expected execution**
1. run end-to-end throughput benchmark
2. enable `runtime_debug_mode=1` once for profile
3. analyze swimlane/bubble report
4. correlate improvement with AIC/AIV utilization and wait reduction

**Expected output format**

```markdown
## Evidence
- End-to-end: +3.1%
- AIC utilization: 89% -> 93%
- AIV wait predecessor: reduced
- Control overhead: unchanged (low)
```
