---
name: tune-incore
description: PyPTO 算子核内性能调优技能。通过分析单 task 的实现指令及 operation，完成核内的性能调优，包括指令级优化、核内流水优化、特殊 Shape 处理等。当用户需要进行核内性能调优、单 task 耗时分析、指令级优化时使用此技能。触发词：核内性能调优、单 task 优化、指令级优化、核内流水、Operation 实现优化。
---

# PyPTO 算子核内性能调优

## 概述

核内性能调优通过分析单 task 的实现指令及 operation，完成核内的性能调优。适用于深度性能调优后仍需要进一步优化的场景。

## 前置条件

1. **完成深度性能调优**：泳道图分析和合图调优已完成
2. **精度校验通过**：确保算子计算正确
3. **识别出单 task 瓶颈**：通过泳道图定位到耗时较长的 task

## 调优方向

### 1. 特殊 Shape 处理

#### 1.1 小 Shape 矩阵乘优化

当矩阵 Shape 较特殊时，可以使用 Vector 操作提前处理输入矩阵。

**案例**：左右矩阵 Shape 分别为 (884736, 16) 和 (16, 16) 的矩阵乘

```python
def matmul_kernel(a, b, out):
    # 构造 c：将四个重复的右矩阵在对角线拼成 (64, 64)
    pypto.set_vec_tile_shapes(64, 64)
    d = pypto.full([16, 16], 0.0, pypto.DT_BF16)
    c1 = pypto.concat([b, d, d, d], 1)
    c2 = pypto.concat([d, b, d, d], 1)
    c3 = pypto.concat([d, d, b, d], 1)
    c4 = pypto.concat([d, d, d, b], 1)
    c = pypto.concat([c1, c2, c3, c4], 0)

    # a 变形
    a = pypto.reshape(a, [221184, 64])

    # 矩阵乘
    pypto.set_pass_options(cube_l1_reuse_setting={-1: 9})
    pypto.set_cube_tile_shapes([512, 512], [64, 64], [64, 64], True)
    e = pypto.matmul(a, c, pypto.DT_BF16)
    e = pypto.reshape(e, [884736, 16])
    pypto.assemble(e, [0, 0], out)
```

**效果**：从 500us 优化到 40us

### 2. L2 Cache 策略

通过 `set_cache_policy` 控制 Tensor 是否经过 L2 Cache，减少 Cache 争用，提升数据搬运效率。

#### 2.1 API 说明

```python
tensor.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
```

- **API 文档**：`docs/api/tensor/pypto-Tensor-set_cache_policy.md`
- **当前 Python 可用策略**：仅 `CachePolicy.NONE_CACHEABLE`（C++ 层还有 `PREFETCH`，但 Python API 未暴露）
- **效果**：标记后该 Tensor 的数据访问将绕过 L2 Cache，直接访问主存（HBM）

#### 2.2 适用场景

根据官方文档，以下两类数据适合设置 `NONE_CACHEABLE`：

1. **只读一次的权重矩阵**：类似于 weight 这种常量，算子仅从内存读取一次、不复用，没有必要占用 L2 Cache 空间
2. **过大的输出 Tensor**：输出 shape 过大时，下层算子最先使用的内存不是上层最后的输出结果，进 L2 后反而触发回写导致性能恶化

#### 2.3 调优策略

**逐个尝试法（适用于简单算子）**：对候选 Tensor 逐个设置 NONE_CACHEABLE，每次实测对比。

**批量设置法（适用于融合算子）**：当算子包含多个大型权重矩阵时，应考虑**同时对所有权重设置 NONE_CACHEABLE**。单独对某个权重设置可能因 L2 Cache 争用反而恶化，但全部绕过 L2 后可释放 Cache 容量给 KV Cache、中间激活等频繁访问的数据。

#### 2.4 ⛔ 注意事项

1. **输入 Tensor（hidden_states、residual 等）通常不适合 NONE_CACHEABLE**：这些 Tensor 虽然只读一次，但数据量小，L2 Cache 的硬件预取已经足够高效，绕过反而增加延迟
2. **输出 Tensor（output、residual_out）不适合 NONE_CACHEABLE**：输出需要写入主存，绕过 L2 会增加写回开销
3. **单独对某个大权重设置可能无效甚至恶化**：在融合算子中，单独绕过某个权重可能打破 L2 Cache 的整体平衡，导致其他权重访问变慢
4. **必须实测验证**：Cache 策略的效果高度依赖算子的数据访问模式和硬件状态，无法仅凭理论判断

**🔥 案例**：[权重矩阵批量 NONE_CACHEABLE](cases/weight-none-l2-cacheable.md)（Pangu 7B Fused Layer，5 个权重同时设置，437→354 us，-19.1%，含 5 轮迭代失败分析）

### 3. 增加冗余计算避免冗余依赖

通过增加冗余计算来避免冗余依赖和搬运。

**案例**：GLM MoE Fusion

```python
# 将 e_score_bias_2d 复制 tile_batch 份后进行 cast 操作
# 使每一份的 cast 都和对应 batch 的其他操作进行了合图
# 避免一对多的子图依赖，减少调度开销和搬运

e_score_bias_2d_tile = pypto.tensor([tile_batch, ne], e_score_bias_2d.dtype, "e_score_bias_2d_tile")
for tmp_idx in range(tile_batch):
    pypto.assemble(e_score_bias_2d, [tmp_idx, 0], e_score_bias_2d_tile)
e_score_bias_2d_cast = pypto.cast(e_score_bias_2d_tile, tile_logits_fp32.dtype)
```

### 4. 尾轴长度优化

尽量避免处理尾轴长度较小的 Tensor。

**解决方案**：
- 使用 concat、transpose 或 reshape 等 Operation 来增大尾轴
- 设置较大的 TileShape

### 5. TileOperation 实现检查

当进行上述优化后算子性能仍然较差时，需要考虑 TileOperation 本身实现是否较差。

**排查方法**：
1. 构造单独 Operation 的用例
2. 与 Ascend C 小算子的性能对比
3. 确认性能较差后检查是否使用了更优的指令

## 调优检查清单

**⛔ 必须按以下清单逐项执行。每项标记为 ✅已尝试 或 ❌已失败（附原因），禁止跳过。完整优化点信息参考 [shared/optimization_catalog.md](../shared/optimization_catalog.md)。**

**优化优先级**：
1. ⭐⭐⭐ **P0 - 特殊 Shape 处理** → 详见 [I-1]
2. ⭐⭐ **P1 - L2 Cache + 依赖与搬运优化** → 详见 [I-2][I-3][I-4]
3. ⭐ **P2 - 实现检查** → 详见 [I-5]

**🔥 P0 - 特殊 Shape [I-1]**：
- [ ] [I-1] Matmul 的 Shape 是否特殊（如 M 很大 N 很小）
- [ ] 是否可以用 Vector 预处理构造标准 Shape

**🔥 P1 - L2 Cache + 依赖与搬运 [I-2~I-4]**：
- [ ] [I-2] 只读一次的大型权重矩阵是否设置了 L2 Cache 策略（`NONE_CACHEABLE`）；融合算子中应对所有权重同时设置，避免 L2 争用失衡 → **🔥 案例**：[权重矩阵批量 NONE_CACHEABLE](cases/weight-none-l2-cacheable.md)（-19.1%）
- [ ] [I-3] 是否存在一对多的子图依赖（可通过冗余计算消除）
- [ ] [I-4] 尾轴是否过小（< 32B 对齐）

**P2 - 实现检查 [I-5]**：
- [ ] [I-5] 单个 Operation 是否与 Ascend C 对比过性能


## 常见问题

### Q1: 何时需要进行核内性能调优？

A: 当深度性能调优后，泳道图显示某个或某些 task 耗时明显过长，且无法通过 Stitch、TileShape、合图等方式优化时。

### Q2: 如何判断 Operation 实现效率低？

A:
1. 构造单独 Operation 的测试用例
2. 与 Ascend C 小算子性能对比
3. 如果差距明显，说明 Operation 实现可能需要优化

### Q3: 增加冗余计算会影响精度吗？

A: 不会。冗余计算是指增加一些不影响最终结果的计算（如复制数据），目的是优化调度和合图，不会改变计算逻辑。

## TileOperation 检查流程

对核内每个 TileOperation，按以下流程检查效率：

1. **检查操作数连续性**：输入 tensor 是否在内存中连续，不连续需先调用 `pypto.reshape` 或 `pypto.transpose` 调整
2. **检查数据搬运方向**：Gather → 从 HBM 到 L1 应使用 `set_cube_tile_shapes` 配置的块大小；Scatter → 从 L1 到 HBM 应使用 `pypto.assemble`
3. **检查计算与搬运重叠**：使用 `submit_before_loop=True` 确保子循环正确提交

## 尾轴优化案例

**场景**：尾轴为 1 或非整除时，最后一块数据量小于固定块大小

**问题**：最后一块可能触发额外的零填充计算，浪费算力

**优化方案**：

```python
# 使用 valid_shape 标记有效数据范围
for i in pypto.loop(range(total_tiles)):
    tile = pypto.view(input, shape=[BLOCK_SIZE], offsets=[i * BLOCK_SIZE], valid_shape=[actual_last_size if i == last_tile else BLOCK_SIZE])
    result = compute(tile)
    pypto.assemble(result, offsets=[i * BLOCK_SIZE], output=output)
```

## 参考资料

- [set_cache_policy API 文档](../../../../docs/api/tensor/pypto-Tensor-set_cache_policy.md)
- [CachePolicy 数据类型](../../../../docs/api/datatype/CachePolicy.md)
- [性能调优文档](../../../../docs/tutorials/debug/performance.md)
- [GLM MoE Fusion 案例](../../../../models/glm_v4_5/glm_moe_fusion.py)
- [MLA Prolog Quant 案例](../../../../models/deepseek_v32_exp/mla_prolog_quant_impl.py)
- [典型案例库](cases/README.md)
