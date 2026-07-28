---
name: pypto-aicpu-perf-coding
description: PyPTO 昇腾 AICPU / Control-Flow / Schedule 热路径高性能编码原则与评审清单。涵盖禁 STL 与热路径堆分配、显式指针类型、引用传参、平凡默构与结构体布局、忌整结构初始化、Host 以存代算、循环外提、Host 严禁随意 rtMemcpy、惰性 bump、flat 小池、hit 零拷贝等。当用户修改 machine 设备侧调度/schedule、写 AICPU 热路径、做 ctrl-flow CPU 性能优化、评审 DeviceTask/ItemPool/AOT/slab/CF cache/device_sche 相关改动，或提到 AICPU 高性能编码、禁容器、平凡初始化时使用。
---

# AICPU 热路径高性能编码

面向 `framework/src/machine/` 下 **AICPU Control-Flow 与 Schedule 调度热路径**（含 `device_sche*`；非 AICore kernel）。
Host encode/UT 可用 STL，但仍须遵守「Host 严禁随意新增 rtMemcpy」等 Host 侧条款。

详细原则见：[references/coding-principles.md](references/coding-principles.md)（**必读**后再改代码或下评审结论）。

## 何时使用

- 修改 / 新增：`device_*`、`device_sche*`、`dev_workspace`、`item_pool`、`aot_binary`、`spsc_queue`、CF cache、stitch/task build
- 用户提到：AICPU 性能、ctrl cpu、schedule、禁容器、平凡初始化、热路径、`MakeDynDeviceTask`
- Code review：machine 设备侧 / Host launcher 拷贝是否引入回归

**不要**用本 skill 替代：workspace OOM（`pypto-machine-workspace`）、精度（`pypto-precision-*`）、Pass 分析（`pypto-pass-*`）。

## Agent 工作流

1. **读原则**：打开 `references/coding-principles.md`，至少覆盖 §2.1–§2.5、§2.9–§2.11。
2. **划定范围**：CF / Schedule 热路径？Host launcher？二者约束不同（见原则 Host 条款）。
3. **对照清单**（命中即改或书面例外）：
   - [ ] 无热路径 STL 容器 / 业务 `new`
   - [ ] 地址用显式指针类型，不用 `uint64_t` 长期冒充地址
   - [ ] 局部对象优先引用传参，避免无必要取址
   - [ ] 热点结构体 trivial 默构 + 布局热冷分离 / 少洞
   - [ ] 无整结构 memset；Shell/Fill/bump；按 `usedSize` 拷
   - [ ] 能 Host 判断的 if 不放到 AICPU（以存代算 / encode 旗标）
   - [ ] 循环边界与循环不变量已外提
   - [ ] Host 无新增直连 `rtMemcpy` H2D；必须拷贝则 `NormalizedRtMemcpy` / RELAXED
   - [ ] Schedule 路径与 CF 路径一并检查，不只看 stitch
4. **落地或评审**：优先 `ItemPool` / 自研 `Vector` / 定长数组 / 引用接口；输出违规点与建议。
5. **验证**：相关 UT；必要时上板 `PERF_EVT_*`。
   注意：**CSE 仅 Host 生成 CF 代码时关注，AICPU 运行时编码不涉及、运行时也感知不到。**

## 硬约束（摘要）

| 必须 | 禁止 |
|---|---|
| 定长数组 / `Vector`+workspace / `ItemPool` / `SPSC` | 热路径 STL、业务 `new/delete` |
| 显式 `T*`；局部按引用传递 | `uint64_t` 当万能地址；无必要 `&local` |
| trivial 默构 + 合理布局 | 默构清大数组；热点结构塞冷字段 |
| Host 预计算；循环外提 | AICPU 重复 if/算本可 Host 完成的事 |
| 必须 H2D 时用 `NormalizedRtMemcpy` | Host 随意新增直连 `rtMemcpy` |

违反 §2.1–§2.5、§2.9、§2.11：**即使功能正确，也视为回归。**

## 参考代码

```text
framework/src/machine/utils/dynamic/item_pool.h
framework/src/machine/utils/dynamic/vector.h
framework/src/machine/utils/dynamic/spsc_queue.h
framework/src/machine/utils/dynamic/device_task.h
framework/src/machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h
framework/src/machine/device/dynamic/aot_binary.h
framework/src/machine/device/dynamic/device_sche.cpp
framework/src/machine/runtime/memory_utils/memory_pool.cpp  # NormalizedRtMemcpy
```
