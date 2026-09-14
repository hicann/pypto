# 时序 / 重叠流水检视规则

> 本文件为 skill `pypto-machine-code-review` 的 §7 细则。
>
> early launch、整网压测时，ctrl / sche / aicore 经常叠着跑。ctrl 自己这条流是串行的，但下一轮 ctrl 会撞上上一轮还没跑完的 sche、aicore。另外 device OS 会让单个 AICPU 连跑大约 950ms 后强制让出大约 50ms。
>
> 不要看数据挂在哪个结构上，要看还有谁在读。

---

## 1. 先看谁在读，再决定能不能改

ctrl 和自己不重叠，所以 **只有 ctrl 用的 DevProg 字段可以写**。sche 或 aicore 也会读、而且每轮值会变的字段，不能在共享对象上原地改，本轮值放到 `DevStartArgs` 或按 slot 切开的 cache 里。

| 对象 | 谁在读 | 能不能按轮次改共享那份 |
|---|---|---|
| DevProg 上 ctrl 自己用的字段（`ctrlFlowCacheAnchor`、记录态 cache、`memBudget` 补丁、`ResetRerun` 等） | 只有 ctrl | 可以写 |
| `devArgs.nrValidAic` / `scheCpuNum` | ctrl 和 sche 都读（sche 入口会拷 `devProg->devArgs`） | 共享里只留满核容量；本轮核数写到 `DevStartArgs` |
| ring 头 / `runtimeDataRingBufferInited` | ctrl 和 sche 都用 | 走现成的申请/释放、等待 inited，不要另改一套 |
| `KernelArgs` / sharedBuffer（`parallelDevTask`、shake） | sche 和 aicore | 见第 2 点，不能当本轮私有缓冲直接改 |
| slot 上 CF cache 的 `rawTensorAddr` | ctrl 做 restore 时写，aicore 读 | 见第 2 点，绕回前要等这个 slot 的 AICore 跑完 |

判定：PR 改了某个字段，先列出 early launch 下还可能活着的读者。读者里有 sche 或 aicore，且这个值每轮会变 → Blocker。

---

## 2. 槽位要等真正用完的人，清干净再交给下一轮

第 1 点管的是「这个字段能不能改」。第 2 点管的是「这块缓冲用完后怎么还」：sche 退出不等于 AICore 已经读完。`Deallocate` 发生在 sche 都退出时，这时 AICore 可能还在跑。

还槽、停核的顺序必须是：

```text
先让 STOP 被对面看见 → 再清槽 → 再 GOODBYE / 释放
```

中间的 barrier 不能塞进可选分支（例如 `if (NeedsFastPathRegClose)`）。early launch 下复用 slot 时，restore 前要保证上一任 AICore 已经退出。

判定：出现下面任一情况 → Blocker。

- 复用 ring slot / sharedBuffer / shake 时，同步只等到 sche，没有等到 AICore
- 停核顺序不是「STOP 可见 → 清槽 → GOODBYE」，或把 STOP 后的 barrier 放进可选 if

---

## 3. 等别的 AICPU、失败就报错的等待，要盖过 OS 节流

单个 AICPU 被节流时会让出大约 50ms；超时用的是硬件计数器，让出期间时间照样走。

- 超时就是功能失败、并且在等**别的 AICPU** 起来：等待时间要 ≥ `GetOsThrottleSafeWaitTimeout()`
- 失败只是降级、或只是快速探测：继续用短超时，不要一律拉长

判定：改了或加了 spin 等待，先看失败后是直接报错还是降级。直接报错、对端又可能被节流 → 必须盖过节流窗口。
