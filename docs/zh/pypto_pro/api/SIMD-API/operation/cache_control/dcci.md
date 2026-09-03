# pypto_pro.language.system.dcci

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

对指定地址对应的数据缓存执行清理并失效（Data Cache Clean and Invalidate，DCCI）。典型用途是在跨核或跨流水共享数据时，清除当前执行单元可能持有的旧缓存副本，使后续访问能够观察到已发布的数据。

dcci只处理缓存状态，不等价于流水同步、跨核事件或内存屏障。调用者仍须保证生产者写入已经完成，并使用与通信协议匹配的同步接口建立先后关系。

## 函数原型

```python
pypto_pro.language.system.dcci(
    target: Union[Tensor, Tile],
    offset: Optional[Union[int, Sequence[int]]] = None,
    *,
    cache_line: CacheLine = pypto_pro.language.CacheLine.ENTIRE_DATA_CACHE,
    dst: DcciDst = pypto_pro.language.DcciDst.AUTO,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| target | 输入 | GM Tensor变量，或已分配在UB中的Tile。其他Tile内存空间不支持。GM Tensor下标表达式表示单个元素，不能作为Tensor目标；UB Tile可使用完整Tile或其subview，DCCI从该Tile表达式对应的有效地址开始操作。 |
| offset | 输入 | 可选，元素偏移，单位为target.dtype元素。GM Tensor支持各维偏移列表/元组，也支持整型常量或运行时整型标量表达式表示的线性偏移；列表/元组长度须与Tensor维数一致，框架按Tensor stride换算线性偏移。UB Tile仅支持整型常量或运行时整型标量表达式表示的线性偏移。缺省时为0，即使用目标起始地址；偏移不得使有效地址越出目标已分配范围。 |
| cache_line | 输入 | 可选，编译期[pypto_pro.language.CacheLine](../../basic_data_structures/CacheLine.md)枚举值，默认pypto_pro.language.CacheLine.ENTIRE_DATA_CACHE。缓存行为64字节；SINGLE_CACHE_LINE的地址无需由用户向下对齐，硬件操作包含该地址的缓存行。若数据跨越多个缓存行，须逐行调用或使用ENTIRE_DATA_CACHE。 |
| dst | 输入 | 可选，编译期[pypto_pro.language.DcciDst](../../basic_data_structures/DcciDst.md)枚举值，默认pypto_pro.language.DcciDst.AUTO。各枚举值的含义和适用目标参见DcciDst。 |

显式指定dst时，其取值必须与target的存储区域和硬件访问路径匹配：GM Tensor使用CACHELINE_OUT或CACHELINE_ALL，UB Tile使用CACHELINE_UB；CACHELINE_ATOMIC仅用于硬件原子缓存路径。普通GM/UB场景建议使用AUTO。

## 返回值说明

无返回值。

## 约束说明

1. cache_line和dst必须在编译期确定，不能由运行时Scalar或Tensor动态选择。
2. SINGLE_CACHE_LINE只覆盖一个64字节缓存行。处理地址区间[addr, addr + bytes)时，调用次数至少为该区间覆盖的缓存行数，不能只对首地址调用一次。
3. ENTIRE_DATA_CACHE作用于整个数据缓存，offset不会缩小其作用范围；该模式开销大于单缓存行操作。
4. DCCI不是同步原语。生产者通过MTE3等流水写出数据后，必须先同步到S流水再执行DCCI或发布标志；消费者也必须先完成相应跨核等待，再执行缓存处理和数据读取。具体事件号和同步模式由上层通信协议决定。
5. target为UB Tile，或dst选择CACHELINE_UB时，调用前必须确保CTRL寄存器的CTRL[49]已置1以开启UB datacache模式；本接口不会自动修改该控制位。
6. 频繁对整个缓存执行DCCI会造成明显性能损失；已知共享数据范围时应优先按64字节缓存行处理。

## 调用示例

### 单缓存行失效

```python
pl.system.dcci(
    inp,
    [0, 0],
    cache_line=pl.CacheLine.SINGLE_CACHE_LINE,
    dst=pl.DcciDst.AUTO,
)
```

### 全缓存失效

```python
# 全缓存失效
pl.system.dcci(inp, cache_line=pl.CacheLine.ENTIRE_DATA_CACHE)
```
