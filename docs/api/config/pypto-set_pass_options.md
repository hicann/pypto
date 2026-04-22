# pypto.set\_pass\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

修改Pass优化参数信息。其主要功能是在编译流程中，针对特定的优化策略和具体的Pass，动态修改其运行时参数配置，从而实现精细化的控制和调试。

## 函数原型

```python
set_pass_options(*,
                     vec_nbuffer_setting: Optional[Dict[int, int]] = None,
                     cube_l1_reuse_setting: Optional[Dict[int, int]] = None,
                     cube_nbuffer_setting: Optional[Dict[int, int]] = None,
                     sg_set_scope: Optional[Union[int, Tuple[int, bool, bool]]] = None,
                     pg_partition_algorithm: Optional[str] = None,
                     )
```

## 参数说明

| 参数名                  | 输入/输出 | 说明                                                                 |
|-------------------------|-----------|----------------------------------------------------------------------|
| vec_nbuffer_setting     | 输入      | 含义：合图参数，用于配置相同结构AIV子图的合并数量。 <br> 说明：该参数适用于结构相同的AIV子图合并。 <br> 类型：dict[int, int] <br> 取值：<br> {-1: 1}：跳过AIV子图合并 <br> {} （空字典）：自动合并，根据AIV核心数自动计算合并粒度<br> {-1: N, 0: N2, ...}：手动合并，默认粒度为N <br> 默认值：{} 空字典 <br> 影响Pass范围： NBufferMerge |
| cube_l1_reuse_setting | 输入 | 含义：合图参数，用于配置重复搬运同一GM数据的子图合并数量。<br> 说明：该参数适用于含有CUBE计算的子图合并。 <br> 类型： dict[int, int] <br> 取值：<br>{-1: 1}：跳过L1Reuse合并 <br> {} （空字典）：自动合并，根据AIC核心数自动计算合并粒度<br> {-1: N, 0: N1, ...}：手动合并，默认合并粒度为N。 <br> 默认值：{} 空字典 <br> 影响Pass范围：L1ReuseMerge |
| cube_nbuffer_setting    | 输入      | 含义：合图参数，用于配置相同结构AIC子图的合并数量。 <br> 说明：该参数适用于结构相同的AIC子图合并。 <br> 类型：dict[int, int] <br> 取值：<br>{-1: 1}：跳过AIC子图合并 <br> {} （空字典）：自动合并，根据AIC核心数自动计算合并粒度<br> {-1: N, 0: N1, ...}：手动合并，默认合并粒度为N <br>默认值：{-1: 1} <br> 影响Pass范围： L1ReuseMerge |
| sg_set_scope            | 输入      | 含义：控制合图（子图划分）行为，将 operation 赋予结构化的 ScopeInfo。 <br> 类型：`Tuple[int, bool, bool]` 或 `int` <br> **tuple 格式**：`(scope_id, allow_parallel_merge, allow_cross_scope_merge)`，各字段含义如下： <br> - `scope_id`（int）：scope 标识，取值范围 -1~2147483647。相同 scope_id 的相邻 Operation 归入同一子图；-1 表示不参与 scope 合并，由合图算法决定子图划分。 <br> - `allow_parallel_merge`（bool）：控制同一 scope_id 下 Operation 的合并方式。取值 True/False。<br>&emsp;&emsp;False（默认）：仅允许存在上下游连接通路的 Operation 合并，即 Operation A 的输出作为 Operation B 的输入时才可合并到同一子图。<br>&emsp;&emsp;True：允许位于并行分支（无数据依赖）的同一 scope_id Operation 也合并到同一子图。 <br> - `allow_cross_scope_merge`（bool）：控制带有 scope 的子图是否可与无 scope（scope_id=-1）的子图合并，扩大scope子图。取值 True/False。<br>&emsp;&emsp;False（默认）：带有 scope 的子图保持独立，不与其他子图合并。<br>&emsp;&emsp;True：允许带有 scope 的子图与 scope_id=-1 的子图合并。不同 scope_id 的子图之间不可合并。 <br> **int 格式**：传入单个 int 时等价于 `(scope_id, False, False)`，即仅设置 scope_id，不允许并行分支合并和跨 scope 合并。 <br> 默认值：(-1, False, False) <br> 影响Pass范围：GraphPartition <br> 配置建议：1）视图类Operation与其对应的计算类Operation应配置相同的 scope_id。2）Reshape Operation较为特殊，部分场景会单独成子图，手动控制合图行为可能失效。|
| pg_partition_algorithm  | 输入      | 含义：指定切分算法。<br> 说明：配置GraphPartition环节进行子图切分所采用的算法。当同时配置了 `sg_set_scope` 时，无论选择哪种切分算法，都会优先尊重 `sg_set_scope` 的强制合图约束。<br> 类型：str <br> 取值范围："Iso", "OspSarkar", "OspBsp" <br> 影响Pass范围：GraphPartition <br> 算法选择指导：请参考下文。|

### 算法选择指导 (Algorithm Selection Guidance)

| 参数值 (Value) | 适用场景 (Applicable Scenario) |
| :--- | :--- |
| **"Iso"** | 基于同构的切分算法 (Isomorphism-based partitioning)；适用于常规通用场景 (suitable for general-purpose use)。 |
| **"OspSarkar"** | 基于关键路径缩减的瓦片内核融合。 (Fuses tile kernels based on critical path reduction.) |
| **"OspBsp"** | 基于 BSP 模型融合瓦片内核，用于并行计算和同构检测。 (Fuses tile kernels based on the BSP model for parallel computation and isomorphism detection.) |

## 返回值说明

无。

## 约束说明

- 设置时机：不要求在图编译开始前调用，可以在任何时候进行设置。
- 类型安全：必须确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
- 作用范围：参数设置是局部的，只会影响当前jit或者loop内的编译过程，若未设置，则继承上层作用域。
- sg_set_scope 一致性约束：同一 scope_id 的所有 Operation 必须设置相同的 `allow_parallel_merge` 和 `allow_cross_scope_merge`，否则编译报错。
- scope_id 为 -1 时，`allow_parallel_merge` 和 `allow_cross_scope_merge` 必须为 False。
- 不同 scope_id 的子图之间不可合并，`allow_cross_scope_merge` 仅控制带 scope 的子图与无 scope（scope_id=-1）的子图合并。

## 调用示例

```python
   pypto.set_pass_options(
                       vec_nbuffer_setting={},
                       cube_l1_reuse_setting={},
                       cube_nbuffer_setting={})
```

### sg_set_scope 配置示例

```python
import pypto

# int 格式：等效于 (48, False, False)
pypto.set_pass_options(sg_set_scope=48)

# tuple 格式：scope_id=1，允许并行分支合并，不允许跨 scope 合并
pypto.set_pass_options(sg_set_scope=(1, True, False))

# tuple 格式：scope_id=2，允许与无 scope 的子图合并
pypto.set_pass_options(sg_set_scope=(2, False, True))

# 恢复默认（不参与 scope 合并）
pypto.set_pass_options(sg_set_scope=-1)
```

### dict类型配置说明

#### 键值对含义

Key (hashorder): 同构子图组id。<br>
- 值 M: 匹配 hashorder 为 M 的特定子图组。<br>
- 值 -1: 匹配所有未显式指定的子图组。<br>

Value (N): 表示合并粒度。即：同构子图组内每N个子图合并为一个新子图执行。<br>

#### 配置行为

Pass 在处理子图合并时，遵循 “精确匹配 > 默认配置 > 自动处理” 的逻辑：<br>
- 精确匹配: 若 hashorder 命中字典中的特定 Key，则按其对应的 Value N 进行合并。<br>
- 默认配置: 若未精确命中，但字典中存在 -1，则按 -1 对应的 Value 执行合并。<br>
- 自动处理: 若既未精确命中也无 -1 配置，则自动计算合并粒度进行合并优化。<br>

#### 配置示例

| 配置                  | 说明                                                                 |
|---------------------- |----------------------------------------------------------------------|
|{-1: 1}|跳过子图合并。合并粒度为1，即所有同构子图组内的子图不进行合并。|
|{0: 5}|对于hashorder为0的同构子图组，每5个子图合并为一个子图；<br>其他同构子图组，根据硬件核心数自动计算合并粒度并进行合并。|
|{0: 5, 2: 8, -1: 2}    |hashorder为0的同构子图组，每5个子图合并为一个子图；<br>hashorder为2的同构子图组，每8张子图合并为一个子图；<br>其他的同构子图组使用-1对应的默认合并粒度，即每2张子图合并为一个子图。<br> |
|{0: 5, -1: 1}    |hashorder为0的同构子图组，每5个子图合并为一个子图；<br>其他同构子图组不做处理。 |
