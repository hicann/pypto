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
                     vec_nbuffer_setting: Optional[Dict[str, int]] = None,
                     cube_l1_reuse_setting: Optional[Dict[str, int]] = None,
                     cube_nbuffer_setting: Optional[Dict[str, int]] = None,
                     sg_set_scope: Optional[Union[int, Tuple[int, bool, bool]]] = None,
                     pg_partition_algorithm: Optional[str] = None,
                     auto_mix_partition: Optional[int] = None,
                     )
```

## 参数说明

| 参数名                  | 输入/输出 | 说明                                                                 |
|-------------------------|-----------|----------------------------------------------------------------------|
| vec_nbuffer_setting     | 输入      | 含义：合图参数，用于配置相同结构AIV子图的合并数量。 <br> 说明：该参数适用于结构相同的AIV子图合并。<br><br>类型：dict[str, int]。支持两种 key 格式：<br> ① **函数粒度 key**：字符串 `"func{magic}_{order}"` 或 `"DEFAULT"`，可实现不同 root function 间的精细化配置。`func{magic}_{order}` 匹配特定 function（funcMagic=magic）的特定同构子图组（hashorder=order）；`DEFAULT` 匹配所有未显式指定的子图组。配置后合图信息会直接展示在泳道图的 hashOrder-hint 字段中（含 subGraphCount，代表合并前的子图数量，可根据核心数匹配合并力度）。详情参见下方"函数粒度key配置说明"。<br> ② **语义标签 key**：任意非 func 前缀字符串，按语义标签控制合图粒度（详见下方语义标签key配置说明）。可与函数粒度 key 共存。<br> 取值：<br> {"DEFAULT": 1}：跳过AIV子图合并 <br> {} （空字典）：自动合并，根据AIV核心数自动计算合并粒度<br> {"DEFAULT": N, "func8_0": N2}：手动合并，默认粒度为 N，func8_0 对应子图组粒度为 N2<br> 默认值：{} 空字典 <br> 影响Pass范围： NBufferMerge |
| cube_l1_reuse_setting | 输入 | 含义：合图参数，用于配置重复搬运同一GM数据的子图合并数量。<br> 说明：该参数适用于含有CUBE计算的子图合并，与cube_nbuffer_setting同时配置时，先进行此项合并，再进行cube_nbuffer_setting相关合并。<br><br>类型：dict[str, int]。支持两种 key 格式：<br> ① **函数粒度 key**：字符串 `"func{magic}_{order}"` 或 `"DEFAULT"`，可实现不同 root function 间的精细化配置。合图信息会直接展示在泳道图的 hashOrder-hint 字段中（含 subGraphCount，代表合并前的子图数量，可根据核心数匹配合并力度）。详情参见下方"函数粒度配置说明"。<br> ② **语义标签 key**：按语义标签控制（详见下方语义标签key配置说明）。<br> 取值：<br>{"DEFAULT": 1}：跳过L1Reuse合并 <br> {} （空字典）：自动合并<br> {"DEFAULT": N, "func8_0": N2}：手动合并，默认粒度为 N<br> 默认值：{} 空字典 <br> 影响Pass范围：L1CopyInReuseMerge |
| cube_nbuffer_setting    | 输入      | 含义：合图参数，用于配置相同结构AIC子图的合并数量。 <br> 说明：该参数适用于结构相同的AIC子图合并，与cube_l1_reuse_setting同时配置时，先进行cube_l1_reuse_setting相关合并，再进行此项合并。<br><br>类型：dict[str, int]。支持两种 key 格式：<br> ① **函数粒度 key**：字符串 `"func{magic}_{order}"` 或 `"DEFAULT"`，可实现不同 root function 间的精细化配置。合图信息会直接展示在泳道图的 hashOrder-hint 字段中（含 subGraphCount，代表合并前的子图数量，可根据核心数匹配合并力度）。详情参见下方"函数粒度配置说明"。<br> ② **语义标签 key**：按语义标签控制（详见下方语义标签key配置说明）。<br> 取值：<br>{"DEFAULT": 1}：跳过AIC子图合并 <br> {} （空字典）：自动合并<br> {"DEFAULT": N, "func8_0": N2}：手动合并，默认粒度为 N<br>默认值：{"DEFAULT": 1} <br> 影响Pass范围：L1CopyInReuseMerge |
| sg_set_scope            | 输入      | 含义：手动控制子图切分参数。<br> 说明：通过为 Operation 分配 scope，使得相同 scope_id（非-1） 的相邻 Operation 强制合并归入同一子图，从而覆盖切分算法的自动划分结果。 <br> 类型：`Tuple[int, bool, bool]` 或 `int` <br> **tuple 格式**：`(scope_id, allow_parallel_merge, allow_cross_scope_merge)`，各字段含义如下： <br> - `scope_id`（int）：scope 标识，取值范围 -1~2147483647。相同 scope_id 的相邻 Operation 归入同一子图；-1 表示不参与 scope 合并，由切分算法决定子图划分。 <br> - `allow_parallel_merge`（bool）：控制同一 scope_id 下 Operation 的合并方式。取值 True/False。<br>&emsp;&emsp;False（默认）：仅允许存在上下游连接通路的 Operation 合并，即 Operation A 的输出作为 Operation B 的输入时才可合并到同一子图。<br>&emsp;&emsp;True：允许位于并行分支（无数据依赖）的相同 scope_id 的 Operation 也合并到同一子图。 <br> - `allow_cross_scope_merge`（bool）：控制带有 scope 的子图是否可与无 scope（scope_id=-1）的子图合并，扩大scope子图。取值 True/False。<br>&emsp;&emsp;False（默认）：带有 scope 的子图保持独立，不与其他子图合并。<br>&emsp;&emsp;True：允许带有 scope 的子图与 scope_id=-1 的子图合并。不同 scope_id 的子图之间不可合并。 <br> **int 格式**：传入单个 int 时等价于 `(scope_id, False, False)`，即仅设置 scope_id，不允许并行分支合并和跨 scope 合并。 <br> 默认值：(-1, False, False) <br> 影响Pass范围：GraphPartition <br> 配置建议：1）视图类Operation与其对应的计算类Operation应配置相同的 scope_id。2）Reshape Operation较为特殊，部分场景会单独成子图，手动控制合图行为可能失效。|
| pg_partition_algorithm  | 输入      | 含义：指定切分算法。<br> 说明：配置GraphPartition环节进行子图切分所采用的算法。当同时配置了 `sg_set_scope` 时，无论选择哪种切分算法，都会优先遵从 `sg_set_scope` 的强制合图约束。<br> 类型：str <br> 取值范围："Iso", "OspSarkar", "OspBsp" <br> 默认值："Iso" <br> 影响Pass范围：GraphPartition <br> 算法选择指导：请参考下文。|
| auto_mix_partition      | 输入      | 含义：控制ReduceCopyMerge Pass中的自动混合子图切分行为。<br> 说明：该参数用于控制CV混合场景下子图的自动合并策略。<br> 类型：int <br> 取值：0：不进行自动CV Mix合图；1： 进行自动CV Mix合图。<br> 默认值：0 <br> 影响Pass范围：ReduceCopyMerge |

## 返回值说明

无。

## 约束说明

- 设置时机：不要求在图编译开始前调用，可以在任何时候进行设置。
- 类型安全：必须确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
- 作用范围：参数设置是局部的，只会影响当前jit或者loop内的编译过程，若未设置，则继承上层作用域。
- 语义标签key：setting 的字符串 key 必须与至少一个 operation 通过 `pypto.set_semantic_label` 设置的 semantic_label 完全匹配，否则编译时报错。
- sg_set_scope 一致性约束：同一 scope_id 的所有 Operation 必须设置相同的 `allow_parallel_merge` 和 `allow_cross_scope_merge`，否则编译报错。
- scope_id 为 -1 时，`allow_parallel_merge` 和 `allow_cross_scope_merge` 必须为 False。
- 不同 scope_id 的子图之间不可合并，`allow_cross_scope_merge` 仅控制带 scope 的子图与无 scope（scope_id=-1）的子图合并。
- auto_mix_partition 为 1 时，编译器会评估相邻子图，若合并预估能带来性能收益且不会形成环，则会合并成MIX子图，否则不会进行合并。

## 调用示例

```python
   # 函数粒度配置（func{magic}_{order} 格式）
   pypto.set_pass_options(
       vec_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_l1_reuse_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1})

   # 纯 DEFAULT
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2})

   # 语义标签 key 配置（可与函数粒度 key 共存）
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("")
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})
```


### dict类型配置说明（函数粒度 key / 语义标签 key）

### 函数粒度 key 配置说明（func{magic}\_{order}）

#### 功能概述

通过 `"func{magic}_{order}"` 格式的 key，可以针对**特定 function** 的**特定同构子图组**（hashorder）设置合并粒度，实现不同 root function 间的精细化配置。配置的 hashOrder 和 subGraphCount 信息会直接展示在泳道图的 hashOrder-hint 字段中，格式为 `l1ReuseInfo hashOrder: func8_0, subGraphCount: 24`，可根据子图数量和核心数匹配合并力度。

#### 键值对含义

Key: 字符串格式 `"func{magic}_{order}"` 或 `"DEFAULT"`。<br>
- `"func{magic}_{order}"`：匹配 functionMagic 为 magic 的 function 中，hashorder 为 order 的同构子图组。<br>
- `"DEFAULT"`：匹配所有未显式指定的同构子图组。<br>

Value (N): 表示合并粒度。即同构子图组内每 N 个子图合并为一个新子图执行。N=1 表示不合并。

#### 格式约束

- `func` 前缀必须小写。
- magic 和 order 必须为整数。
- magic 和 order 之间以下划线 `_` 分隔。
- 示例合法 key：`"func0_0"`、`"func123_5"`、`"func8_1"`。
- 示例非法 key：`"Func0_0"`（大写F）、`"func_0"`（缺少 magic）、`"func123"`（缺少 order）。

#### 配置行为

Pass 在处理当前 function 的子图合并时，遵循 "func{magic}\_{order} 精确匹配 > DEFAULT 默认配置 > 自动处理" 的逻辑：<br>
- 精确匹配：若 funcMagic 和 hashorder 双双命中，则按其对应的 Value N 进行合并。<br>
- DEFAULT 默认配置：若未精确命中，但字典中存在 `DEFAULT`，则按 `DEFAULT` 对应的 Value 执行合并。<br>
- 自动处理：若既未精确命中也无 `DEFAULT`，则自动计算合并粒度。<br>

#### 配置示例

| 配置 | 说明 |
|------|------|
|`{"DEFAULT": 1}`|所有同构子图组跳过合并。|
|`{"DEFAULT": 4, "func8_0": 1, "func8_1": 1}`|默认 4 个同构子图为一组进行合并，func8 中 hashorder 0 和 1 的子图组跳过合并（不合并）。|
|`{"DEFAULT": 2, "func8_1": 4}`|默认 2 个同构子图为一组进行合并，func8 中 hashorder 1 的子图四个为一组进行合并。|
|`{"func8_0": 2}`|func8 函数中，hashorder 为 0 的子图两个为一组进行合并；其他同构子图组，根据硬件核心数自动计算合并粒度并进行合并。|

### 语义标签 key 配置说明

#### 功能概述

除函数粒度 key 外，`vec_nbuffer_setting`、`cube_l1_reuse_setting` 和 `cube_nbuffer_setting` 还支持使用字符串 key，即通过 `pypto.set_semantic_label` 设置的语义标签名称。字符串 key 允许用户精确控制特定 operation 所在子图（允许多个）的合并粒度，无需关心其 hashorder 编号。

#### 字符串键值对含义

Key (label): 语义标签名称，必须与至少一个 operation 的 `semantic_label` 完全匹配。<br>
Value (N): 表示合并粒度。<br>

#### 优先级机制

字符串 key 的优先级**高于**函数粒度 key 的默认配置。处理流程为：<br>
1. 首先根据函数粒度 key（`func{magic}_{order}` / `DEFAULT`）确定各同构子图组的基础合并粒度。<br>
2. 然后字符串 key 的值**直接替换**（而非取 max）对应子图组的合并粒度。<br>
3. 当多个不同的字符串 label 指向同一个同构子图组时，取这些 label 值中的最大值。<br>

#### vec_nbuffer_setting / cube_nbuffer_setting 的语义标签行为

字符串 key 覆盖其所在 operation 对应的**整个同构子图组**的合并粒度。

#### cube_l1_reuse_setting 的语义标签行为

与 `vec_nbuffer_setting` 和 `cube_nbuffer_setting` 不同，`cube_l1_reuse_setting` 的字符串 key **仅作用于包含对应标签 operation 的子图**，不展开到整个同构组。即同构组内可能只有部分子图被字符串 key 覆盖，其他子图保持函数粒度 key 的值。

#### 语义标签配置示例

| 配置                                | 说明                                                                 |
|-------------------------------------|----------------------------------------------------------------------|
|{"DEFAULT": 2, "V1": 1}|所有同构子图组默认合并粒度为2；但 V1 标签所在的同构子图组合并粒度被替换为1。|
|{"V1": 3}|V1 标签所在的同构子图组合并粒度为3；其他同构子图组自动计算合并粒度。|
|{"DEFAULT": 2, "V1": 1, "V2": 3}|默认合并粒度为2；V1 所在组替换为1；V2 所在组替换为3。若某一组同时有 V1 和 V2 两种OP，则取 max(1, 3) = 3。|

#### 配置示例
```python
   # 混合函数粒度 key 和语义标签 key 配置
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("") # 通过更改语义标签，来精确控制只有该mul OP的语义标签是"V1"
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})

   # 纯语义标签 key 配置
   pypto.set_pass_options(cube_l1_reuse_setting={"MM1": 4})
```


### sg_set_scope 配置说明

#### 配置示例

```python
# int 格式：等效于 (10, False, False)，仅设置 scope_id
pypto.set_pass_options(sg_set_scope=10)

# tuple 格式：scope_id=1，允许并行分支合并，不允许跨 scope 合并
pypto.set_pass_options(sg_set_scope=(1, True, False))

# tuple 格式：scope_id=2，允许与无 scope 的子图合并
pypto.set_pass_options(sg_set_scope=(2, False, True))

# 恢复默认（不参与 scope 合并，由合图算法自动决定）
pypto.set_pass_options(sg_set_scope=-1)
```

#### 典型场景

##### 场景一：整张计算图不切分

当需要将整个计算图保持不切分时，因数据切块会产生多条并行分支，这些分支之间无直接数据依赖，默认会被切分算法拆为独立子图。推荐设置 `sg_set_scope=(scope_id, True, False)`，通过 `allow_parallel_merge=True` 使相同 scope_id 的并行分支 Operation 合并到同一子图。

##### 场景二：A5 CV 混合场景，构造 Mix 子图以减少 GM 搬运

当 Cube 操作的前后均有 Vec 操作时，目标是构造一个包含 Cube 和 Vec 的 Mix 子图，避免中间结果在 GM 上反复搬运。根据是否明确 scope 边界，分为以下两种情况：

**场景 2.1：明确 scope 边界**

当可以明确划分 Cube 操作及其紧邻 Vec 操作的边界时，使用 `(scope_id, False, False)` 标记边界，使 Cube 和紧邻的 Vec 强制归入同一子图，形成 Mix 子图。

```python
# 明确标记 Cube 及紧邻 Vec 为同一 scope，形成 Mix 子图
pypto.set_pass_options(sg_set_scope=(1, False, False))
# ... Cube 操作 ...
# ... 紧邻的 Vec 操作 ...
pypto.set_pass_options(sg_set_scope=-1)
```

**场景 2.2：不明确 scope 边界，仅标记 CV 合并边界**

当无法明确划分边界，但需要 Cube 前后的 Vec 子图与 Cube 子图合并形成 Mix 子图时，使用 `(scope_id, False, True)` 标记 Cube 及其紧邻的 Vec 作为合并锚点。通过 `allow_cross_scope_merge=True`，该 scope 子图中的 Vec 可与前后无 scope（scope_id=-1）的 Vec 子图合并，形成更大的子图以减少 GM 数据搬运，并与 Cube 子图一起形成 Mix 子图。

```python
# 前置 Vec 操作（scope_id=-1，由切分算法自动决定）
vec_out = some_vec_op(x)

# 标记 CV 合并锚点，允许与无 scope 的相邻子图合并
pypto.set_pass_options(sg_set_scope=(1, False, True))
# ... Cube 操作 ...
matmul_result = pypto.matmul(vec_out, w)
# ... 紧邻的 Vec 操作 ...
pypto.set_pass_options(sg_set_scope=-1)

# 后续 Vec 操作（scope_id=-1），可与上方 scope 子图合并为更大子图
result = other_vec_op(add_result)
```


### pg_partition_algorithm 算法选择指导 (Algorithm Selection Guidance)

| 参数值 (Value) | 适用场景 (Applicable Scenario) |
| :--- | :--- |
| **"Iso"** | 基于同构的切分算法 (Isomorphism-based partitioning)；适用于常规通用场景 (suitable for general-purpose use)。 |
| **"OspSarkar"** | 基于关键路径缩减的瓦片内核融合。 (Fuses tile kernels based on critical path reduction.) |
| **"OspBsp"** | 基于 BSP 模型融合瓦片内核，用于并行计算和同构检测。 (Fuses tile kernels based on the BSP model for parallel computation and isomorphism detection.) |
