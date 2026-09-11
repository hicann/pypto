# pypto.set\_pass\_options

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-18T12:10:00.077Z pushedAt=2026-08-26T09:10:38.136Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Modifies the **Pass** optimization parameter information. Its main function is to dynamically modify the runtime parameter settings of specific optimization strategies and specific **Pass**es during the compilation process, thereby enabling fine-grained control and debugging.

## Prototype

```python
set_pass_options(*,
                     vec_nbuffer_setting: Optional[Dict[str, int]] = None,
                     cube_l1_reuse_setting: Optional[Dict[str, int]] = None,
                     cube_nbuffer_setting: Optional[Dict[str, int]] = None,
                     sg_set_scope: Optional[Union[int, Tuple[int, bool, bool]]] = None,
                     auto_mix_partition: Optional[int] = None,
                     sg_set_ooo_scope: Optional[int] = None,
                     ooo_sched_mode: Optional[str] = None,
                     sg_set_tunevf_mode: Optional[int] = None,
                     )
```

## Parameters

| Parameter | Input/Output | Description |
|-------------------------|-----------|----------------------------------------------------------------------|
| vec_nbuffer_setting | Input | Meaning: Graph fusion parameter used to configure the number of isomorphic vector subgraphs to be merged. <br> Description: This parameter applies to the fusion of vector subgraphs with the same structure.<br><br>Type: dict[str, int]. Two key formats are supported:<br> (1) **Function granularity key**: A string `"func{magic}_{order}"` or `"DEFAULT"`, which enables fine-grained configuration across different root functions. `func{magic}_{order}` matches a specific isomorphic subgraph group (hashorder=order) of a specific function (funcMagic=magic); `DEFAULT` matches all subgraph groups that are not explicitly specified. After configuration, the graph fusion information is directly displayed in the **hashOrder-hint** field of the swimlane diagram (including **subGraphCount**, which represents the number of subgraphs before fusion and can be used to match the fusion strength based on the number of cores). For details, see "Function Granularity Key Configuration Description" below.<br> (2) **Semantic label key**: Any string without the func prefix, which controls the graph fusion granularity by semantic label (for details, see "Semantic Label Key Configuration Description" below). It can coexist with the function granularity key.<br> Values:<br> {"DEFAULT": 1}: Skip vector subgraph fusion. <br> {} (empty dictionary): Automatic fusion, which automatically calculates the fusion granularity based on the number of available vector cores.<br> {"DEFAULT": N, "func8_0": N2}: Manual fusion, with a default granularity of N and a granularity of N2 for the subgraph group corresponding to func8_0.<br> Default value: {} (empty dictionary). <br> Affected Pass scope: NBufferMerge. |
| cube_l1_reuse_setting | Input | Meaning: Graph fusion parameter used to configure the number of subgraphs that repeatedly move the same GM (Global Memory) data to be merged.<br> Description: This parameter applies to the fusion of subgraphs containing Cube computation. When configured together with **cube_nbuffer_setting**, this fusion is performed first, followed by the fusion related to *cube_nbuffer_setting*.<br><br>Type: dict[str, int]. Two key formats are supported:<br> (1) **Function granularity key**: A string `"func{magic}_{order}"` or `"DEFAULT"`, which enables fine-grained configuration across different root functions. The graph fusion information is directly displayed in the **hashOrder-hint** field of the swimlane diagram (including **subGraphCount**, which represents the number of subgraphs before fusion and can be used to match the fusion strength based on the number of cores). For details, see "Function Granularity Configuration Description" below.<br> (2) **Semantic label key**: Controlled by semantic label (for details, see "Semantic Label Key Configuration Description" below).<br> Values:<br>{"DEFAULT": 1}: Skip L1Reuse fusion. <br> {} (empty dictionary): Automatic fusion.<br> {"DEFAULT": N, "func8_0": N2}: Manual fusion, with a default granularity of N.<br> Default value: {} (empty dictionary). <br> Affected Pass scope: L1CopyInReuseMerge. |
| cube_nbuffer_setting | Input | Meaning: Graph fusion parameter used to configure the number of AIC subgraphs with the same structure to be merged. <br> Description: This parameter applies to the fusion of AIC subgraphs with the same structure. When configured together with **cube_l1_reuse_setting**, the fusion related to **cube_l1_reuse_setting** is performed first, followed by this fusion.<br><br>Type: dict[str, int]. Two key formats are supported:<br> (1) **Function granularity key**: A string `"func{magic}_{order}"` or `"DEFAULT"`, which enables fine-grained configuration across different root functions. The graph fusion information is directly displayed in the **hashOrder-hint** field of the swimlane diagram (including **subGraphCount**, which represents the number of subgraphs before fusion and can be used to match the fusion strength based on the number of cores). For details, see "Function Granularity Configuration Description" below.<br> (2) **Semantic label key**: Controlled by semantic label (for details, see "Semantic Label Key Configuration Description" below).<br> Values:<br>{"DEFAULT": 1}: Skip AIC subgraph fusion. <br> {} (empty dictionary): Explicitly enable automatic fusion.<br> {"DEFAULT": N, "func8_0": N2}: Manual fusion, with a default granularity of N.<br>Default value: {"DEFAULT": 1}, which means AIC subgraph fusion is skipped by default. <br> Affected Pass scope: L1CopyInReuseMerge. |
| sg_set_scope | Input | Meaning: Parameter for manually controlling subgraph partitioning.<br> Description: By assigning a scope to an Operation, adjacent Operations with the same **scope_id** (not -1) are forcibly merged into the same subgraph, thereby overriding the automatic partitioning result of the partitioning algorithm. <br> Type: `Tuple[int, bool, bool]` or `int`. <br> **Tuple format**: `(scope_id, allow_parallel_merge, allow_cross_scope_merge)`, where each field is described as follows: <br> - `scope_id` (int): Scope identifier, with a value range of **-1** to **2147483647**. Adjacent Operations with the same **scope_id** are grouped into the same subgraph; **-1** means the Operation does not participate in scope fusion, and the subgraph partitioning is determined by the partitioning algorithm. <br> - `allow_parallel_merge` (bool): Controls how Operations with the same **scope_id** are merged. Value: True/False.<br>&emsp;&emsp;False (default): Only Operations with an upstream-downstream connection path can be merged, that is, Operation A and Operation B can be merged into the same subgraph only when the output of Operation A serves as the input of Operation B.<br>&emsp;&emsp;True: Operations with the same **scope_id** that are located on parallel branches (without data dependency) can also be merged into the same subgraph. <br> - `allow_cross_scope_merge` (bool): Controls whether a subgraph with a scope can be merged with a subgraph without a scope (scope_id=-1) to enlarge the scoped subgraph. Value: True/False.<br>&emsp;&emsp;False (default): A subgraph with a scope remains independent and is not merged with other subgraphs.<br>&emsp;&emsp;True: A subgraph with a scope can be merged with a subgraph whose **scope_id** is **-1**. Subgraphs with different **scope_id** values cannot be merged with each other. <br> **Int format**: Passing a single int is equivalent to `(scope_id, False, False)`, that is, only **scope_id** is set, and parallel branch fusion and cross-scope fusion are not allowed. <br> Default value: (-1, False, False). <br> Affected Pass scope: GraphPartition. <br> Configuration suggestions: 1) A view-type Operation and its corresponding computation-type Operation should be configured with the same **scope_id**. 2) The Reshape Operation is special; in some scenarios it forms a separate subgraph, and manual control of graph fusion may not take effect. |
| auto_mix_partition | Input | Meaning: Controls the automatic mixed subgraph partitioning behavior in the ReduceCopyMerge pass.<br> Description: This parameter controls the automatic fusion strategy for subgraphs in CV mix scenarios. When the value is **1**, the compiler evaluates adjacent subgraphs; if the estimated fusion brings performance benefits and does not form a cycle, they are merged into a MIX subgraph; otherwise, no fusion is performed.<br> Type: int. <br> Values: 0: Do not perform automatic CV Mix graph fusion; 1: Perform automatic CV Mix graph fusion.<br> Default value: 0. <br> Affected Pass scope: ReduceCopyMerge. |
| sg_set_ooo_scope | Input | Meaning: Controls OoO scheduling within a MIX subgraph.<br> Description: By assigning an **ooo_scope** to an Operation, adjacent Operations with the same **ooo_scope_id** (not -1) are forcibly merged into the same **ooo_task**, so that adjacent Operations with the same **ooo_scope_id** (not -1) are placed as adjacent as possible on the pipeline generated by OoO scheduling. Parallel branch fusion and cross-**ooo_scope** fusion are not allowed, and Operations with different loop iteration counts cannot be merged. <br> Type: `int`, which sets **ooo_scope_id**. <br> Default value: -1. <br> Value range: -1 or 1 to 100000. <br> `ooo_scope_id`: **ooo_scope** identifier. Adjacent Operations with the **same ooo_scope_id** are grouped into the same **ooo_task**; -1 means the Operation does not participate in **ooo_scope** fusion, and the **ooo_task** partitioning is determined by the partitioning algorithm. <br> Affected Pass scope: OoOSchedule. <br> Configuration requirements: This function takes effect only on MIX subgraphs. When this function is used together with loop unroll, the unroll setting must not exceed 10000. |
| ooo_sched_mode | Input | Meaning: Controls OoO scheduling within a MIX subgraph.<br> Description: Sets the pipeline scheduling mode of **ooo_task** within a MIX subgraph. <br> Type: `str`. <br> Default value: "". <br> Value range: {"", "GAPMIN", "HLF"}. <br> Affected Pass scope: OoOSchedule. <br> Configuration description: When the value is "" (default), scheduling based on topological-order traversal and local search (GapMin scheduling + local-search) is used; when the value is **"GAPMIN"**, only GapMin scheduling is performed and local-search is skipped; when the value is **"HLF"**, Highest Level First scheduling is used (tasks are sorted in descending order by the longest path to the sink, and then EFT insertion scheduling is performed). |
| sg_set_tunevf_mode | Input | Meaning: Controls the behavior mode of the VF (Vector Fusion) tuning passes.<br> Description: Controls the execution behavior of the **TuneTileOpSeqForVF** and **TuneSyncForVF** Passes.<br> Type: `int`. <br> Default value: 0. <br> Value range: {0, 1, 2}. <br> - **0**: Balanced mode, which automatically adjusts the op order based on the op sequence output by the OoO pass and automatically balances the overall performance benefits of pipeline and VF fusion.<br> - **1**: Instruction pipeline priority mode, which does not change the op execution order arranged by OoO.<br> - **2**: VF fusion priority mode, which adjusts the op order as much as possible to ensure a wider range of VF fusion without considering the benefit evaluation of performance modeling.<br> Affected Pass scope: TuneTileOpSeqForVF, TuneSyncForVF. |


## Return Value

None

## Constraints

- Setting timing: It is not required to call before graph compilation starts; the setting can be performed at any time.
- Type safety: The type of the passed **value** must exactly match the type defined for the parameter; otherwise, undefined behavior or runtime errors may occur.
- Scope: Parameter settings are local and only affect the compilation process within the current jit or loop. If not set, the settings are inherited from the upper-level scope.
- Semantic label key: The string key of **setting** must exactly match the semantic_label set on at least one operation through `pypto.set_semantic_label`; otherwise, a compilation error is reported.
- `sg_set_scope` consistency constraint: All operations with the same `scope_id` must have the same `allow_parallel_merge` and `allow_cross_scope_merge` settings; otherwise, a compilation error is reported.
- When `scope_id` is -1, `allow_parallel_merge` and `allow_cross_scope_merge` must be **False**.
- Subgraphs with different `scope_id` values cannot be merged. `allow_cross_scope_merge` only controls the merging of subgraphs with a scope and subgraphs without a scope (`scope_id` = -1).
- Usage instructions for `auto_mix_partition`:
   - Ascend 950PR/Ascend 950DT: Supported.
   - Atlas A3 training products/Atlas A3 inference products: Not supported. Automatic CV mix graph fusion is not performed.
   - Atlas A2 training products/Atlas A2 inference products: Not supported. Automatic CV mix graph fusion is not performed.
- **sg_set_scope** usage instructions:
   - Ascend 950PR/Ascend 950DT: Support scope configuration for pure Vector, pure Cube, and CV mix scenarios.
   - Atlas A3 training products/Atlas A3 inference products: Support scope configuration for pure Vector or pure Cube, but does not support scope configuration for CV mix scenarios.
   - Atlas A2 training products/Atlas A2 inference products: Support scope configuration for pure Vector or pure Cube, but does not support scope configuration for CV mix scenarios.
- **sg_set_ooo_scope** usage instructions:
   - Ascend 950PR/Ascend 950DT: Supported.
   - Atlas A3 training products/Atlas A3 inference products: Not supported, because CV mix graph fusion is not supported.
   - Atlas A2 training products/Atlas A2 inference products: Not supported, because CV mix graph fusion is not supported.
- **ooo_sched_mode** usage instructions:
   - Ascend 950PR/Ascend 950DT: Supported.
   - Atlas A3 training products/Atlas A3 inference products: Not supported, because CV mix graph fusion is not supported.
   - Atlas A2 training products/Atlas A2 inference products: Not supported, because CV mix graph fusion is not supported.
- **sg_set_tunevf_mode** usage instructions:
   - Ascend 950PR/Ascend 950DT: Supported.
   - Atlas A3 training products/Atlas A3 inference products: Not supported, because CV mix graph fusion is not supported.
   - Atlas A2 training products/Atlas A2 inference products: Not supported, because CV mix graph fusion is not supported.
   - **mode=2** only affects the **NeedAdjustOpSeq** judgment in **TuneSyncForVF**, and does not affect **TuneTileOpSeqForVF**.

## Example

```python
   # Function granularity configuration (func{magic}_{order} format).
   pypto.set_pass_options(
       vec_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_l1_reuse_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1},
       cube_nbuffer_setting={"DEFAULT": 4, "func8_0": 1, "func8_1": 1})

   # Pure DEFAULT.
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2})

   # Semantic label key configuration (can coexist with function granularity keys).
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("")
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})
```

### dict Type Configuration Description (Function Granularity Key / Semantic Label Key)

### Function Granularity Key Configuration Description (func{magic}_{order})

#### Overview

Using a key in the `"func{magic}_{order}"` format, you can set the fusion granularity for a **specific isomorphic subgraph group** (hashorder) of a **specific function**, enabling fine-grained configuration across different root functions. The configured **hashOrder** and **subGraphCount** information is directly displayed in the **hashOrder-hint** field of the swimlane diagram in the format `l1ReuseInfo hashOrder: func8_0, subGraphCount: 24`, allowing the fusion strength to be matched based on the number of subgraphs and the number of cores.

#### Key-Value Pair Meaning

Key: A string in the format `"func{magic}_{order}"` or `"DEFAULT"`.<br>

- `"func{magic}_{order}"`: Matches the isomorphic subgraph group whose **hashorder** is **order** in the function whose **functionMagic** is **magic**.<br>
- `"DEFAULT"`: Matches all isomorphic subgraph groups that are not explicitly specified.<br>

Value (N): Indicates the fusion granularity, that is, every N subgraphs in an isomorphic subgraph group are fused into a new subgraph for execution. N=1 indicates no fusion.

#### Format Constraints

- The `func` prefix must be lowercase.
- **magic** and **order** must be integers.
- **magic** and **order** are separated by an underscore `_`.
- Examples of valid keys: `"func0_0"`, `"func123_5"`, `"func8_1"`.
- Example of an invalid key: `"Func0_0"` (uppercase F), `"func_0"` (missing magic), `"func123"` (missing order).

#### Configuration Behavior

When processing subgraph fusion for the current function, the Pass follows the logic of "exact match of `func{magic}_{order}` > `DEFAULT` configuration > automatic processing":<br>

- Exact match: If both **funcMagic** and **hashorder** are matched, fusion is performed according to the corresponding **Value N**.<br>
- `DEFAULT` configuration: If no exact match is found but `DEFAULT` exists in the dictionary, fusion is performed according to the **Value** corresponding to `DEFAULT`.<br>
- Automatic processing: If neither an exact match nor `DEFAULT` exists, the fusion granularity is calculated automatically.<br>

#### Configuration Example

| Configuration | Description |
|------|------|
|`{"DEFAULT": 1}`|All isomorphic subgraph groups skip fusion.|
|`{"DEFAULT": 4, "func8_0": 1, "func8_1": 1}`|By default, four isomorphic subgraphs are fused as one group. In **func8**, the subgraph groups with **hashorder** **0** and **1** skip fusion (not fused).|
|`{"DEFAULT": 2, "func8_1": 4}`|By default, two isomorphic subgraphs are fused as one group. In **func8**, the subgraphs with **hashorder 1** are fused four per group.|
|`{"func8_0": 2}`|In the **func8** function, the subgraphs with **hashorder 0** are fused two per group. Other isomorphic subgraph groups are fused with the fusion granularity automatically calculated based on the number of hardware cores.|

### Semantic Label Key Configuration Description

#### Overview

In addition to the function granularity key, `vec_nbuffer_setting`, `cube_l1_reuse_setting`, and `cube_nbuffer_setting` also support string keys, that is, the semantic label names set through `pypto.set_semantic_label`. String keys allow users to precisely control the fusion strength of the subgraph (multiple subgraphs allowed) where a specific Operation resides, without the need to know its hashorder number.

#### Meaning of String Key-Value Pairs

Key (label): Semantic label name, which must exactly match the `semantic_label` of at least one Operation.<br>
Value (N): Indicates the fusion granularity.<br>

#### Priority Mechanism

The priority of a string key is **higher than** the default configuration of a function granularity key. The processing flow is as follows:<br>

1. First, determine the base fusion granularity of each isomorphic subgraph group based on the function granularity key (`func{magic}_{order}` / `DEFAULT`).<br>
2. Then, the value of the string key **directly replaces** (rather than takes the max of) the fusion granularity of the corresponding subgraph group.<br>
3. When multiple different string labels point to the same isomorphic subgraph group, the maximum value among these labels is used.<br>

#### Semantic Label Behavior of vec_nbuffer_setting / cube_nbuffer_setting

A string key overrides the fusion granularity of the **entire isomorphic subgraph group** to which its corresponding Operation belongs.

#### Semantic Label Behavior of cube_l1_reuse_setting

Unlike `vec_nbuffer_setting` and `cube_nbuffer_setting`, the string key of `cube_l1_reuse_setting` **applies only to the subgraph containing the corresponding label Operation** and is not expanded to the entire isomorphic group. That is, within an isomorphic group, only some subgraphs may be covered by the string key, while other subgraphs retain the value of the function granularity key.

#### Semantic Label Configuration Example

| Configuration                                | Description                                                                 |
|-------------------------------------|----------------------------------------------------------------------|
|{"DEFAULT": 2, "V1": 1}|All isomorphic subgraph groups have a default fusion granularity of **2**; however, the fusion granularity of the isomorphic subgraph group containing the V1 label is replaced with **1**.|
|{"V1": 3}|The fusion granularity of the isomorphic subgraph group containing the V1 label is **3**; the fusion granularity of other isomorphic subgraph groups is automatically computed.|
|{"DEFAULT": 2, "V1": 1, "V2": 3}|The default fusion granularity is **2**; the group containing V1 uses **1**; the group containing V2 uses **3**. If a group contains both V1 and V2 OPs, **max(1, 3) = 3** is used.|

#### Configuration Example

```python
   # Mixed function granularity key and semantic label key configuration.
   pypto.set_semantic_label("V1")
   sij_scale = pypto.mul(sij, softmax_scale)
   pypto.set_semantic_label("") # Change the semantic label to precisely control that only this mul OP has the semantic label "V1".
   ...
   pypto.set_pass_options(vec_nbuffer_setting={"DEFAULT": 2, "V1": 1})

   # Pure semantic label key configuration.
   pypto.set_pass_options(cube_l1_reuse_setting={"MM1": 4})
```

### sg_set_scope Configuration Description

#### Configuration Example

```python
# int format: equivalent to (10, False, False), setting only scope_id.
pypto.set_pass_options(sg_set_scope=10)

# tuple format: scope_id=1, allowing parallel branch fusion but not cross-scope fusion.
pypto.set_pass_options(sg_set_scope=(1, True, False))

# tuple format: scope_id=2, allowing fusion with subgraphs without a scope.
pypto.set_pass_options(sg_set_scope=(2, False, True))

# Restore the default (not participating in scope fusion, automatically determined by the graph fusion algorithm).
pypto.set_pass_options(sg_set_scope=-1)
```

#### Typical Scenarios

##### Scenario 1: Entire Computation Graph Not Partitioned

When the entire computation graph needs to remain unpartitioned, data tiling produces multiple parallel branches that have no direct data dependency between them, and these branches are split into independent subgraphs by the partitioning algorithm by default. It is recommended to set `sg_set_scope=(scope_id, True, False)` so that, through `allow_parallel_merge=True`, the parallel branch Operations with the same **scope_id** are fused into the same subgraph.

##### Scenario 2: CV Mix Scenario; Constructing a Mix Subgraph to Reduce GM Data Movement (Ascend 950PR/Ascend 950DT)

When both the preceding and following operations of a Cube operation are Vec operations, the goal is to construct a Mix subgraph containing both Cube and Vec, avoiding repeated data movement of intermediate results on GM. Depending on whether the scope boundary is explicit, the following two cases are distinguished:

**Scenario 2.1: Explicit Scope Boundary**

When the boundary between a Cube operation and its adjacent Vec operations can be explicitly defined, use `(scope_id, False, False)` to mark the boundary, forcing the Cube and the adjacent Vec into the same subgraph to form a Mix subgraph.

```python
# Explicitly mark the Cube and the adjacent Vec as the same scope to form a Mix subgraph.
pypto.set_pass_options(sg_set_scope=(1, False, False))
# ... Cube operations...
# ... adjacent Vec operations...
pypto.set_pass_options(sg_set_scope=-1)
```

**Scenario 2.2: Unclear Scope Boundary; Only the CV Fusion Boundary Marked**

When the boundary cannot be clearly defined, but the Vec subgraphs before and after the Cube need to be fused with the Cube subgraph to form a Mix subgraph, use `(scope_id, False, True)` to mark the Cube and its adjacent Vec as fusion anchors. With `allow_cross_scope_merge=True`, the Vec in this scope subgraph can be fused with the preceding and following Vec subgraphs that have no scope (scope_id=-1), forming a larger subgraph to reduce GM data movement, and forming a Mix subgraph together with the Cube subgraph.

```python
# Preceding Vec operation (scope_id=-1, determined automatically by the partitioning algorithm)
vec_out = some_vec_op(x)

# Mark the CV fusion anchor and allow fusion with adjacent subgraphs without a scope.
pypto.set_pass_options(sg_set_scope=(1, False, True))
# ... Cube operation...
matmul_result = pypto.matmul(vec_out, w)
# ... Adjacent Vec operation...
pypto.set_pass_options(sg_set_scope=-1)

# Subsequent Vec operation (scope_id=-1), which can be fused with the scope subgraph above into a larger subgraph.
result = other_vec_op(add_result)
```

### **sg_set_ooo_scope** configuration description

#### Configuration Example

```python
# Set the ooo_scope_id.
pypto.set_pass_options(sg_set_ooo_scope=10)

# Restore the default (not participating in ooo_scope fusion, automatically determined by the partitioning algorithm).
pypto.set_pass_options(sg_set_ooo_scope=-1)
```

#### Typical Scenario: Controlling the Execution Order of Operations

As shown in the following example, by wrapping **mul** and **exp** in the same **ooo_scope**, you can make **exp** execute before **add** in the OoO scheduling result.
```python
# Because ooo_scope needs to be enabled within the MIX subgraph, use sg_set_scope to construct the MIX subgraph.
pypto.set_pass_options(sg_set_scope=1)
# Cube operation.
matmul_result = pypto.matmul(a, b)
pypto.set_pass_options(sg_set_ooo_scope=1)
# Vec operation (ooo_scope_id=1).
c = pypto.mul(matmul_result, scale)
pypto.set_pass_options(sg_set_ooo_scope=-1)
d = pypto.add(c, bias)
pypto.set_pass_options(sg_set_ooo_scope=1)
# Vec operation (ooo_scope_id=1).
e = pypto.exp(c)
pypto.set_pass_options(sg_set_ooo_scope=-1)
pypto.set_pass_options(sg_set_scope=-1)
```

### sg_set_tunevf_mode Configuration Description

#### Configuration Example

```python
# Instruction pipeline priority mode. Does not change the op execution order scheduled by OoO.
pypto.set_pass_options(sg_set_tunevf_mode=1)

# VF fusion priority mode. Ignores the benefit evaluation of performance modeling and adjusts the op order as much as possible to ensure a wider range of VF fusion.
pypto.set_pass_options(sg_set_tunevf_mode=2)

# Restore the default behavior, that is, the balanced mode.
pypto.set_pass_options(sg_set_tunevf_mode=0)
```
