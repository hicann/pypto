# pypto.experimental.set\_runtime\_options

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->

## 功能说明

该接口用于设置**实验性运行时配置**。它将`tile_fwk_config.json`中运行时里尚未稳定的参数，转变为可编程接口。后续新增的运行时实验特性也通过本接口扩展，不进入`pypto.frontend.jit(runtime_options=...)`的稳定参数表。

## 函数原型

```python
set_runtime_options(
    *,
    stitch_function_num_per_pool: Optional[list[int]] = None,
)
```

## 参数说明

| 参数名               | 输入/输出 | 说明                                                                 |
|----------------------|-----------|----------------------------------------------------------------------|
| stitch_function_num_per_pool | 输入      | 含义：分别设置Workspace三类内存池支持的stitch深度。该参数为实验特性，后续版本可能存在变更，暂不支持于生产环境。<br> 说明：三个元素分别独立控制对应内存池的深度，不表示字节数。配置格式为`[root_inner_depth, assemble_outcast_depth, exclusive_outcast_depth]`，各维含义如下：<br> - **root_inner_depth**：表示单次DeviceTask内，该池需预留的包含RootInner的root function个数上限，RootInner表示单个root function产生、且不会被其他root function使用的内部临时tensor数据。<br> - **assemble_outcast_depth**：表示单次DeviceTask内，该池需预留的包含Assemble outcast的root function个数上限，Assemble outcast表示单个root function产生、会被其他root function使用的tensor数据，由Assemble写入。<br> - **exclusive_outcast_depth**：表示单次DeviceTask内，该池需预留的包含Exclusive outcast的root function个数上限，Exclusive outcast表示单个root function产生、会被其他root function使用的tensor数据，非Assemble写入。<br> `[0, 0, 0]`表示关闭精细Workspace模式；任意元素不为0时启用精细Workspace模式。启用后，有实际内存需求的池对应值必须大于0，否则编译时会因内存预留不足报错；无实际内存需求的池可配置为0。详细配置方法参见[示例3: 精细Workspace模式](#stitch_function_num_per_pool_detail)。<br> 类型：list of int，固定包含3个元素 <br> 取值范围：每个元素 0 ~ 1024 <br> 默认值：`[0, 0, 0]` <br> 影响pass范围：NA |

## 返回值说明

void：Set方法无返回值。设置操作成功即生效。

## 约束说明

- 类型安全：必须确保传入的`stitch_function_num_per_pool`为包含3个整数的list或tuple，每个元素取值范围为0～1024，且不能使用bool。
- 内存配置项关系：
   - `stitch_function_num_per_pool`和稳定`runtime_options`下的`stitch_function_max_num`、`max_workspace_kb`共同影响Workspace预留与stitch行为，职责不同，不宜混用。
   - `stitch_function_max_num`：控制单次DeviceTask内可处理的最大root function个数；在未启用精细Workspace模式、且未启用`max_workspace_kb`内存驱动时，三个内存池的默认最大深度取该值。
   - `max_workspace_kb`：控制Workspace的总量上限；取值需大于算子可运行的最小Workspace时启用内存驱动，由配置的Workspace上限反推单次DeviceTask可处理的最大root function个数，启用后`stitch_function_max_num`配置会失效。
   - `stitch_function_num_per_pool`：分别控制RootInner / Assemble outcast / Exclusive outcast三个内存池；启用精细Workspace模式后，分别指定三个池的root function个数，不再要求三者等于同一数量，启用后，稳定`runtime_options`下的`stitch_function_max_num`与`max_workspace_kb`配置会失效。

## 调用示例

```python
pypto.experimental.set_runtime_options(stitch_function_num_per_pool=[64, 1, 1])
```

### 示例3: 精细Workspace模式 <a id="stitch_function_num_per_pool_detail"></a>

以下示例用于说明默认与精细控制的差异。问题规模与循环约定如下（后文预算数值均基于该设定）：

```python
B_STATIC, L_STATIC, H_STATIC, D_STATIC = 1, 64, 1, 16

pypto.experimental.set_runtime_options(stitch_function_num_per_pool=[64, 1, 1])

@pypto.frontend.jit
def k_tmp_to_d_emb(
    dy: pypto.Tensor([B_STATIC, L_STATIC, H_STATIC, D_STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    output1: pypto.Tensor([B_STATIC, L_STATIC, D_STATIC], pypto.DT_FP32),
    output2: pypto.Tensor([B_STATIC, L_STATIC, D_STATIC], pypto.DT_FP32),
):
    tmp_assemble = pypto.tensor([B_STATIC, L_STATIC, H_STATIC, D_STATIC], output1.dtype, "tmp_assemble")
    tmp_exclusive = pypto.tensor([B_STATIC, L_STATIC, H_STATIC, D_STATIC], output2.dtype, "tmp_exclusive")

    # Loop0：Exclusive write
    for i_idx, t in pypto.loop_unroll(0, 1, 1, name="l_loop_0"):
        pypto.set_vec_tile_shapes(1, 64, 1, 256)
        tmp_exclusive[:] = pypto.add(dy, dy)

    # Loop1：Assemble write
    for j_idx, t in pypto.loop_unroll(0, L_STATIC, 1, name="l_loop_1"):
        pypto.set_vec_tile_shapes(1, 64, 1, 256)
        dy_v = dy[0, j_idx : j_idx + t, 0]
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dx = pypto.matmul(dy_v, weight[0], pypto.DT_FP32, b_trans=True)
        pypto.set_vec_tile_shapes(1, 64, 1, 512)
        tmp_assemble[0, j_idx : j_idx + t, 0] = dx + 0.0

    # Loop2：Read
    for k_idx, t in pypto.loop_unroll(0, L_STATIC, 1, name="l_loop_2"):
        pypto.set_vec_tile_shapes(1, 64, 1, 512)
        output1[0, k_idx : k_idx + t] = tmp_assemble[0, k_idx : k_idx + t, 0]
        output2[0, k_idx : k_idx + t] = tmp_exclusive[0, k_idx : k_idx + t, 0]
```

该示例中的tensor内存分布如下：

| Tensor | 数据归属 |
| --- | --- |
| dy、weight、output1、output2 | 输入参数，不进Workspace内存池 |
| dy_v | dy的切片视图，复用同一块内存 |
| dx | RootInner |
| tmp_assemble | Assemble outcast |
| tmp_exclusive | Exclusive outcast |

将stitch_function_num_per_pool配置为非全零三元组后，三个内存池的最大深度相互独立，可分别按实际需求设置。对本示例，运行时根据内存峰值得到的推荐配置为[64, 1, 1]，说明如下：

- **root_inner_depth**：由默认的128下调为64。dx只存在于loop1的root function内，无需为loop0、loop2预留深度。
- **assemble_outcast_depth**：由默认的128下调为1。tmp_assemble在循环外创建，作为全局变量，由多个root function共享同一块内存。
- **exclusive_outcast_depth**：由默认的128下调为1。tmp_exclusive为非Assemble类型的整块写入，仅在loop0的一次循环（即一个root function）中产生。

经过上述设置，可以有效提升workspace内存的使用率。
