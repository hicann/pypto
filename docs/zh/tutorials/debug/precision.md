# 精度调试

## 简介

当PyPTO算子执行后无功能告警或报错，但输出数据不符合预期时，可基于以下方法进行精度问题的定界和定位。精度问题主要来源于两个方面：

- 功能错误：硬件静默故障、软件静默功能问题、公式实现错误引起的明显数据错误或误差。
- 计算误差：数据类型、算法（切分、累积、公式近似）差异等引起的明显数据误差。

## 整体流程

![](../figures/zh-cn_image_0000002532402113.png)

## 确认问题合法性

**步骤一**：确认问题判定方式是否合理（经验判断与实验证明）。

- 误差阈值是否使用合理，例如：
    - 存在bfloat16等低精度数据类型的算子使用了float32的误差阈值。
    - 计算路径较深的算子使用了小算子的误差阈值。
- 对于不合理部分根据经验进行修正，若问题消失则完成精度调试。

**步骤二**：确认问题可稳定复现。

- 重复执行多轮输出数据一致且为异常值。
- 更换其它环境多轮输出仍旧一致且为异常值。
- 若无法稳定复现则明确为功能问题，建议中止精度调试。

## 基础预检

基础预检为指导性说明，指出常见但易被忽视的高概率出错问题。如果用户或调试人员确认相应检查项无误，可以跳过这些步骤。

1. 借助asys工具检查硬件问题。
    1. 使用硬件自检工具排除硬件安装问题。
    2. 使用硬件压测工具排除硬件故障。

2. 检查软件问题。
    1. 基于安装指导确认软件版本正确。
    2. 执行项目中example用例，确认结果正确。

3. 检查用户侧是否引入问题。
    1. 多方评审算子代码，确认：
        - 计算过程与算法原型一致。
        - 数据类型、计算类型与竞品实现一致（若无竞品实现，需由提供算子实现方案的设计人员完成类型确认）。

    2. 若无法确认，则分析后续发现的问题点时，需额外分析是否为用户侧引入。

## 规避已知问题

精度调试前，应确保已规避当前软件存在的已知问题，详细请参见[已知问题](../appendix/issue.md)。

## 缩小问题规模

缩小问题规模通常是一个可选步骤，旨在简化问题，提高复现和定位的效率。

- 缩小问题规模后需能复现同样问题，然后继续进行后续的工具自检或人工调试
- 对于缩小后出现新问题的情况，建议尝试其它缩小方法以复现原始问题，不建议将新问题纳入关键定位流程。

然而，在某些情况下，缩小问题规模是必选步骤，例如对于较大的模型：

- 主机内存不足导致自检工具无法执行。
- 文件存储空间过小导致自检工具无法保存中间计算数据。
- 其它分析流程或工具的耗时超过主观容忍范围，甚至无法执行等阻塞式情况。

通常通过以下方法缩小问题规模：

- 减少子图数量和大小。例如减少loop的次数或减少cube/vector Tiling块的个数（即增大TileShape的大小）。
- 裁剪模型。例如调小模型的Shape规格，如batch\_size、seq\_len等。
- 采用二分法移除尾部计算。
    1. 按模型计算的顺序，采用二分法移除靠近尾部的计算，并将断开的输出加入算子的输出列表。
    2. 执行算子并观察、分析新的输出列表
        - 如果数据正常（无inf/nan、无主观认为随机的值、或与参考基准数据误差较小）则返回上步继续二分操作。
        - 如果数据存在异常，则将代码恢复到本次移除前的状态，作为最新的候选问题场景。
        - 如果裁剪后的模型规模已经很小，可以停止二分操作并选择最新的候选问题场景进行后续的定位。

## 工具自检和分析

### 工具简介

PyPTO在计算图编译的各Pass阶段拥有完整的中间表示，可翻译成第三方计算代码，并在其它计算单元（例如Host CPU）上模拟计算过程。该工具通过模拟计算结果与基准数据的误差对比，可以检测算子异常或者某个Pass的处理结果是否存在异常，并定位首个出现异常的计算节点。

主要特性及使用场景：

- Tensor Graph校验：用于校验算子代码、框架前端处理的正确性。基于用户提供的基准（golden）输入输出数据，与Tensor Graph模拟计算的最终结果对比检测整体计算的正确性。常用于以下情况：
    - 当用户存在可用的算子基准（golden）输入、输出数据时，可先使能粗检特性粗略排除算子代码、框架前端处理是否引入差异。

- Pass阶段校验：用于自检Pass的正确性。基于各Pass模拟计算的结果，对比检测Pass正确性及异常计算节点。常用于以下情况：
    - 当用户算子精度刚刚出现问题且没有明确方向，可先使能自检特性排除Pass处理阶段是否引入潜在错误。
    - 当用户大致明确某个Pass出问题时，使能自检特性获取该Pass及前序Pass的模拟计算中间数据，对比数据找出潜在出问题的计算操作。

- 中间结果分析：指定单个计算结果，保存到文件或者以可读形式打印到输出、日志。
    - 当Tensor Graph校验失败时，可使用pass_verify_print/pass_verify_save特性打印、保存模拟计算的中间数据，对比数据找出潜在出问题的计算操作。

### 使用约束

当前精度调试工具存在以下限制（完整计算流表示仅保存在pass运行上下文中），无法使用检测功能：

- 不支持上板执行的中间数据检查，仅支持前端及pass的检查。
- 不支持特定pass，特定pass（例如SubgraphToFunction）属于中间的优化过程缺少完整计算信息，工具内部做自动跳过处理。
- 不支持pass间的自动对比校验（需人工进行数据对比）。
- 不支持程序退出后在任意运行环境构造并模拟计算。需在算子编译期间，所对应的主机CPU及进程上构造并模拟计算。
- 不支持基于昇腾AI处理器调用Ascend C构造并模拟计算。
- 不支持基于GPU构造并模拟计算。
- 不支持包含GATHER_IN_UB和GATHER_IN_L1两个operation的校验。
- 如ExpandFunction校验结果出现B200BU报错，则该场景仅在InferDynShape后校验结果有效。
- inplace的op目前只确保pass24及以后的pass校验通过。

### 环境准备

最新master分支代码及0.1.1之后版本（不含0.1.1版本）支持在运行时在线编译精度工具所需C++二进制，不需重新编译安装PyPTO,但需确认在线编译所需的构建工具符合以下要求：

    - cmake >= 3.16.3
    - make
    - g++ >= 9.4.0

早期PyPTO源码需要重新编译并安装PyPTO后才能使用该工具。

1. 确认GCC安装并升级到9.4.0或更高版本。
2. 重新通过源码编译安装PyPTO。主要区别是在编译安装命令中增加选项--no-build-isolation，其他操作请参见[编译安装](../../install/prepare_environment.md)。

    ```bash
    python3 -m pip install . --verbose --no-build-isolation
    ```

### 工具使用操作步骤

1. 开启精度调试开关。参考样例为：[hello_world.py](../../../../examples/00_hello_world/hello_world.py)。

    ```python
    ...
    verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        ...
    }

    @pypto.frontend.jit(verify_options=verify_options)
    def add_kernel(
        input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
        input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
        out: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = input0 + input1

    ...
    ```

    **verify_options参数说明**

    | 参数名 | 类型 | 默认值 | 说明 |
    |--------|------|--------|------|
    | `enable_pass_verify` | bool | False | 总体使能开关，决定所有`pass_verify_*`选项和接口是否生效。必须设置为`True`才能使其他参数生效 |
    | `pass_verify_save_tensor` | bool | False | 是否将模拟计算数据存盘。设置为`True`时会在`{work_path}/output/output_*/`目录下生成`verify_*`目录 |
    | `pass_verify_save_tensor_dir` | str | "{RUNNING_DIR}/output/output_{TS}" | 检测结果及数据的保存路径。可指定绝对路径 |
    | `pass_verify_pass_filter` | List[str] | 空 | 配置待自检的Pass名称列表。不指定则默认校验特定pass；指定`"all"`则校验所有pass；指定`[]`不校验pass只校验tensor_graph |
    | `pass_verify_error_tol` | List[float] | [1e-3, 1e-3] | 精度对比的容差配置。第一个值为相对误差容差（rtol），第二个值为绝对误差容差（atol） |

2. 设置golden数据（可选）

    如果需要进行tensor_graph验证，需要设置golden数据：

    ```python
    ...
    def test_add():
        shape = (1, 16, 1, 64)
        input_data0 = torch.rand(shape, dtype=torch.float)
        input_data1 = torch.rand(shape, dtype=torch.float)
        torch_add = torch.add(input_data0, input_data1)
        # 设置golden数据
        pypto.set_verify_golden_data(goldens=[None, None, torch_add])

        input_data0 = input_data0.to('npu')
        input_data1 = input_data1.to('npu')
        out = torch.empty(shape, dtype=torch.float, device='npu')

        add(input_data0, input_data1, out)
    ...
    ```

    **set_verify_golden_data接口说明**

    **函数原型**：

    ```python
    set_verify_golden_data(in_out_tensors=None, goldens=None)
    ```

    **参数说明**：

    | 参数名 | 类型 | 说明 |
    |--------|------|------|
    | `in_out_tensors` | List[Union(pypto.Tensor, torch.Tensor)] | 将用户（可选）执行算子时实际的输入、输出列表按照相同位置对应地设置到检测工具。jit调用模式下，该选项不需设置 |
    | `goldens` | List[Union(pypto.Tensor, torch.Tensor)] | 将用户已有的计算基准数据（golden）输出设置到工具中做对比检测。该列表与算子输入、输出参数列表的长度一致、位置对应。若相应位置设置为None，表示跳过该位置的数据对比。**注意：torch.Tensor的device属性需为CPU，不支持NPU** |

    **约束说明**：
    - 该函数需设置`pypto.set_verify_options(enable_pass_verify=True)`后生效

3. 执行修改后用例。

    ```bash
    python3 examples/00_hello_world/hello_world.py
    ```

4. 打印类似以下输出，指示对应的自检结果为通过（PASS）、未通过（FAIL\(ED\)）或跳过校验（NO\_COMPARE）：

    ```text
    2025-mm-dd HH:MM:SS:xxx V | tensor_graph Verify for 3 data view list index 0 result NO_COMPARE
    2025-mm-dd HH:MM:SS:xxx V | tensor_graph Verify for 3 data view list index 1 result NO_COMPARE
    2025-mm-dd HH:MM:SS:xxx V | tensor_graph Verify for 3 data view list index 2 result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_00_RemoveRedundantReshape Verify result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_01_AutoCast Verify result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_02_InferMemoryConflict Verify result PASS
    ...
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_34_InsertSync Verify result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_35_MixSubgraphSplit Verify result PASS
    2025-mm-dd HH:MM:SS:xxx V | function_TENSOR_loop_0_Unroll1_PATH0_hiddenfunc0_8.pass_36_CodegenPreproc Verify result PASS
    ```

5. 执行结束后，在$\{work\_path\}/output/output\_\*/目录（\*代表时间戳）下生成verify\_\*目录，存放检测结果文件与日志。

    ```text
    ├── tensor_graph # 保存前端初始计算图模拟计算后的中间数据，作为基础数据
    │   ├── *.data
    │   └── ...
    ├── verify_graph_data_metainfo.csv # 结果报告，保存中间数据元信息及对应数据文件名
    ├── verify_graph_result_brief.csv # 精度比对摘要（PASS/FAIL/NO_COMPARE、误差统计等）
    ├── verify_graph_result_brief.log # 精度比对异常详情（失败项、异常路径、错误明细）
    ├── interpreter.log # interpreter模块拆分日志（默认记录ERROR / EVENT）
    ├── Pass_{PASS_SEQ}_{PASS_NAME} # 保存中间pass计算图模拟计算后的中间数据，作为待测数据
    │   ├── *.data
    │   └── ...
    ```

    其中，`verify_graph_result_brief.log`和`interpreter.log`位于同一个`verify_*`目录下：
    - `verify_graph_result_brief.log`：偏向校验结果摘要与异常明细（对比失败、异常路径）。
    - `interpreter.log`：偏向interpreter执行过程中的拆分日志（当前默认仅ERROR/EVENT落盘）。

6. 后续处理建议。

    对于tensor_graph校验结果中标记FAIL的情况，建议：

    1. 多方评审检查PyPTO前端代码的正确性。
    2. 在前端代码无明显异常的前提下，可使用`pass_verify_print`和`pass_verify_save`保存/打印中间结果进行进一步分析（详见步骤7）。

    对于tensor_graph校验结果通过，Pass阶段校验结果中标记FAIL的情况，建议：
    1. 建议收集相关结果信息，并提交ISSUE进行处理。

7. 使用`pass_verify_print`和`pass_verify_save`分析中间结果（可选）。

    **使用场景**：当Tensor Graph校验失败时，可使用这两个接口打印、保存模拟计算的中间数据，对比数据找出潜在出问题的计算操作。

    **重要说明**：
    - `pass_verify_print`和`pass_verify_save`保存的是**tensor graph验证阶段模拟计算的结果**
    - 这些结果是在主机CPU上通过模拟执行计算图得到的
    - **与实际在NPU上板执行的结果可能存在差异，主要用于算法逻辑验证**

    **使用示例**：

    ```python
    @pypto.frontend.jit(verify_options=verify_options)
    def add_kernel(
        input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
        input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
        out: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        # 保存中间结果到文件
        pypto.pass_verify_save(input1, "input1_by_pass_verify")
        # 打印中间结果到控制台
        pypto.pass_verify_print(input0)
        out[:] = input0 + input1

    def add(input_data0, input_data1, out):
        add_kernel(input_data0, input_data1, out)

    def test_add():
        shape = (1, 4, 1, 64)
        input_data0 = torch.rand(shape, dtype=torch.float, device='npu')
        input_data1 = torch.rand(shape, dtype=torch.float, device='npu')
        out = torch.empty(shape, dtype=torch.float, device='npu')

        add(input_data0, input_data1, out)
    ...
    ```

    **执行修改后用例**

    ```bash
    python3 examples/00_hello_world/hello_world.py
    ```

    **控制台输出示例**：

    ```text
    input0:<64x64xFP16/64x64xFP16>
    [[0.03955 0.6094 0.1519 ... 0.7339 0.8789 0.8662]
     [0.6284 0.01465 0.6333 ... 0.2422 0.03516 0.8423]
     [0.231 0.02686 0.6055 ... 0.7466 0.2529 0.2231]
     ...
     [0.3477 0.4243 0.05273 ... 0.9287 0.1138 0.5083]
     [0.05273 0.9941 0.4985 ... 0.8345 0.8613 0.188]
     [0.3184 0.8047 0.833 ... 0.7734 0.2578 0.1392]]
    ```

    **生成的文件结构**：

    执行结束后，在`{work_path}/output/output_*/`目录（*代表时间戳）下生成`tensor/`目录：

    ```text
    ├── tensor/
    │   ├── input1_by_pass_verify.data     # 保存的指定模拟计算数据，格式为Tensor数据的直接内存转储
    │   ├── input1_by_pass_verify.csv      # 模拟计算数据的元数据，包括数据类型、shape信息
    ```

    **后续数据处理建议**：

    根据元数据信息使用常用的`torch.from_file()`、`numpy.load()`等接口打开数据文件并转换为可解析的数值，再进一步进行通常开发者使用的数据分析方法，例如：检查异常数据的偏移规律、异常数据的值特征（inf/nan/zero等）。

## 上板执行tensor dump

### 1. 功能概述

支持在上板执行时dump leaf function的输入输出数据，用于精度问题定位，支持和模拟计算结果对比分析。

### 2. 启用方式

```python
import os

# 设置环境变量启用上板dump,或者执行前单独设置环境变量export PTO_DATADUMP_ENABLE=true
os.environ["PTO_DATADUMP_ENABLE"] = "true"

# 配置验证选项
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    verify_options={
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True
    }
)
def kernel(...):
    ...
```

### 3. Dump数据输出路径

```
output/output_*/dump_tensor_*/device_{deviceId}/
└── {taskId}_{seqNo}_{callopMagic}_{rootHash}_{funcHash}_{rawMagic}_{timeStamp}_{dataType}_{input/output}{index}.tdump
```

### 4. 数据处理工具

**工具位置：** `tools/verifier/parse_dump_tensors.py`

**主要功能：**

- 解析dump的二进制数据（.tdump文件），提取tensor数据为.data文件
- 自动合并分片tensor为完整的raw tensor（针对多个task处理同一raw tensor的场景）
- 支持codegen pass tensor对比验证（需配合`enable_pass_verify`使用）

**使用方法：**

```bash
# 基本用法（未使能enable_pass_verify，不进行验证）
python3 tools/verifier/parse_dump_tensors.py \
    --dump_tensor_path output/output_20260101120000/dump_tensor_20260101120000/device_0

# 带验证的用法（需先开启enable_pass_verify并运行算子）
python3 tools/verifier/parse_dump_tensors.py \
    --dump_tensor_path output/output_20260101120000/dump_tensor_20260101120000/device_0 \
    --verify_path output/output_20260101120000/verify_20260101120000
```

**参数说明：**

| 参数 | 必需/可选 | 说明 | 默认值 |
|------|-----------|------|--------|
| `--dump_tensor_path` | 必需 | dump数据目录路径，指向`device_x`目录 | 无 |
| `--verify_path` | 可选 | verify结果目录路径（包含verify_graph_data_metainfo.csv）| `""`（不对比验证） |

**输出文件：**

```
output/output_*/dump_tensor_*/device_0/
├── *.data                                # 提取的tensor数据文件
├── raw_{rawMagic}_{dataType}_{ioflag}.data  # 合并后的raw tensor（如有分片）
└── ../                                   # 上级目录生成对比结果报告
    └── verify_task_result_cmp~{timestamp}.csv  # 对比验证结果报告
```

**verify_task_result_cmp~{timestamp}.csv字段说明：**

字段前缀说明：

- `B>`前缀：表示上板dump的原始数据
- `A>`前缀：表示验证数据（来自pass verify）
- `AB>`前缀：表示对比验证结果

**基础信息字段：**

| 字段 | 说明 |
|------|------|
| B>taskId | 任务ID |
| ROOT_CALL:opmagic | 算子调用magic标识 |
| ROOT_CALL:rawmagic | 原始tensor magic标识 |
| B>validshape | tensor实际shape |
| B>offset | tensor在raw tensor中的偏移 |
| B>rawShape | 原始完整tensor的shape |
| B>tensorAddr | tensor内存地址 |
| B>datatype | 数据类型（字符串，如FP32、INT8） |
| IO_FLAG | 输入/输出标记（input/output） |
| B>seqNo | 序列号 |
| B>TIMESTAMP | 时间戳 |
| B>funcId | Function ID |
| ROOT_FUNC:hash | Root Function hash值 |
| FUNC:hash | Function hash值 |

**验证对比字段（启用--verify_path时）：**

| 字段 | 说明 |
|------|------|
| A>PHASE_NAME | 验证数据的阶段名称（如Pass_36_CodegenPreproc） |
| A>FILENAME | 验证数据文件路径 |
| A>datatype | 验证数据的数据类型 |
| A>validshape | 验证数据的shape |
| AB>RESULT | 对比结果：PASS、FAIL、NO_CMP |
| error_count | 误差元素数量（对比失败时） |
| error_rate | 误差元素占比（对比失败时） |
| max_abs_error | 最大绝对误差（对比失败时） |
| max_rel_error | 最大相对误差（对比失败时） |
| mean_abs_error | 平均绝对误差（对比失败时） |
| mean_rel_error | 平均相对误差（对比失败时） |
| result_reason | 未对比原因（NO_CMP时，如"unsupported dtype: BOTTOM"） |

**对比验证流程：**

1. **数据匹配**：通过`ROOT_CALL:opmagic`、`ROOT_CALL:rawmagic`、`IO_FLAG`、`B>offset`匹配上板数据与验证数据
2. **容差配置**：根据数据类型自动选择容差
   - FP32/FP64：标准容差（rtol=1e-3, atol=1e-3）
   - FP16/BF16/FP8：放宽容差（rtol=1e-2, atol=1e-2）
3. **Shape处理**：自动处理shape不一致的对比（取公共部分）
4. **不支持类型**：HF4、HF8、BOTTOM等类型标记为NO_CMP

**Raw Tensor合并说明：**

当多个task处理同一个raw tensor的不同分片时，脚本会自动：

1. 按`ROOT_CALL:rawmagic`分组
2. 根据`B>offset`和`B>validshape`计算切片位置
3. 合并所有分片数据到完整raw tensor
4. 生成的文件命名为：`raw_{rawMagic}_{dataType}_{ioflag}.data`

## 算子级别的输入输出tensor dump

### 1. 功能概述

支持整网中算子级别的输入输出上板dump的能力。

### 2. 启用方式

在脚本运行目录下创建acl.json文件，内容如下：

```json
{
    "dump":{
        "dump_path":"/your/path",
        "dump_mode":"all",
        "dump_debug":"off",
        "dump_op_switch":"on"
    }
}
```

在要执行的用例test.py中添加如下配置：

```python
import torch
import torch_npu

torch.npu.init_dump()
torch.npu.set_dump("acl.json")
```

### 3. Dump数据输出路径

数据输出路径就是acl.json里面配置的dump_path，在该路径下会生成如下文件：

```
/your/path
└── 20260415084134/0
    └── TENSOR_batchmatmul_3d_kernel.TENSOR_batchmatmul_3d_kernel.29.46.1776242496294291
```

调用CANN已有的工具解析该文件，命令如下：

```bash
python3 /${CANN_PACKAGE_PATH}/Ascend/cann-9.0.0/tools/operator_cmp/compare/msaccucmp.py convert -d /your/path/20260415084134/0 -out /your/path/20260415084134/0/out
```

解析后生成如下npy文件：

```
/your/path
└── 20260415084134/0
    ├── out/
    │   ├── TENSOR_batchmatmul_3d_kernel.TENSOR_batchmatmul_3d_kernel.29.46.1776242496294291.input.0.npy
    │   ├── TENSOR_batchmatmul_3d_kernel.TENSOR_batchmatmul_3d_kernel.29.46.1776242496294291.input.1.npy
    │   └── TENSOR_batchmatmul_3d_kernel.TENSOR_batchmatmul_3d_kernel.29.46.1776242496294291.input.2.npy
    └── TENSOR_batchmatmul_3d_kernel.TENSOR_batchmatmul_3d_kernel.29.46.1776242496294291
```
