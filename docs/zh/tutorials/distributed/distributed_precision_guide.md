# PyPTO 通信算子精度定位问题指南

## 概述

PyPTO 通信算子用于实现多卡间的数据传输与同步，是分布式计算场景的核心组件。与纯计算算子相比，通信算子涉及跨进程数据交换、信号量同步、切块策略等复杂机制，其精度问题定位具有独特的挑战性。

本指南提供系统化的通信算子精度问题定位方法，帮助开发者快速定位和解决精度异常。

## 通信算子精度问题特点

与纯计算算子相比，通信算子精度问题具有以下特殊性：

| 特性             | 纯计算算子   | 通信算子                                            |
| ---------------- | ------------ | --------------------------------------------------- |
| **执行环境**     | 单卡执行     | 多卡并行执行                                        |
| **数据流**       | 单向数据传递 | 多向数据交换                                        |
| **同步机制**     | 不需要卡间同步 | 需卡间信号同步                                      |
| **内存管理**     | 普通内存     | 共享内存                                            |
| **问题来源**     | 计算逻辑错误 | 计算逻辑错误、同步问题、数据切分问题                |

## 精度定位整体流程

```.txt
精度问题定位流程
    │
    ├─ 通信算子精度问题定位思路
    │   ├─ 基础预检
    │   │   ├─ 精度误差阈值检查
    │   │   ├─ Device 日志检查
    │   │   └─ 通信算子标准测试用例检查
    │   ├─ 检查精度问题的复现情况
    │   └─ 减小问题规模
    │
    ├─ 通信算子维测方法
    │   ├─ 通过 shmem_get 读取 Win 区数据
    │   ├─ 利用 aicore_print 打印 TileOp 信息
    │   └─ 定位出现问题的OP
    │
    ├─ 典型错误场景
    │   ├─ 切块语义不匹配
    │   ├─ 多轮通信需 shmem_clear_data
    │   ├─ 依赖关系配置错误
    │   └─ Golden 与算子计算逻辑不一致
    │
    └─ 回归与验证
        ├─ 精度验证要点
        └─ 回归测试范围
```

## 通信算子精度问题定位思路

在进行精度定位前，需先完成基础检查，然后根据问题的可复现性特点和问题规模进行分析。

### 基础预检

在进行精度定位前，需先完成以下基础检查，确保问题确实是精度问题而非功能异常或环境问题。

#### 精度误差阈值检查

- [ ] 阈值检查

**检查目的**：确保误差阈值设置符合数据类型的精度范围和累加操作的特性。

通信算子中使用 `shmem_put` 的 `AtomicType.ADD` 进行累加时，累加精度由 `shmem_tensor` 的 dtype 决定，阈值需根据累加方式和累加次数调整：

| 累加精度 | 建议阈值 | 说明 |
|---------|---------|------|
| FP32 累加 | ≤ 5e-3 | FP32 精度较高，累加误差较小，阈值应满足千分之五（0.005） |
| FP16 累加 | 按累加次数计算 | FP16 mantissa 10 bit，相对精度约 1/1024（≈0.001），需按累加次数计算阈值 |
| BF16 累加 | 按累加次数计算 | BF16 mantissa 7 bit，相对精度约 1/128（≈0.0078），需按累加次数计算阈值 |

**低精度累加阈值计算方法**：

> [!CAUTION]注意
> 以下计算采用线性误差估计模型，实际浮点累加误差遵循更复杂的规律（Kahan 求和公式），当数据范围差异大时，误差可能显著偏离线性估计。建议在实际验证时，根据数据分布特点调整阈值。

```python
def calculate_accumulation_threshold(
    dtype: str, 
    world_size: int, 
    safety_factor: float = 2.0
):
    """
    计算低精度累加的理论误差阈值
    
    精度特性：
    - FP16: mantissa 10 bit，相对精度约为 1/1024 ≈ 0.001
    - BF16: mantissa 7 bit，相对精度约为 1/128 ≈ 0.0078
    
    每次累加的舍入误差约为精度类型的 0.5 倍（ulp）
    累加 N 次后，误差粗略估计为 N * 0.5 * dtype_precision
    
    Args:
        dtype: 数据类型，"FP16" 或 "BF16"
        world_size: 累加次数（通信的 rank 数量）
        safety_factor: 安全系数，建议 2.0 以覆盖更多误差来源
    
    Returns:
        建议的相对误差阈值
    """
    # 精度类型对应的相对精度
    precision_map = {
        "FP16": 1 / 1024,   # mantissa 10 bit
        "BF16": 1 / 128,    # mantissa 7 bit
    }
    
    dtype_precision = precision_map.get(dtype)
    if dtype_precision is None:
        raise ValueError(f"Unsupported dtype: {dtype}, only FP16/BF16 supported")
    
    # 单次累加误差估计
    single_accumulation_error = 0.5 * dtype_precision
    
    # N 次累加后的累积误差（线性估计）
    accumulated_error = world_size * single_accumulation_error
    
    # 应用安全系数
    threshold = accumulated_error * safety_factor
    
    return threshold


# 示例计算
print("FP16 累加阈值示例: ")
for ws in [2, 4, 8, 16]:
    threshold = calculate_accumulation_threshold("FP16", ws)
    print(f"  world_size={ws}: {threshold:.6f}")

print("\nBF16 累加阈值示例: ")
for ws in [2, 4, 8, 16]:
    threshold = calculate_accumulation_threshold("BF16", ws)
    print(f"  world_size={ws}: {threshold:.6f}")
```

**输出示例**：

```.txt
FP16 累加阈值示例：
  world_size=2: 0.001953
  world_size=4: 0.003906
  world_size=8: 0.007812
  world_size=16: 0.015625

BF16 累加阈值示例：
  world_size=2: 0.015625
  world_size=4: 0.031250
  world_size=8: 0.062500
  world_size=16: 0.125000
```

**说明**：

- 以上为线性估计，实际误差可能因数据分布、数值范围等因素有所不同
- **FP16 的动态范围有限**：最大值为 65504，累加时需注意溢出问题
- **大数吃小数问题**：若数据数值差异较大（如大值和小值混合累加），误差可能显著高于理论值
- 对于精度要求较高的场景，建议使用 FP32 进行累加以获得更高精度（参见 Golden 与算子计算逻辑不一致）

**检查要点**：

- Atomic Add 累加操作需根据累加精度和累加次数调整阈值：
  - FP32 累加：阈值应 ≤ 5e-3（千分之五）
  - FP16 累加：阈值按累加次数计算，world_size=8 时约为 0.0078（约千分之八）
  - BF16 累加：阈值按累加次数计算，world_size=8 时约为 0.0625（约千分之六十三）
- 建议根据实际数据类型和累加方式调整阈值，避免阈值过严导致误报

#### Device 日志检查

- [ ] plog 日志无 error 报错，排除编译问题
- [ ] Device 日志无 error 报错，排除功能问题

**检查目的**：排除编译问题和功能异常，确保是纯精度问题。

**操作步骤**：

```bash
# 1. 检查 plog 日志（用于检查编译阶段的错误）
grep -i "error" $ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-*.log

# 2. 检查 Device 日志（日志路径由 ASCEND_PROCESS_LOG_PATH 环境变量指定）
grep -i "error" $ASCEND_PROCESS_LOG_PATH/debug/device*/device*.log
```

> **说明**：`ASCEND_PROCESS_LOG_PATH` 表示日志的落盘路径，设置方法请参考 [环境变量配置说明](../../trouble_shooting/README.md)。

**检查要点**：

- plog 日志中无 error 报错表明编译阶段正常
- Device 日志中无 error 报错表明算子功能执行正常
- 若 plog 中存在 Pass 报错，需先定位并修复编译问题
- 若存在 AICore error、AICPU error 等错误，需先定位并修复功能问题
- 编译问题与功能问题应优先解决，再进行精度定位

#### 通信算子标准测试用例检查

- [ ] 通信算子标准测试用例能正常执行且精度无异常

**检查目的**：排除环境配置问题，确认通信基础功能正常。

**操作步骤**：

```bash
# 进入测试用例目录（从 pypto 工程根目录）
cd models/experimental/distributed

# 执行通信算子测试用例
python3 test_matmul_allreduce_addrmsnorm.py
```

**检查要点**：

- Python 测试用例框架内部已集成多进程启动机制，无需配置 mpirun
- 若标准用例执行失败或精度异常，说明环境配置可能有问题
- 若标准用例正常但当前算子异常，问题可能在于算子实现本身

上述三项检查通过后，可确认问题为纯精度问题，进入后续定位阶段。

### 检查精度问题的复现情况

通过多次执行观察精度异常的表现形式，初步判断问题类型：

#### 情况 1：稳定位置精度异常

**特征**：多次执行后（固定随机数种子），精度异常的位置和误差值稳定一致。

**问题推断**：切分或数据处理逻辑问题，而非同步问题。

**定位方向**：参考典型错误场景排查：

- **切块语义不匹配**：检查 shmem_signal 与 shmem_put、shmem_get 与 shmem_wait_until 的 TileShape 是否匹配
- **Golden 计算逻辑不一致**：检查累加精度是否一致
- **数据类型处理**：检查输入输出数据类型转换是否正确

#### 情况 2：随机位置精度异常

**特征**：多次执行后，精度异常的位置和误差值不一致，甚至部分执行正常、部分执行异常。

**问题推断**：同步问题或数据竞争，可能涉及：

- **依赖关系配置错误**：pred 参数传递错误，导致执行顺序不符合语义
- **缺少屏障同步**：`shmem_clear_data` 后未执行 `shmem_barrier_all`，导致数据竞争
- **信号量语义不匹配**：写入和等待的数据范围不一致

**定位方向**：重点检查第三阶段中的依赖关系和同步机制：

- 验证 `pred` 参数是否正确传递
- 验证 `shmem_barrier_all` 是否在首轮通信前执行
- 验证切分逻辑是否正确，可参考 [通信算子切块设置指南](distributed_operation_tiling_guide.md)

### 减小问题规模

缩小问题规模旨在简化场景，提高定位效率。缩小后可查看 Pass 图是否符合预期，分析各算子的连接关系、数据流、切分方式。需确保缩小后仍能复现原始问题。

通常通过以下方法缩小问题规模：

**方法 1：减小 world_size**

将 world_size 从大值减小到最小可复现值（如 world_size = 2）。若 world_size = 2 正常但更大值异常，问题可能与累加次数相关（参见 Golden 与算子计算逻辑不一致）。

**方法 2：减少切块数量或不切块**

增大 TileShape 减少切块数量，或直接使用不切分的配置。不切分时问题消失，说明切分逻辑有问题（参见切块语义不匹配）；不切分时问题仍存在，说明问题与切分无关。

**方法 3：减小 Shape 规格**

调小模型的 Shape 规格（如 batch_size、hidden_size），降低计算和通信规模，便于直接对计算结果进行分析。

**方法 4：减小维度**

减少 tensor 的维度数量（如从 3D 降到 2D），简化数据结构和计算逻辑。维度减少后问题消失，说明问题可能与特定维度的处理相关；维度减少后问题仍存在，说明问题与维度数量无关。

## 通信算子维测方法

当无法简单对问题进行定界时，可以借助以下维测方法做进一步的定位。

### 通过 shmem_get 读取 Win 区数据

**目的**：验证 Win 区数据是否符合预期，确定问题发生的位置。

- 若 Win 区数据正确 → 问题出现在 `shmem_get` 之后
- 若 Win 区数据异常 → 问题出现在 Win 区数据写入阶段

**适用场景**：

- 验证 `shmem_put` 写入或累加结果是否正确
- 检查是否存在数据覆盖或残留问题

**操作步骤**：

1. **将 shmem_get 的输出加入算子输出列表**：

    ```python
    # 正常的通信流程完成后，读取 Win 区数据
    win_data = pypto.distributed.shmem_get(
        shmem_tensor, my_pe, shmem_shape, [0, 0],
        pred=[wait_until_out], valid_shape=shmem_shape)

    # 将 win_data 加入算子输出列表，用于后续精度对比
    return [output, win_data]
    ```

2. **对比 Win 区数据与预期值**：在测试脚本中获取输出并与 golden 对比

    ```python
    # 运行算子获取输出
    outputs = run_operator(...)
    output, win_data = outputs

    # 对比 Win 区数据与预期值
    expected_win_data = golden_result
    max_diff = torch.abs(win_data - expected_win_data).max().item()
    print(f"Win 区数据最大误差: {max_diff}")
    ```

    **检查要点**：

    - Win 区数据是否与 golden 计算结果一致
    - 多轮通信场景下，首轮数据是否正确清理
    - 累加精度是否符合预期（BF16/FP16 累加误差较大）

    **注意事项**：

    - `shmem_get` 必须在 `shmem_wait_until` 之后执行，确保数据写入完成
    - 将 shmem_get 输出加入算子返回列表

### 利用 aicore_print 打印 TileOp 信息

通过在 CodeGen 生成的 CCE 文件中添加打印语句，可以获取 TileOp 运行时的 addr、shape、offset 等关键信息。结合 Pass 图分析这些信息是否符合预期，例如确定数据实际拷贝的大小和位置是否正确，帮助定位切分或偏移问题。

**适用场景**：

- 验证动态 shape 的实际运行时值
- 检查 TileOp 的数据偏移是否正确
- 定位越界访问或地址错误问题

**前置配置**：

1. **配置固定 CCE 输出路径**：修改 `framework/src/interface/configs/tile_fwk_config.json`

    ```json
    "codegen": {
        "fixed_output_path": true,
        "force_overwrite": false,
        "parallel_compile": 1
    }
    ```

2. **启用打印开关**：确保 `framework/src/interface/machine/device/tilefwk/aicore_print.h` 中

    ```c
    #define ENABLE_AICORE_PRINT 1
    ```

3. **开启 device 日志并指定落盘路径**：设置方法请参考 [环境变量配置说明](../../trouble_shooting/README.md)。

    **操作步骤**：

    1. **生成 CCE 文件**：首次运行测试用例，生成 CCE 文件到 `./kernel_aicore/` 目录

    2. **添加打印语句**：在目标 TileOp 执行后添加打印，打印需要验证的信息，例如打印 addr、shape 等信息

    ```cpp
    GMTileTensorFP32DIM2_1 gmTensor_1((__gm__float*)(RUNTIME_COA_GET_PARAM_ADDR_MAYBE_CONST(2, 0, 0, 1), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 1)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 1)))))
    // 打印gmTensor_1的 addr
    AicoreLogF(param->ctx, "gmTensor_1 addr: %lu", RUNTIME_COA_GET_PARAM_ADDR_MAYBE_CONST(2, 0, 0, 1));

    // 打印gmTensor_1的 shape
    AicoreLogF(param->ctx, "shape_0: %lu, shape_1: %lu", Std::get<0>(gmTensor_1.GetShape()), Std::get<1>(gmTensor_1.GetShape()));
    ```

4. **查看打印结果**：打印数据位于指定的device日志落盘路径

5. **分析打印结果**：结合 Pass 图分析 TileOp 实际运行时的入参信息（addr、shape、offset 等）是否符合预期，例如数据拷贝的大小和位置是否正确。

### 定位出现问题的OP

定位首个出现精度问题的OP，详细方法参考 [精度调试指南](../debug/precision.md) 中的缩小问题规模和工具自检和分析章节。

## 典型错误场景

### 切块语义不匹配

#### shmem_signal 与 shmem_put 切块语义不匹配

**问题现象**：特定位置数据错误，部分元素精度异常。

**原因分析**：`shmem_signal` 写入的信号量标识的数据块大小与 `shmem_put` 实际写入的数据块大小不匹配。

**错误示例**：

```python
shmem_shape = [64, 64]
pypto.set_vec_tile_shapes(16, 64)
put_dummy = pypto.distributed.shmem_put(input_tensor, [0, 0], shmem_tensor, 0,
              put_op=pypto.AtomicType.ADD, pred=[input_tensor])
pypto.set_vec_tile_shapes(33, 64)
pypto.distributed.shmem_signal(shmem_tensor, 0, 1, shmem_shape,
              [0, 0], target_pe=0, sig_op=pypto.AtomicType.ADD, pred=[put_dummy])
```

**问题分析**：

- `shmem_put` TileShape 为 [16, 64]，数据被切分为 4 块
- `shmem_signal` TileShape 为 [33, 64]，信号量被切分为 2 块
- `tile_shmem_signal0` 对应 `tile_shmem_put0` 和 `tile_shmem_put1`
- `tile_shmem_signal0` 标识写入 [33, 64] 数据，实际只写入 [32, 64] 数据
- **信号量语义不匹配，可能导致精度异常**

**正确配置**：

```python
shmem_shape = [64, 64]
pypto.set_vec_tile_shapes(16, 64)
put_dummy = pypto.distributed.shmem_put(input_tensor, [0, 0], shmem_tensor, 0,
              put_op=pypto.AtomicType.ADD, pred=[input_tensor])

pypto.set_vec_tile_shapes(32, 64)
pypto.distributed.shmem_signal(shmem_tensor, 0, 1, shmem_shape,
              [0, 0], target_pe=0, sig_op=pypto.AtomicType.ADD, pred=[put_dummy])
```

**验证逻辑**：

- `shmem_put` 将数据切分为 4 块：每块 [16, 64]
- `shmem_signal` 将信号量切分为 2 块：每块对应 [32, 64]
- 第 1 个 signal 块覆盖前 2 个 put 块（共 [32, 64]）
- **信号量语义正确匹配**

#### shmem_get 与 shmem_wait_until 切块语义不匹配

**问题现象**：读取数据区域超出实际写入范围，导致精度异常。

**原因分析**：`shmem_get` 读取的数据块大小与 `shmem_wait_until` 等待的信号量对应的数据块大小不匹配。

**错误示例**：

```python
shmem_shape = [64, 64]
pypto.set_vec_tile_shapes(16, 64)
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0], cmp=pypto.OpType.EQ)

pypto.set_vec_tile_shapes(33, 64)
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out])
```

**问题分析**：

- `shmem_wait_until` TileShape 为 [16, 64]，等待 4 个信号量块
- `shmem_get` TileShape 为 [33, 64]，读取 2 个数据块
- `tile_shmem_get0` 对应等待前 2 个 wait_until 块
- 前两个 wait_until 块对应的数据区域为 [32, 64]
- **但 `tile_shmem_get0` 试图读取 [33, 64] 数据，超出实际写入范围**

**正确配置**：

```python
shmem_shape = [64, 64]
pypto.set_vec_tile_shapes(16, 64)
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0], cmp=pypto.OpType.EQ)

pypto.set_vec_tile_shapes(32, 64)
output = pypto.distributed.shmem_get(
shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out], valid_shape=shmem_shape)

### 多轮通信场景需先进行 shmem_clear_data

**问题现象**：多轮通信场景下，首轮数据影响后续轮次，精度异常。

**原因分析**：多轮通信需在每轮前执行 `shmem_clear_data` 清理共享内存，且 `shmem_clear_data` 之后必须执行 `shmem_barrier_all` 进行屏障同步，确保所有 rank 清理完成后再开始通信。

**正确配置**：

```python
# 每轮通信前：先清理，再屏障同步
data_clear_out = pypto.distributed.shmem_clear_data(shmem_tensor, shmem_shape, [0, 0])
signal_clear_out = pypto.distributed.shmem_clear_signal(shmem_tensor)

# `shmem_clear_data` 和 `shmem_clear_signal` 之后必须执行 `shmem_barrier_all`，确保所有 rank 清理完成
barrier_out = pypto.distributed.shmem_barrier_all(
    shmem_barrier_signal, [data_clear_out, signal_clear_out])

# 后续通信操作需依赖 barrier_out
put_out = pypto.distributed.shmem_put(..., pred=[barrier_out])
```

**关键说明**：

- **shmem_clear_data 后必须执行 shmem_barrier_all**：若 `shmem_clear_data` 后直接开始通信，部分 rank 可能还在清理中，其他 rank 已开始写入，导致数据竞争或首轮数据残留
- **每轮通信前执行 shmem_clear_data**：多轮场景需在每轮前执行 `shmem_clear_data`，避免首轮数据影响后续轮次

### 依赖关系配置错误

**问题现象**：精度异常，且每次执行问题表现可能不一样（随机位置异常）。

**原因分析**：依赖关系配置错误或算子语义理解偏差，导致执行顺序或等待条件不符合预期。

#### 常见错误类型

**错误 1：pred 参数传递错误**

`pred` 参数传递错误，导致通信算子执行顺序不符合语义要求，产生数据竞争：

```python
# 错误示例：shmem_get 未依赖 wait_until_out，可能在数据写入完成前读取
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    pred=[barrier_out])  # 错误：未依赖 wait_until_out

# 正确示例：shmem_get 必须依赖 wait_until_out
wait_until_out = pypto.distributed.shmem_wait_until(...)
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    pred=[wait_until_out])  # 正确：依赖 wait_until_out
```

**错误 2：wait_until 等待条件不正确**

`shmem_wait_until` 等待条件设置错误，未等待所有 rank 完成写入，导致读取结果不完整：

```python
# 问题示例（AllReduce）：等待值设置为 1，仅等待一个 rank 完成
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, 1, shmem_shape, [0, 0],  # 错误：等待值应为 world_size
    cmp=pypto.OpType.EQ, 
    clear_signal=True,
    pred=[barrier_out])

# 结果：仅等待一个 rank 完成，其他 rank 数据未写入，读取结果不完整
```

**等待条件根据算子语义确定**，确保与实际累加次数匹配：

```python
# 正确示例（AllReduce）：等待所有 rank 发送完成，等待值为 world_size
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0],
    cmp=pypto.OpType.EQ,
    clear_signal=True,
    pred=[barrier_out])
```

#### 标准通信流程依赖关系

```.txt
shmem_clear_data → shmem_clear_signal → shmem_barrier_all → 
    shmem_put → shmem_signal → shmem_wait_until → shmem_get
```

**关键原则**：

1. **初始化阶段**：`shmem_clear_data` 和 `shmem_barrier_all` 必须在通信前执行，确保初始状态正确
2. **写入阶段**：shmem_signal 必须依赖 shmem_put，确保数据写入完成后再通知
3. **读取阶段**：shmem_get 必须依赖 shmem_wait_until，确保所有 rank 写入完成后再读取
4. **等待条件**：wait_until 的等待条件应与算子语义匹配

#### 正确配置示例

```python
# 1. 清理共享内存和信号量
data_clear_out = pypto.distributed.shmem_clear_data(
    shmem_tensor, shmem_shape, [0, 0], pred=[input_tensor])
signal_clear_out = pypto.distributed.shmem_clear_signal(
    shmem_tensor, pred=[input_tensor])

# 2. 屏障同步，确保所有 rank 完成清理
barrier_out = pypto.distributed.shmem_barrier_all(
    shmem_barrier_signal, [data_clear_out, signal_clear_out])

# 3. 数据写入（依赖 barrier）
put_out = pypto.distributed.shmem_put(
    input_tensor, [0, 0], shmem_tensor, dyn_idx,
    put_op=pypto.AtomicType.ADD, pred=[barrier_out])

# 4. 信号量通知（依赖 put）
pypto.distributed.shmem_signal(
    shmem_tensor, dyn_idx, 1, shmem_shape, [0, 0],
    target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])

# 5. 等待所有 rank 写入完成（依赖 barrier）
wait_until_out = pypto.distributed.shmem_wait_until(
    shmem_tensor, my_pe, world_size, shmem_shape, [0, 0],
    cmp=pypto.OpType.EQ, clear_signal=True, pred=[barrier_out])

# 6. 数据读取（依赖 wait_until）
output = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    pred=[wait_until_out], valid_shape=shmem_shape)
```

### Golden 与算子计算逻辑不一致

**问题现象**：精度误差存在但误差不大，计算结果与 golden 存在可预期的精度差异。

**原因分析**：Golden 的计算逻辑（如累加精度、运算顺序等）与算子实现不一致，导致计算结果存在合理的精度差异。这是一种常见的精度问题类型，下面以累加精度差异为例进行说明。

#### 案例：累加精度差异（AllReduce）

**Golden 实现**（CPU 上计算）：

```python
# Golden: 在 CPU 上对 BF16 数据进行累加
# 注意：torch 在 CPU 上执行 BF16 浮点运算时，会自动转换为 FP32 做加法，最后将结果转为bf16
def allreduce_golden(inputs):
    result = torch.zeros(shape, dtype=torch.bfloat16)
    for tensor in inputs:  # inputs 为 BF16 类型
        result += tensor  # 实际按 FP32 累加
    return result
```

**算子实现**（NPU 上计算）：

```python
# 算子: shmem_tensor dtype 为 BF16，累加在 BF16 精度下进行
shmem_tensor = pypto.distributed.create_shmem_tensor(
    group_name, world_size, pypto.DT_BF16, shmem_shape)  # BF16 累加

# shmem_put 使用 AtomicType.ADD，累加精度由 shmem_tensor dtype 决定
put_out = pypto.distributed.shmem_put(
    input_tensor, [0, 0], shmem_tensor, target_pe,
    put_op=pypto.AtomicType.ADD, pred=[...])
```

**问题分析**：

| 对比项 | Golden | 算子实现 |
|-------|--------|---------|
| 累加精度 | FP32（torch CPU 自动转换） | BF16 |
| 精度损失 | 小 | 大（BF16 mantissa 仅 7 bit） |
| 结果差异 | - | 累加次数越多，误差越大 |

**解决方法**：确保 Golden 与算子实现使用相同的累加精度：

```python
# 算子使用 FP32 累加（与 Golden 一致）
shmem_tensor = pypto.distributed.create_shmem_tensor(
    group_name, world_size, pypto.DT_FP32, shmem_shape)  # FP32 累加
```

**关键检查点**：

- Golden 的累加精度（CPU 上 torch 对 BF16 的处理方式）
- 算子中 `shmem_tensor` 的 dtype（决定累加精度）
- 确保两者一致，避免计算模式差异导致误差

## 回归与验证

修复后需验证精度并回归测试，确保问题彻底解决。

**验证要点**：

- 与 golden 数据对比，确认精度达标
- 验证多卡一致性
- 验证可复现性（多次执行结果一致）

**回归测试范围**：

- 原问题场景验证
- 不同 shape 规格测试
- 不同 world_size 测试
- 多轮执行稳定性测试
