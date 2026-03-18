---
name: pypto-operator-develop-workflow
description: PyPTO 算子开发工作流程。用于开发华为昇腾 AI 处理器自定义算子。在接到算子开发任务时使用，确保开发过程规范、高效、符合官方最佳实践。
tag: [PyPTO，算子开发]
---

# PyPTO 算子开发工作流程

本技能提供 PyPTO 算子开发的完整工作流程指导。

## 工作流程概览

```
需求检查 → 环境准备 → Plan 模式 → 开发实现 → 测试验证 → 高阶参数使能
```

## 核心原则

1. **遇问题先探索，不简化代码**
   - 第一步：使用 `Explore Agent` 搜索查找 API 资料，选择合适的 pypto operation
   - 第二步：综合审视代码，查阅官方示例
   - 第三步：定位问题点后修复
   - 禁止：下意识简化代码、凭直觉实现、遇到错误就推翻重写

2. **充分了解后再决策**
   - 查阅资料、搜索代码、理解原理
   - 不要轻易下结论

3. **持续探索更优方案**
   - 方案走通后，继续搜索是否有更优实现

4. **环境兼容性验证**
   - 确认 API/方法适用于 A3 服务器，CANN 8.5.0

## 常见问题与解决方案

### 最常见错误 TOP 6

1. **BFloat16 转 NumPy 失败**：必须先 `.float()` 再 `.numpy()`
2. **环境变量未设置**：先运行 `npu-smi info | grep "No running processes found"` 确认空闲卡，再设置 `export TILE_FWK_DEVICE_ID=<空闲卡号>`
3. **动态轴定义位置错误**：必须在 jit 函数外部定义
4. **Tile Shape 未设置**：matmul 前必须调用 `set_cube_tile_shapes`
5. **精度标准不合理**：bfloat16 使用 atol=0.0001, rtol=0.0078125
6. **使用 PyTorch 作为 Golden 函数**：使用 NumPy 实现 golden 函数时，bfloat16 数据类型转换不够准确

### 注意点
1. **动态数据范围使用 valid_shape**：最后一块数据量可能会小于固定块大小时，view函数中一定要指定valid_shape
2. **动态循环边界使用 unroll_list**：当循环次数是动态的，需要使用 unroll_list，多层循环嵌套时，最内层使用 `unroll_list`。
3. **pypto tensor初始化**：torch用zeros函数定义变量，pypto不需要，创建tensor即可，后续会有写入整个覆盖这个tensor
4. **op准确度**：禁止无中生有op，确保使用的op是pypto支持的
5. **装饰器jit的使用**：选择最新的不用wrapper的pypto.frontend.jit前端写法，参考`docs/api/pypto-frontend-jit.md`

## 开发阶段

### 阶段一：需求检查

必需信息清单：
- 算子名称
- 数学公式
- 输入/输出规格（shape、dtype）
- 支持的数据类型
- 精度要求
- 服务器类型

### 阶段二：环境准备
1. 检查cann包是否安装
```bash
# 检查 环境变量是否设置
echo ${PATH} |grep cann-8.5.0 # 检查cann-8.5.0是否配置
```
2. 检查pto-isa源码是否获取
```bash
# 检查 环境变量是否设置
echo ${PTO_TILE_LIB_CODE_PATH} # 检查依赖的pto-isa源码是否获取
```
**⚠️ 重要提示**：
- 首先，需要检查这个路径（pto-isa源码路径）是否存在；
- 如果不存在，参考`scripts/environment_prepare.sh`脚本进行环境初始化；
- 如果不存在，初始化也不成功，可以参考`docs/install/prepare_environment.md`获取pto-isa源码章节，并进行环境变量的设置。

3. 验证关键api文档及用例目录
```bash
ls docs/api/    # API 接口说明
ls examples/            # 示例代码
```
4. 设置device_id
```bash
# 查找空闲卡
npu-smi info | grep "No running processes found"

# 根据空闲卡设置环境变量（假设卡1空闲）
export TILE_FWK_DEVICE_ID=1
export PTO_TILE_LIB_CODE_PATH=./pto_isa/pto-isa/
```

-**⚠️ 重要提示**：
-- 未设置TILE_FWK_DEVICE_ID会导致运行时提示："If no NPU environment is available"

如果以上信息未检查通过或者有疑问，请参考`docs/install`中的参考资料进行环境准备。

### 阶段三：Plan 模式

**进入条件**（满足任一即进入）：
- 非最简单算子（与现有示例几乎相同除外）
- 需要组合多个 API
- 用户明确要求制定计划

**跳过条件**：
- 与现有示例几乎完全相同
- 只需单个 API
- 用户明确表示不需要

### 阶段四：开发实现

开发顺序：
1. 创建算子目录结构
   - 创建custom/my_operator/
2. 编写 golden 及测试用例
   - 创建my_operator.py
   - 测试用例涉及随机数生成时，设置随机数种子
3. 编写 operator 实现代码
   - 创建my_operator_impl.py
   - 使用@pypto.frontend.jit的方式实现

**⚠️ 重要提示**：
- 实现代码与测试代码分开

### 阶段五：测试验证
优先使用npu模式进行验证
验证步骤：
1. 没有安装pypto的话，编译whl包并安装，有的话，则不需要反复编译
   python3 build_ci.py -f python3 --disable_auto_execute
2. 执行算子进行验证
   **检查到存在npu卡的时候，直接使用run_mode=npu执行**
   python3 custom/my_operator/my_operator.py
3. 验证如果失败，不要跳过问题或者简化算子，正向排查问题

测试类型：
- 单元测试：基本功能验证
- 精度测试：精度符合要求
- 性能测试：性能符合要求

**⚠️ 重要提示**：
- 有npu卡的情况下，不要使用run_mode=sim
- 为了节省时间，编译一次whl包并安装即可
- **完成阶段五后，必须继续执行阶段六：高阶参数使能**

### 阶段六：高阶参数使能 ⭐
当阶段五保证算子基础版本正确后，请做以下高阶参数的使能
1. 涉及loop的话，请使能loop_unroll
2. 设置stich相关参数,比如stitch_function_inner_memory，stitch_function_outcast_memory，stitch_function_num_initial等参数

## 环境兼容性

**当前环境**：A3 服务器，CANN 8.5.0

查阅资料时必须确认 API/方法适用于当前环境。

## 注意事项
1. 当编译或执行时长超过10分钟时，如果是卡住了，请中断并杀掉相关进程，重新检查代码。
2. 优先使用npu模式进行精度验证
