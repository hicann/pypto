---
name: pass-ut-generate
description: 根据Pass业务描述，生成单元测试用例（UT）。当用户输入业务情况时，能根据业务，生成对应Pass的Ut用例。
license: 完整条款见 LICENSE.txt
---

# Pass 业务单元测试生成

## 概述

本技能用于分析如何根据用户描述生成对应的Pass侧单元测试用例（UT），结合pypto-pass-module-analyzer/SKILL.md技能分析Pass业务，帮助设计相关单元测试用例（UT）。

## 触发机制
 	 
 	当用户输入包含以下关键字或者相关内容时，自动触发此技能：
 	 
 	 （1）设计Pass模块XXX的UT用例**：设计指定 Pass 模块的功能的UT用例
 	 （2）设计Pass模块XXX的XXX功能**：设计指定 Pass 模块的指定功能的UT用例
 	 （3）设计XXX功能的相关Pass模块的UT用例**：设计与XXX功能相关Pass的UT用例
 
 	触发示例：

 	 （1）"设计Pass模块AutoCast的UT用例"
 	 （2）"设计Pass模块AutoCast的对于不支持BF16 OP插入Cast的功能"
 	 （3）"设计删除冗余Op功能的相关Pass的UT用例"

## 使用场景
 	 
 	当需要设计 PyPTO pass 中模块的功能或业务单元测试时使用此技能。

## UT生成流程一

### 步骤 1：分析业务

    根据用户描述的业务情况，分析相关业务：
        （1）当描述为具体Pass的具体业务时，根据@.opencode/skills/pypto-pass/pypto-pass-module-analyzer/SKILL.md，分析业务场景，根据当前业务设计相应的UT用例，例如：设计Splitk这个Pass消除RedunceAcc功能的UT；
        （2）当描述为模糊Pass业务时，根据@.opencode/skills/pypto-pass/pypto-pass-module-analyzer/SKILL.md，进行相关Pass业务总结，挑选出符合业务的Pass，分析业务场景，设计对应的UT用例，例如：请针对Pass中对于视图类Op，view、assemble处理的Pass设计对应的UT，验证功能实现；
        （3）当描述为设计Pass的UT时，根据@.opencode/skills/pypto-pass/pypto-pass-module-analyzer/SKILL.md，进行相关Pass业务总结，分析业务场景，设计该Pass业务的UT用例，例如：针对Splitk这个Pass设计相关UT用例。

### 步骤 2：环境配置

    在pypto/framework/tests/ut/passes/src/test_xxx.cpp寻找对应的测试文件，观察当前文件中，是否初始化环境，若已完成初始化，则跳过该步骤。其中xxx一般为Pass名称，
    当未找到对应的测试文件时，可以查找test_xxx.cpp中创建类的名字是否与所要求设计Pass名字是否一致。
    初始化环境详细步骤：
        （1）创建新的类，命名xxx.cpp，文件名为test_xxx.cpp，其中xxx为输入的pass名称
        （2）声明该类继承于gtest框架
        （3）该类中，编写相关函数：
                1.所有测试用例全局初始化函数--static void SetUpTestCase() {}，若未明确指定内容，则为空实现；
                2.所有测试用例全局清理函数--static void TearDownTestCase() {}， 若未明确指定内容，则为空实现；
                3.每个测试用例执行前的初始化测试环境函数--void SetUp() override {}，若未指定内容，则默认生成以下代码体；
```cpp
                    void SetUp() override {
                        Program::GetInstance().Reset();
                        config::Reset();
                        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH); // 其中 pypto/framework/src/interface/configs/config_manager_ng.h中的COMPILE_STAGE策略，通过pass所在文件夹目录，得到所处的编译策略
                        config::SetHostConfig(KEY_STRATEGY, "XXXTestStrategy"); // 其中xxx为pass名字，表示host侧KEY_STRATEGY
                        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false); // 其中 pypto/framework/src/interface/configs/config_manager.h中的Platform KEYs表示平台策略，需要根据用户传入的进行修改，若未传入，则采用该默认方式
                        TileShape::Current().SetVecTile({64, 64}); // 设置vector的tile块大小
                        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64}); // 设置cube的tile块大小
                    }
```
                4.每个测试用例执行后的清理测试环境函数--void TearDown() override {}，若为明确指定内容，则为空实现。

### 步骤 3：搭建测试用例框架

    搭建测试用例框架 TEST_F(XXXX, XXX){ }，其中XXXX为上述新建的测试类，XXX为该测试用例名字，可根据根据业务内容生成。
    另外，在TEST_F(XXXX, XXX){ }上方位置处可以添加该测试用例注释，描述经过该pass前后的变化，例如：
 ```cpp
            /*
        TESTRemoveDummyExpand
        inCast{8,16}->expand->ubTensor{8,16}->exp->outCast1{8,16}
                                            ->sqrt->outCast2{8,16}
                                            ->reciprocal->outCast3{8,16}
        inCast{8,16}->exp->outCast1
                    ->sqrt->outCast2
                    ->reciprocal->outCast3
        */
        TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest1) {
            ...
        }
```

### 步骤 4：构建function

    构建整张计算图function，利用智能指针进行创建，并在创建后判断是否为空。若用户未明确指定参数，则belongTo为新建Program实例，funcMagicName和funcRawName均为TestXXX，XXX为pass名字，parentFunc为空指针。
    
    详细的信息如下：
        Function类常用构造函数：
```cpp
        Function(const Program &belongTo, const std::string &funcMagicName, const std::string &funcRawName,
            Function *parentFunc);
```
        Program常用获取实例：
```cpp
        Program &Program::GetInstance() {
            static Program sProgram;
            return sProgram;
        }
```

        详细代码可以参考：pypto/framework/src/interface/program/program.cpp和pypto/framework/src/interface/function/function.h

### 步骤 5：创建Tensor

    构建计算图中的Tensor，利用智能指针，根据业务需求，创建所需要的Tensor。

    LogicalTensor类常用构造函数：
        LogicalTensor(Function &function, DataType t, Shape tshape, TileOpFormat tformat = TileOpFormat::TILEOP_ND, std::string tname = "",
        NodeType tnodetype = NodeType::LOCAL);
    
    详细LogicalTensor类信息，请参考：pypto/framework/src/interface/tensor/logical_tensor.h

### 步骤 6：创建Operation及绑定function输入输出

    构建计算图中的Operation，利用智能指针，根据业务需求，创建所需要的Operation。

    常用的创建Operation及绑定function函数：
```cpp
            Operation &Program::AddOperation(const Opcode opCode,
        const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
        const std::vector<std::shared_ptr<LogicalTensor>> &oOperand) {
        // Add the operation to the current function
        if (currentFunctionMagicName_ == PROGRAM_ENTRY_FUNCTION_NAME) {
            FUNCTION_LOGE("Error: No active function to add operation.");
            ASSERT(false) << "No active function to add operation.";
        }
        return currentFunctionPtr_->AddOperation(opCode, iOperand, oOperand);
    }
```

    详细创建Operation及绑定function函数信息，请参考：pypto/framework/src/interface/program/program.cpp

    Opcode信息，请参考：pypto/framework/src/interface/operation/opcode.h和pypto/framework/src/interface/operation/opcode.cpp

### 步骤 7：对业务功能进行校验

    根据业务功能，校验pass运行后的处理结果是否符合预期。

    例如校验function经过pass后，assemble数量是否符合预期，可以通过遍历function，查找assemble的数量进行比对，具体代码如下：
```cpp
        uint32_t assemble_num = kNumZero;
        for (auto &op : currFunctionPtr->Operations()) {
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                ++assemble_num;
            }
        }
        EXPECT_EQ(assemble_num, kNumZero);
```

### 步骤 8：利用现有的环境，执行生成的UT，并对错误进行修改(若当前环境正常，需在当前环境中验证)

    当生成UT用例后，执行Python3 build_ci.py -c -u=xxx.* -j=24来验证用例是否正确，xxx为该测试用例类的名字，对于错误进行改正。
    当执行超时时，优先执行Python3 build_ci.py -u=xxx.* -j=24。
    若此时还存在执行超时问题，则执行Python3 build_ci.py -u=xxx.xx -j=24，xx为测试用例
    例如: python3 build_ci.py -c -u=TestRemoveRedundantOpPass.* -j=24 -f=cpp
          python3 build_ci.py -u=TestRemoveRedundantOpPass.* -j=24 -f=cpp
          python3 build_ci.py -c -u=TestRemoveRedundantOpPass.TestIntermediateOutcast -j=24 -f=cpp

### 步骤 9：统计UT覆盖率

    在当前环境中，通过GCov来统计代码UT覆盖率，主要包括行覆盖率和方法覆盖率。
    使用方法：
        python3 build_ci.py -c -u=xxx.* --gcov -j=24 -f=cpp
    xxx为该测试用例类的名字,xx为测试用例
    执行结束时,会在build/路径下，生成cov_result 目录，打开index.html观察对应pass的UT覆盖率情况，若覆盖率<=80%，则针对未覆盖的业务，重复以上步骤，进行该Pass UT补充。
    
## UT生成流程二

### 步骤 1：分析业务

    同UT生成流程一中步骤1：分析业务

### 步骤 2：环境配置

    同UT生成流程一中步骤2：环境配置

### 步骤 3：搭建测试用例框架

    同UT生成流程一中步骤3：搭建测试用例框架

### 步骤 4：构建function

    利用ComputationalGraphBuilder类来构建function，通过调用AddTensor()和AddTensors()来实现function中Tensor的构建，调用AddOp()和AddOps()来实现function中Op的构建。
    通过调用SetInCast()和SetOutCast()来实现对function的输入输出构建。

    ComputationalGraphBuilder类信息，请参考：pypto/framework/tests/ut/passes/src/computational_graph_builder.h
    Opcode信息，请参考：pypto/framework/src/interface/operation/opcode.h和pypto/framework/src/interface/operation/opcode.cpp
    Operation信息，请参考: pypto/framework/src/interface/operation/operation.cpp
    详细LogicalTensor类信息，请参考: pypto/framework/src/interface/tensor/logical_tensor.h
    Tensor创建过程中DataType信息，请参考：pypto/framework/include/tilefwk/data_type.h

### 步骤 5：对业务功能进行校验

    同UT生成流程一中步骤7：对业务功能进行校验

### 步骤 6：利用现有的环境，执行生成的UT，并对错误进行修改(若当前环境正常，需在当前环境中验证)

    同UT生成流程一中步骤8：利用现有的环境，执行生成的UT，并对错误进行修改(若当前环境正常，需在当前环境中验证)

### 步骤 7：统计UT覆盖率

    同UT生成流程一中步骤9：统计UT覆盖率

## 注意事项

    要符合常见的CPP代码编程规范，例如常见的编程错误：未使用的变量定义、未修改的引用加入const、魔鬼数字等；
    生成的用例请真实执行，并对生成的用例进行验证修改；
    对于UT覆盖率，请打印出来当前所设计UT的覆盖率。

## 参考资料

| 类别 | 文件路径 |
|------|----------|
| 生成流程一示例 | `pypto/framework/tests/ut/passes/src/test_removeredundantop.cpp` |
| 生成流程二示例 | `pypto/framework/tests/ut/passes/src/test_cube_process.cpp` |
| ComputationalGraphBuilder | `pypto/framework/tests/ut/passes/src/computational_graph_builder.h` |
| function信息 | `pypto/framework/src/interface/program/program.cpp和pypto/framework/src/interface/function/function.h`|
| Opcode 定义 | `pypto/framework/src/interface/operation/opcode.h` |
| op属性定义 | `framework/src/interface/operation/attribute.h` |
| DataType 定义 | `pypto/framework/include/tilefwk/data_type.h` |
| LogicalTensor 定义 | `pypto/framework/src/interface/tensor/logical_tensor.h` |
| Operation信息 | `pypto/framework/src/interface/operation/operation.cpp`  |
| COMPILE_STAGE策略 | `pypto/framework/src/interface/configs/config_manager_ng.h` |

---

## 编译阶段参考值

| 阶段 | 配置值 |
|------|--------|
| TENSOR GRAPH 执行 | `CS_TENSOR_GRAPH` |
| TILE GRAPH 执行 | `CS_TILE_GRAPH` |
| BLOCK GRAPH 执行 | `CS_EXECUTE_GRAPH` |

根据 Pass 所在文件夹目录选择对应的编译策略。


