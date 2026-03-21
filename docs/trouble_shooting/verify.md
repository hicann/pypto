# VERIFY 组件错误码


- **范围**：FBXXXX
- 本文档说明 VERIFY 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义

相关错误码的统一定义，参见 ` framework/src/interface/interpreter/verify_error.h`与`framework/src/interface/interpreter/calculator/calc_error.h`文件。

## 排查建议

### 典型错误码场景
该场景涉及的错误码属于精度工具组件功能的一部分，出现这些错误码，精度工具仍然在正常发挥作用，用户可结合一下排查建议自行开展精度问题定位：

#### 错误码：0xB4001U：VERIFY_RESULT_MISMATCH
##### 日志示例
该场景为精度校验失败场景，同时会伴随如下输出：
```log
[ERROR] PYPTO(1535746):2026-03-20 09:43:12.887 [flow_verifier.cpp:105][VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED
[INFO ] PYPTO(1535746):2026-03-20 09:43:12.887 [flow_verifier.h:151][VERIFY]:  
    Error rtol=0.001000 atol=0.001000 index:0 golden:10 output:1 absDiff:9 relDiff:1.63636  
    Error rtol=0.001000 atol=0.001000 index:1 golden:10 output:1 absDiff:9 relDiff:1.63636  
    Error rtol=0.001000 atol=0.001000 index:2 golden:10 output:1 absDiff:9 relDiff:1.63636  
    Error rtol=0.001000 atol=0.001000 index:3 golden:10 output:1 absDiff:9 relDiff:1.63636  
    Error rtol=0.001000 atol=0.001000 index:4 golden:10 output:1 absDiff:9 relDiff:1.63636  
    Error rtol=0.001000 atol=0.001000 index:5 golden:10 output:1 absDiff:9 relDiff:1.63636
  All size:256 failNum:256 maxAbsDiff:9 maxRelDiff:1.63636 averageAbsDiff:9 averageRelDiff:1.63636 errorCount:256 errorRatio:1 zeroCount:0 zeroRatio:0
  maxAbs-> index:0 golden:10 output:1 absDiff:9 relDiff:1.63636
  maxRel-> index:0 golden:10 output:1 absDiff:9 relDiff:1.63636
```
首先根据类似示例中`[VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED`确定精度出错的阶段。如示例中，出错阶段为`tensor_graph `，即进入pass前精度出错。
##### 定位手段指导

###### pass_verify_print与pass_verify_save  
对于在`tensor graph`阶段的精度问题，进行人工数据分析。  
精度工具提供`pypto.pass_veirfy_print`与`pypto.pass_verify_save`支持用户将自己编写的pypto kernal函数中的tensor的计算结果打印或者保存下来。（注意：该tensor既可以是最终的输出，也可以是中间产生的tensor，但是打印出来的结果并不是在npu中的计算结果，而是基于精度工具模拟执行和用户前端表达的模拟结果）。  
再开启精度定位前，可先确认精度工具Dump的最终输出与npu计算结果保持一致。
详参见[pass_verify_print接口示例](docs/api/others/pypto-pass_verify_print.md)与
[pass_verify_save接口示例](docs/api/others/pypto-pass_verify_save.md)  。

###### 精度工具skill
对于在`tensor graph`阶段的精度问题，借助ai agent进行精度定位。  
当发现算子精度不对但不知道具体问题在哪，可以调用pypto-binary-search-verify技能，只需要向助手发送明确的指令“算子test_my_op.py 精度验证失败了，请帮我使用算子精度问题查找技能定位是哪里有问题”，或者直接指定“使用pypto-binary-search-verify技能，定位test_my_op.py 的精度问题”, 就可以自动插入检查点，根据测试生成数据文件用对比脚本分析结果，得出出错的op。
###### 精度工具自动比对脚本
对于发生在pass执行阶段的精度问题，使用自动化脚本进行定位。
脚本路径：`tools/verifier/pass_compare.py`  
当某个pass精度对比失败的时候，可以利用 `pass_compare.py` 这个脚本将该对比失败的pass和前面的pass进行精度对比。对比会在精度工具dump数据的目录生成一个类似 `verify_pass@SplitK@ExpandFunction@1773821696834386.csv` 这样的对比结果文件，里面记录了精度对比失败的pass的每个op节点和前面pass对比的结果，未能匹配上的也会记录在表中标注skip。这样就能定位到匹配上的第一个出错的节点。  
脚本使用方法：`python3 pass_compare.py --p ExpandFunction RemoveUndrivenView --verify_path=.....`
`--p`参数后面的是对比的两个pass，空格隔开，前面的是精度对比失败的pass，后面的是作为golden的pass，`--verify_path`参数是精度工具dump数据文件的那个目录的绝对路径。
#### 错误码：0xB200FU：RUNTIME_EXCEPTION 
##### 日志示例
该场景为operation模拟执行失败，往往是operation上有不正确的属性，日志示例如下：
```log
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.112 [operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB (magic=10564) input[0] tensorMagic=94, shape=[82816, 512], offset=[0, 0], dynValidShape=[82816, 512], dynOffset=[]
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.112 [operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB (magic=10564) input[1] tensorMagic=3717, shape=[1, 16], offset=[0, 1792], dynValidShape=[RUNTIME_GetViewValidShapeDim(RUNTIME_GetInputShapeDim(ARG_topk_indices,0),((bIdx*((RUNTIME_GetInputShapeDim(ARG_query_nope,0)/RUNTIME_GetInputShapeDim(ARG_kv_act_seqs,0))/128))+s1Idx),1), RUNTIME_GetViewValidShapeDim(2048,((s2_idx*2048)+1792),16)], dynOffset=[((bIdx*((RUNTIME_GetInputShapeDim(ARG_query_nope,0)/RUNTIME_GetInputShapeDim(ARG_kv_act_seqs,0))/128))+s1Idx), ((s2_idx*2048)+1792)]
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.112 [operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB (magic=10564) input[2] tensorMagic=12209, shape=[1, 512], offset=[0, 0], dynValidShape=[RUNTIME_GetViewValidShapeDim(RUNTIME_GetInputShapeDim(ARG_block_table,0),bIdx,1), 512], dynOffset=[bIdx, 0]
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.112 [operation.cpp:58][VERIFY]:ErrCode: FB200F! ExecuteOperation error: op GATHER_IN_UB (magic=10564) output[0] tensorMagic=3716, shape=[16, 512], offset=[0, 0], dynValidShape=[sym_3716_dim_0, sym_3716_dim_1], dynOffset=[]
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.116 [flow_verifier.cpp:299][VERIFY]:ErrCode: FB200F! VerifyPass failed for function TENSOR_LOOP_L4_s2_SA_LoopUnroll1_Unroll1_PATH0_hiddenfunc0_20, pass InferParamIndex (passIndex: 28, captureIndex: 4): 10447510811372117949, 3415564916459561148, 10564, GATHER_IN_UBOpError
func: TENSOR_LOOP_L4_s2_SA_LoopUnroll1_Unroll1_PATH0_hiddenfunc0_leaf22
filename: /data/l00504208/g00955608/pypto/models/deepseek_v32_exp/sparse_flash_attention_quant_impl.py
lineno: 136
/* /data/l00504208/g00955608/pypto/models/deepseek_v32_exp/sparse_flash_attention_quant_impl.py:136 */
<16 x 512 x DT_INT8 / sym_3716_dim_0 x sym_3716_dim_1 x DT_INT8> %3716@3264#(22)MEM_UB::MEM_UB = !10564 TILE_GATHER_IN_UB(g:22, s:-1) %94@106#(1)MEM_DEVICE_DDR::MEM_DEVICE_DDR, %3717@100(0, 1792)(((bIdx*((RUNTIME_GetInputShapeDim(ARG_query_nope,0)/RUNTIME_GetInputShapeDim(ARG_kv_act_seqs,0))/128))+s1Idx), ((s2_idx*2048)+1792))#(22)MEM_DEVICE_DDR::MEM_D
[ERROR] PYPTO(1693310):2026-03-20 15:08:54.116 [flow_verifier.cpp:299][VERIFY]:EVICE_DDR, %12209@102(bIdx, 0)#(22)MEM_DEVICE_DDR::MEM_DEVICE_DDR #IS_CUBE{0} #block_size{128}
<16x512xINT8/0x512xINT8> = GATHER_IN_UB <82816x512xINT8/82816x512xINT8>, <1x16xINT32/1x16xINT32>, <1x512xINT32/1x512xINT32>
out must have shape [topk_count, hidden_dim]
```
#### 定位指导
核心报错为：  
`<16x512xINT8/0x512xINT8> = GATHER_IN_UB <82816x512xINT8/82816x512xINT8>, <1x16xINT32/1x16xINT32>, <1x512xINT32/1x512xINT32> `
  `out must have shape [topk_count, hidden_dim]`  
  第一行为出错`operation`的简要信息， 示例中，`<16x512xINT8/0x512xINT8> `为输出的`<shape/validshape>`,等号右边为这个operation的`opcode`及其所有输入的`<shape/validshape>`。  
  示例中，`torch cpp`抛出错误信息`out must have shape [topk_count, hidden_dim]`，同时结合上一行信息，可以初步判断该报错来源为该`operation`的输出的`validshape`为空shape。  
  如果需要进一步的信息定位，可参考上方几行更详细的报错，包括该`operation`输入输出`tensor`的信息，以及该`operation`的`IR`。

### 非典型错误码场景
#### 错误码： 0xB0001U: VERIFY_NOT_ENABLE
请检查本地`torch >= 2.1.0`。
#### 其他错误码
对于其他类型的错误码，往往是由于pypto内部缺陷导致，如遇到，请在社区联系开发人员解决。
