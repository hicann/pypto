# dtype 扩展变更清单模板

> 复制此清单用于跟踪每次 dtype 扩展的完整变更。

## 基本信息

- **operation**: {op_name}
- **目标 dtype**: {dtype_list}
- **目标架构**: {a2a3 / a5 / both}
- **日期**: {date}

## 阶段 1：pypto 已支持检查

- [ ] 运行 `check_dtype_support.py` 脚本
- [ ] 确认目标 dtype 未被完全支持（否则任务结束）
- [ ] 记录当前已支持的 dtype 列表

结果：
```
{paste_script_output}
```

## 阶段 2：pto-isa 依赖检查

- [ ] 运行 `check_pto_isa_support.py` 脚本
- [ ] 确认 pto-isa 已支持目标 dtype
- [ ] 如果不支持，终止并报告

结果：
```
{paste_script_output}
```

## 阶段 3：dtype 配置 JSON 修改

- [ ] 确定 operation 对应的 opcode 字符串（在 `.cpp` 中查找 `GetOpSupportedInputDtypes(Opcode::OP_XXX)`）
- [ ] 确定目标架构对应的 JSON 文件（a2a3/a5/kirin9030/kirinx90）
- [ ] 定位 JSON 文件: `framework/src/interface/configs/platform_op_supported_dtypes/{arch}_supported_op_dtypes.json`
- [ ] 在 `ops.{OPCODE}.input_dtypes` 数组中追加目标 dtype（使用 `STR_DATA_TYPE_MAP` 小写友好名，如 `int64`、`float32`）
- [ ] 如果 operation 有多个分发 opcode（如 `ADD`/`ADDS`），每个 opcode 都要修改
- [ ] 无独立 opcode 的复合 operation（如 `Clip`）：确认是否需修改 `.cpp` 内联集合
- [ ] 确认 JSON 语法正确

修改详情：
```
JSON 文件: {json_path}
opcode: {OPCODE}
新增 dtype: {dtype_list}
```

## 阶段 4a：API 文档更新

- [ ] 定位文档文件: `docs/zh/api/tensor_api/operation/pypto-{op}.md`
- [ ] 找到「约束说明」章节中的 dtype 列表
- [ ] 更新 dtype 列表（架构区分型或统一型）
- [ ] 确认 `<!-- npu -->` 标签格式正确
- [ ] 确认 id 编号未被破坏

修改详情：
```
文件: {doc_path}
修改的章节: 约束说明 > Tensor数据类型说明
修改内容: {description}
```

## 阶段 4b：测试用例编写

> 只需编辑 CSV 文件。JSON 由测试脚本自动生成，不要手动编辑。

### CSV 测试用例
- [ ] 定位 CSV 文件: `framework/tests/st/operation/test_case/{Op}_st_test_cases.csv`
- [ ] 确认当前最大 case 编号
- [ ] 为每个新增 dtype 追加测试行
- [ ] 确认数据范围合理
- [ ] 确认 case_name 编号接续现有最大编号

新增测试行：
```
{paste_new_csv_lines}
```

### JSON 测试用例
- [ ] 确认未手动编辑 JSON 文件（JSON 由 `run_operation_test_with_config.py` 从 CSV 自动生成）

## 阶段 5：编译与测试

> 不要直接运行 gtest 二进制。必须通过 `run_operation_test_with_config.py` 脚本执行。

- [ ] 通过 `run_operation_test_with_config.py` 运行新增测试用例
- [ ] 确认 Golden 数据生成成功（日志中出现 `Generate golden success`）
- [ ] 确认所有新增测试用例通过
- [ ] 如架构不支持（如 A2 板子运行仅 A5 支持的 dtype），记录为预期失败

测试结果：
```
编译: PASS / FAIL
Golden 生成: PASS / FAIL
C++ ST 测试: PASS / FAIL ({passed}/{total} 用例通过)
```

## 变更文件汇总

| # | 文件 | 变更类型 | 说明 |
|---|------|---------|------|
| 1 | framework/src/interface/configs/platform_op_supported_dtypes/{arch}_supported_op_dtypes.json | 修改 | {description} |
| 2 | docs/zh/api/tensor_api/operation/pypto-{op}.md | 修改 | {description} |
| 3 | framework/tests/st/operation/test_case/{Op}_st_test_cases.csv | 修改 | 新增 N 条测试用例 |

> 注意：`{Op}_st_test_cases.json` 不在变更清单中——它由测试脚本从 CSV 自动生成。

## 已知问题

- {如有未验证项或环境限制，在此列出}
