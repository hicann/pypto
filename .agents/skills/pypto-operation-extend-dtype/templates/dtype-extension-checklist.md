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

## 阶段 3：C++ 源码修改

- [ ] 定位源码文件: `framework/src/interface/operation/vector/{file}.cpp`
- [ ] 找到 `{OP}_A2A3_TYPES` 定义
- [ ] 找到 `{OP}_A5_TYPES` 定义
- [ ] 修改 A2A3 集合（如需要）
- [ ] 修改 A5 集合（如需要）
- [ ] 检查所有函数重载是否都已修改
- [ ] 确认无编译错误

修改详情：
```
文件: {file_path}
修改的变量: {var_names}
新增的 DT_* 值: {dt_values}
修改的重载函数: {overload_list}
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
| 1 | framework/src/interface/operation/vector/{file}.cpp | 修改 | {description} |
| 2 | docs/zh/api/tensor_api/operation/pypto-{op}.md | 修改 | {description} |
| 3 | framework/tests/st/operation/test_case/{Op}_st_test_cases.csv | 修改 | 新增 N 条测试用例 |

> 注意：`{Op}_st_test_cases.json` 不在变更清单中——它由测试脚本从 CSV 自动生成。

## 已知问题

- {如有未验证项或环境限制，在此列出}
