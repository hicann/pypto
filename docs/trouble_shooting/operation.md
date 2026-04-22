# OPERATION 组件错误码

（待补充）

- **范围**：FCXXXX（含 VECTOR/MATMUL/CONV/视图类OP 子范围）
- 本文档说明 OPERATION 组件的错误码定义、场景说明与排查建议。
- 补充错误码时，可注明 **关联 Skill**（链接至 [.agents/skills](../../.agents/skills) 下对应技能，便于排查时加载）。

## 错误码定义

相关错误码的枚举与码值统一定义在 `framework/include/tilefwk/error_code.h`（OPERATION 大类见 VectorErrorCode、MatmulErrorCode、Conv* 等；子范围说明见下文链接）。

## 子范围拆分

- **FC0XXX - FC2XXX**：VECTOR，见 [vector.md](vector.md)
- **FC3XXX - FC5XXX**：MATMUL，见 [matmul.md](matmul.md)
- **FC6XXX - FC8XXX**：CONV，见 [conv.md](conv.md)
- **FC9XXX**：视图类OP，见 [view_op.md](view_op.md)
