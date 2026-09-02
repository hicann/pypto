---
name: pypto-tensor-ir-dev
description: "PyPTO Tensor IR full-stack development guide."
---

# PyPTO Tensor IR Full-Stack Development Guide

## 1. Layered Architecture and Code Map

```
User Python kernel
    |
    v  python/pypto/pil/parser.py        <- L1 parser layer: Python AST -> PIL (Block/Call/Value/Jump)
    v  python/pypto/pil/dispatcher.py    <- L2 dispatch layer: walk PIL blocks, trace-execute
    v  python/pypto/pil/ops.py           <- L2 ops layer: @impl registry (semantic impls, emit IR)
    v  python/pypto/pil/pir.py           <- L3 build layer: BuildContext(ir.IRBuilder)/Scope/InsertPoint
    v  python/src/bindings/ir|tensor.cpp <- L3 bindings: pybind11 (ir nodes / builder / pass)
    v  framework/include/ir + src/interface/ir   <- L4 core IR: immutable syntax tree + transforms + passes + verifiers
    v  framework/src/interface/tensor/   <- L4 tensor layer: LogicalTensor/Operation/IRContext (npu::tile_fwk)
```

Dependencies point strictly downward. The **core IR (`pypto::ir`) never depends on the tensor layer**; all tensor semantics live in `npu::tile_fwk`. Write generic transforms against the core IR only; include tensor headers only when LogicalTensor semantics (shape/token/slot) are involved.

## 2. L1 Parser Layer (python/pypto/pil/parser.py)

**Responsibility**: `ast2pil(pyfunc)` compiles Python function source into PIL data structures (pure data, no IR). It fetches source via `inspect.getsourcelines`, runs `ast.parse`, then visits nodes one by one.

### PIL data model (defined in pir.py)

| Structure | Fields | Notes |
|---|---|---|
| `Value(id)` | frozen dataclass | PIL temporary (`%N`); **resolved by id through Scope.varmap** |
| `Call(result, callee, args, kwargs, span)` | one call | result=None means void call; Starred/DoubleStarred wrap `*x`/`**d` |
| `Block(id, args, calls, jump, store_names, span)` | basic block | args are block params (loop vars); jump is the single exit |
| `Jump` enum | `END_BRANCH/RETURN/CONTINUE/BREAK` | END_BRANCH = normal end of an if branch |
| `Function(name, span, signature, body, global_vars, params, param_defaults)` | | nested def/lambda compile to Function (entry_point=False); closure values pass through global_vars |

### Key mechanisms

- **Control-flow structuring** (parser.py):
  - `visit_if`: opens one `ctx.new_block()` per branch, ends with `set_jump(Jump.END_BRANCH)`, emits `ctx.call("pil.if_else", (cond, then_block, else_block))` — **if blocks are passed as operands to pil.if_else**.
  - `visit_for`/`visit_while`: loop body is `new_block(args=(loop_var,))`; a body without a jump gets `set_jump(Jump.CONTINUE)`; emits `ctx.call_void("pil.loop", (body, iter))`. while is built from `pil.if_else` + BREAK composition (parser.py:373-391).
  - `pypto.loop(...)` -> LoopKind.DYNAMIC_FOR (compiled to a hardware for; break/continue/return unsupported).
- **store_names tracking**: `ctx.store(name, val)` emits a `pil.store` call and records the name in `current_block.store_names`; `mark_store` records without emitting; on block exit `old.store_names.update(new_block.store_names)` bubbles up. **store_names is the data source for yield/carry decisions** in the L2 layer. Attribute-slot writes (`d.a = x`) register as `"d.a"` via `slot_write` in pir.py.
- **span fixup**: `Source._fix_line_numbers` corrects line offsets from `inspect` source extraction; every node carries an `ir.Span` so errors point at user source lines.
- Unsupported syntax raises via `ctx.raise_error(node, msg)` (while-else/for-else, return inside dynamic for, ...).

## 3. L2 Dispatch & Ops Layer (dispatcher.py + op_registry.py + ops.py)

### op_registry.py: minimal registry

```python
@impl(stub)                 # stub is any callable or a "pil.xxx" string
def xxx_impl(ctx, *args):   # register the impl; with partial=True the signature is (ctx, stub, *args)

dispatch(stub, ctx, *args)  # table lookup + call; unregistered -> call stub itself (Python semantics fallback)
```

- String stubs (`"pil.if_else"`, `"pil.loop"`, `"pil.store"`, ...) are internal control-flow/store calls emitted by the parser and **must** have an impl.
- All other stubs are real objects (`operator.add`, `pypto.Tensor`, `pypto.cond`, `pypto.matmul`, ...): without an impl they execute directly with Python semantics (compile-time evaluation path).

### dispatcher.py: trace engine

- `dispatch_block(block, is_static, rewriter)`: runs `dispatch_call` for each Call in the block.
  - With `is_static=True` (static interpretation mode) the block tail goes through `block_jump`: CONTINUE/BREAK/RETURN raise the matching Signal exception; END_BRANCH returns the block result.
  - **store_names bubbling**: on block exit, merge this block's store_names into the parent (under CollectContext); rewriter handles name rewriting for nested function calls (frame_rewrite maps callee param names to the caller argument's canonical name).
- `dispatch_call(call, scope, ctx)`:
  1. `scope.resolve` resolves callee/args (Value -> varmap lookup; Starred unpacks).
  2. If the callee is a PIL `Function` (user helper) or a compilable plain function, call `call_function` (fresh Scope, param binding, static dispatch); otherwise `dispatch(callee, ...)`.
  3. Write the result into `scope.varmap[call.result.id]`.
  4. `pypto_impl.SetSpan(...)` + `ctx.emit_tensor_stmts()`: **flush C++-side accumulated tensor ops into IR statements** (once per user call).
  5. Exception wrapping: exceptions that are not control-flow Signals become `DispatchError` with a source-location stack from `ctx.call_stack`.
- `call_function`: `pil.store` writes to params inside a nested function are written back after return to caller-scope variables bound to the same object (dispatcher.py:159-168).
- `_get_or_compile` (lru_cache): user helpers lazily compile to PIL Functions; functions from pypto/stdlib modules execute directly without compiling.
- **Control-flow Signals** (pir.py): `ReturnSignal/BreakSignal/ContinueSignal` propagate across layers; impls catch them and convert to the matching IR statement (`_add_jump_stmt`).

### ops.py: core impl semantics

- **Scope/value domain** (pir.py):
  - `Scope(locals, parent, varmap)`: `varmap[Value.id] = actual value` (Python value or SymbolicScalar/Tensor); `locals` binds names; dotted names (`"d.a"`) resolve reflectively through `_resolve_slot`.
  - `ctx.wrap(ir.Expr)` / `ctx.unwrap(val)`: IR <-> Python value conversion. unwrap: bool/int/float -> constants; SymbolicScalar -> as_expr; Tensor -> logical_tensor(); list/tuple -> MakeTuple. wrap: ScalarType -> SymbolicScalar; LogicalTensorType -> `Tensor.from_logical_tensor`; NoneType -> None; UnknownType -> `_Poison` (any use raises "Conflicted var" — the deferred type-conflict mechanism).
  - `Journal`: undo log for attribute-slot writes; `pil.if_else` rolls back `d.a = x` side effects while tracing branches.
- **Key impls** (ops.py):
  - `loop_impl` -> `_dyn_for` (LoopRange: unroll_list expands into multiple ForStmts; factor>1 multiplies step and rewrites nstop) -> `_loop_unroll`: for every carriable name create `create_var_like` + `create_iter_arg(var, initValue)`; inside the body use `ctx.change_return_vars(names)` so BREAK/CONTINUE statements carry those values; at loop end build returnVars and store them back into scope. ForStmt attrs carry `#loop_conds`/`_config_scope`/`unroll_times`.
  - `if_else_impl`: a concrete SymbolicScalar/Python bool condition traces only one branch statically; otherwise `_if_else_stmt`: trace each branch under its own `BeginScope` + `InsertPoint(branch_body)` + `Journal()` isolation -> `carriable_names` (store_names minus opaque values, phi names first) forms the yield set -> `_merge_yield_vars` merges per name (type_equal builds var_like; type conflict builds poison; **when the two branches' valid_shape differs, build phi scalars and add them to yield/returnVars**) -> emit trailing YieldStmt per branch -> build IfStmt. RETURN/BREAK inside a branch goes through `_add_jump_stmt` (with yield values).
  - `carriable_names`: `sort_names` puts phi first + alphabetical; `is_opaque_value` (not None/bool/int/float/SymbolicScalar/Tensor/list/tuple) excludes.
  - `_add_jump_stmt(ctx, jump, operands)`: BREAK/CONTINUE -> create_break/continue_stmt (operands resolved from `ctx.return_var_names`); RETURN -> return_stmt; END_BRANCH -> yield_stmt.
  - Tensor construction/ops (`create_tensor`, view/matmul/...): `ctx.create_tensor_op_stmt(...)` + `ctx.emit(stmt)`; `pypto.assemble` and friends go through `_patch_methods`/`_store_wrapper` for slot_write_inplace tracking.

### pil2ir.py / compile_pipeline.py (top-level drivers)

- `compile(pyfunc, *args)`: `ast2pil` -> (when has_move) a `collect_store_names` pre-scan (CollectContext, collects store_names only, builds no IR) -> the real `pil2ir`: bind args into scope, dispatch the whole function body, end with create_return_stmt.
- `compile_new_ir` (compile_pipeline.py): pil.compile -> create_program -> the default pass pipeline (see 7.3).

## 4. L3 Builder, Base IR, and Bindings

### Core IR nodes (framework/include/ir/)

| File | Contents |
|---|---|
| `core.h` | `ObjectKind` enum (all node kinds, RTTI-free type checks), `IRNode` base |
| `expr.h` | `Expr`/`Var`/`MemRef`/`Call`/`MakeTuple`/`GetItemExpr`/`ConstInt/Float/Bool`/`IterArg{iterVar_, initValue_}`, all binary (Add..BitShiftRight, incl. Min/Max/Eq..Ge) and unary (Abs/Neg/Not/BitNot/Cast) expressions |
| `stmt.h` | `AssignStmt`, `IfStmt`, `YieldStmt`, `ReturnStmt`, `ForStmt`, `WhileStmt`, `SeqStmts`, `BreakStmt`, `ContinueStmt`, `TensorOpStmt` |
| `type.h` | `Type`/`ScalarType`/`TokenType`/`NoneType`/`UnknownType`/`ShapedType`/`LogicalTensorType`/`TupleType`/`MemRefType` |
| `function.h`/`program.h` | `Function(params, body, returnTypes)`; `Program(functions)` indexed by name |
| `reflection/field_traits.h` | `UsualField`/`DefField`/`IgnoreField` field descriptors |

**Immutable + `shared_ptr<const T>`**: `using XxxPtr = std::shared_ptr<const Xxx>`; transformation = COW rebuild (unchanged subtrees return the original pointer). Idiomatic casts: `ir::As<T>(node)` (const view), `AsMut`/`const_pointer_cast` (mutable view, your responsibility), `npu::tile_fwk::AsLogicalTensor(expr)` (tensor layer).

**Reflection descriptors**: every node defines `static constexpr auto GetFieldDescriptors()`. It is the **shared metadata for visitor/mutator/structural_equal/structural_hash/printer/bindings** — a new field or node without a descriptor is silently skipped by traversal, comparison, and Python binding. DEF fields (result/returnVars/iterArgs) count as definitions in SSA analysis.

**Control-flow SSA conventions** (understand before writing a pass):
- IfStmt: `condition_` + `thenBody_` + `elseBody_` (**`std::optional<SeqStmtsPtr>`**) + `returnVars_`; each branch body ends with a YieldStmt whose value count equals the returnVars count (checked by verify_ssa).
- ForStmt/WhileStmt: `iterArgs_` (loop-carried) + `body_` (ends with Yield/Continue carrying next-iteration values) + `returnVars_`; all three counts match.
- `SeqStmts::Make` flattens nesting/collapses single statements; **control-flow bodies stay SeqStmts** (`SeqStmts::Wrap`) — do not use Make when rebuilding branch bodies in a mutator.
- ForStmt attrs (`GetAttr<T>(key)`): `parallel`, `_loop_conds`, `_config_scope`, `unroll_times`, ...

### ir::IRBuilder (framework/include/ir/builder.h, stack-context API)

```
BeginFunction/FuncArg/ReturnType/EndFunction
BeginForLoop + AddIterArg + AddReturnVar ... EndForLoop
BeginWhileLoop + AddWhileIterArg/AddWhileReturnVar/SetWhileLoopCondition ... EndWhileLoop
BeginIf + (BeginElse) + AddIfReturnVar ... EndIf
BeginSection/EndSection; Emit(stmt)
```

A context stack drives it — **Emit attaches to the top of the stack**; EndXxx validates iterArgs/returnVars counts and pops the built statement.

### Tensor-layer builder (framework/src/interface/tensor/irbuilder.h)

- `IRContext` (**global singleton** `IRContext::Get()`): unique variable naming (`GetVarName`: duplicates get a `_N` suffix, origin names propagate to clones), type table, token table, tensor data desc list. `Reset()` runs before every compile.
- Tensor `IRBuilder : ir::IRBuilder` (npu::tile_fwk) additionally provides:
  - `CreateRawTensor` / `CreateTensorVar` (static shape / dynamic validShape / view of a raw tensor, optionally attached to a Function)
  - `CreateScalarVar(sym)` -> SymbolicScalar; `CreateVarLike(name, value)` (builds a new var typed like value: LogicalTensor copies shape/dynvalidshape, scalar -> fresh symbol, None, tuple recurses)
  - `CreateTensorOpStmt` (two overloads: the Operation version with a Function / the IR-only version result/result_token/opcode/args/tokens/attrs)
  - One-shot Create* for everything (CreateIfStmt/CreateForStmt/CreateYieldStmt/... — no context stack, direct construction)
  - `Checkpoint()/Restore()` (frontend branch isolation), `EmitTensorStmts()` (flush C++-side accumulated ops into the current insert point)

### PIL-side wrappers (python/pypto/pil/pir.py)

- `BuildContext(ir.IRBuilder)`: `span` (change_span context), `return_var_names` (change_return_vars: values carried by BREAK/CONTINUE inside loop bodies), `loop_stack`, `call_stack` (DispatchError stack), `phi_id`. `BuildContext.current()`/`parent` chain is maintained by `__enter__/__exit__` (threading-local `_current`).
- `CollectContext(BuildContext)`: used by the collect pre-scan; dispatch_block uses it to bubble store_names to the parent without building IR.
- `InsertPoint(body)`: inside the `with` block `ctx.set_insert_point(ir.InsertPoint(body))` routes emits to the given SeqStmts — **that is how branch/loop bodies get filled**.

### Python bindings (python/src/bindings/ir/ + tensor.cpp)

| File | Binds |
|---|---|
| `ir/ir.cpp` | DataType/Span/Type/IRNode/Expr/Var/statement classes. **Fields bind automatically through GetFieldDescriptors** (`BindFields`); non-reflective APIs (`__str__`, properties, type_equal) are hand-written `.def` |
| `ir/builder.cpp` | All IRBuilder create_*/begin_* methods, InsertPoint, `type_equal` (structural type equality + TypeEqual for LogicalTensor) |
| `ir/pass.cpp` | Pass (all `.def_static` pass factories), IRProperty(Set), IRVerifier |
| `tensor.cpp` | Tensor/LogicalTensor (`.def_property("valid_shape", GetDynValidShape, UpdateDynValidShape)` etc.) |

Adding a binding takes three steps: C++ `.def` -> update the `python/pypto/pypto_impl/__init__.pyi` stub in the same change -> rebuild (section 8) and verify the new attribute actually loads (the .so double-tree trap).

## 5. Tensor Layer (framework/src/interface/tensor/)

| File | Responsibility |
|---|---|
| `logical_tensor.h/.cpp` | **LogicalTensor : ir::Var**. `tensor`(RawTensor) + `offset` + `shape` + `dynValidShape_` (SymbolicScalar list) + `dynOffset_` + `storage_` + read/write token + producers/consumers. `Clone()`/`NextVersion()`/`View()`/`DumpSSA()`/`GetDynValidShape()`/`UpdateDynValidShape()`. Also hosts the generic cast `AsLogicalTensor(expr)` and `TypeEqual` |
| `raw_tensor.h` | RawTensor physical allocation; multiple LTs view the same RawTensor |
| `symbolic_scalar.h` | SymbolicScalar: `FromExpr`/`SubstituteVars`/`Simplify`/`Check`+SatStatus(SAT/UNSAT)/`GetVarRefs` — the carrier of shape expressions |
| `ir.h/.cpp` | Tensor-op IR utilities: `CollectScalarVarRefs` (**includes result_ dynvalidshape refs**), `IsSameRawTensor`, `GetVarMemoryId` (alias-aware liveness), `CollectTensorOpAttrs` |
| `ir_tensor_op_rebuild.h` | `RebuildTensorOpStmt(src, results, resultTokens, args, tokens, span, targetFunc=nullptr)`: rebuild an op after substitution; when src is an Operation it goes through CloneTensorOpStmt and **preserves pass metadata**, otherwise builds a plain TensorOpStmt |
| `ir_func_builder.h/.cpp` | RootFunctionBuilder: IR -> Function/Program assembly; slot linking (LinkIfStmtSlots: returnVar<->yield value same slot; LinkForStmtSlots: init->iterVar->continue value->returnVar chain) |
| `tensor_slot.h/.cpp` | TensorSlotManager::SetSameSlot (storage slot association) |

`Operation : ir::TensorOpStmt` (adds a Function reference + pass metadata). Core-layer TensorOpStmt fields: `result_` (DEF), `result_token_`, `opcode_` ("VIEW"/"ADDS"/"ASSEMBLE"/"MATMUL"/"TENSOR_ALLOC"...), `args_`, `tokens_` (memory-order dependencies), `attrs_`.

**Key semantics**:
- shape (allocated shape) vs dynValidShape (per-dim valid size, may contain symbolic expressions like `min(s - i*128, 128)`); `view(..., valid_shape=[...])` emits a VIEW op whose result LT carries the dynValidShape.
- **A scalar reference inside dynvalidshape is a "use"**: any new code that collects uses must account for it (see `CollectScalarVarRefs`, terminator value collection, and canonicalize carry-liveness — all three already handle it). Missing it means canonicalize prunes a scalar carry while an exported type still names it.
- Tokens (`GetReadToken/GetWriteToken`): memory-hazard serialization (TokenPass/InferTokenPass).

## 6. IRVisitor / IRMutator (transforms/base/ + visitor.cpp/mutator.cpp)

### IRVisitor (read-only)

- Subclasses do `using IRVisitor::VisitExpr_; using IRVisitor::VisitStmt_;` then override `VisitExpr_(const XxxPtr&)` / `VisitStmt_(const XxxPtr&)`. **Default implementations recurse into children**; to stop recursion, do not call the base implementation.
- `VisitVarLike_`: one method for all Var subclasses (LogicalTensor is a Var too).
- `VisitBinaryExpr_`/`VisitUnaryExpr_`: catch all binary/unary expressions at once.

Ready-made utilities (transforms/utils/stmt_utils.h):
- `CollectVarUses(expr)` / `CollectStmtVarRefs(stmt(s), skip_iter_updates)` — Var reference collection (skip_iter_updates skips Yield/Break/Continue carried values)
- `CollectDefinedVars(stmt)` — DEF collection (DefVarCollector)
- `VarExprMap = unordered_map<VarPtr, ExprPtr>` (**key is the old Var pointer**), `LookupVarInExpr`, `VarSubstitutor`, `SubstituteVars(stmt, varMap)`

### IRMutator (COW rebuild)

Rules:
- When overriding, visit children first, **compare pointer-by-pointer** to detect change, and return the original `op` when nothing changed.
- **The else branch uses `std::optional<StmtPtr>`**; rebuild through an explicit split: `has_value() ? make_shared<IfStmt>(..., *elseBody, ...) : make_shared<IfStmt>(..., std::nullopt, ...)`. Never pass an empty StmtPtr to the optional parameter — the optional engages a null body, `SeqStmts::Wrap` produces a SeqStmts holding a null statement, and downstream dereferences crash (historical bug).
- Helpers: `VisitExprList`/`VisitVarList`/`VisitIterArgList` (return `{new vector, changed}`); `var_remap_` (keeps references consistent when a definition pointer changed).
- Deep-substitution reference implementation (`SunkStmtSubstitutor` in merge_stmts_pass.cpp): override `VisitExpr_(VarPtr)` with LookupVarInExpr; for TensorOpStmt substitute result/args/attrs and **clone result while substituting its dynvalidshape**; save/restore the varmap around branch bodies so clones do not leak into sibling branches (the SeqStmts VisitStmt_ saves/restores the varmap).

## 7. Pass Development

### Structure (include/ir/transforms/passes.h)

```cpp
// Simple pass: lambda factories
auto my_pass = ir::pass::CreateFunctionPass([](const ir::FunctionPtr& f) { ... }, "my_pass");
auto my_pass = ir::pass::CreateProgramPass([](const ir::ProgramPtr& p) { ... }, "my_pass");
// Complex pass: inherit PassImpl (operator()/GetName/Get{Required,Produced,Invalidated}Properties)
// PassPipeline: AddPass + Run(program); property verification is driven by PassContext
```

### Built-in pass quick reference

| Pass | Location | Purpose |
|---|---|---|
| `Canonicalize()` | transforms/utils/canonicalize.cpp | Prune unused iterArgs/returnVars/yield values (carry-liveness fixed point) |
| `AggressiveDCE()` | transforms/utils/dead_code_elimination.cpp | Dead code elimination (keeps transitive closure of side-effect ops/returns) |
| `InferTokenPass()` | infer_token_pass.cpp | Add memory-hazard token edges (WAW/WAR) |
| `RemoveRedundantTokenPass()` | remove_redundant_token_pass.cpp | Remove redundant token edges |
| `MergeStmtsIntoIf()` | merge_stmts_pass.cpp | Sink if-successors into branches, SAT-prune dead branches |
| `CreateRootFunctions()` | ir_func_builder.cpp | Program-level lowering to `tile_fwk` functions: dynFunc (entry, keeps control flow) + per tensor-op segment a hiddenFunc/pathFunc pair, chained `dynFunc ->OP_CALL-> pathFunc ->OP_CALL-> hiddenFunc`; wires incast/outcast slots. Stateful (mutates global `tile_fwk::Program`), not pure IR->IR |
| printer/io_text_dumper/io_text_loader | Text serialization; grammar in `transforms/io_text.md` |

### New-pass checklist

1. Implement `framework/src/interface/ir/transforms/my_pass.cpp` (PassImpl in an anonymous namespace; export `ir::Pass pass::MyPass()`).
2. Declare it in `include/ir/transforms/passes.h` under `namespace pass` (or a dedicated header).
3. Register for Python: `.def_static("my_pass", &ir::pass::MyPass, "...")` on the Pass class in `python/src/bindings/ir/pass.cpp`.
4. Add the stub to `python/pypto/pypto_impl/__init__.pyi`.
5. Find the output library: `find build/framework/src -name "my_pass.cpp.o"` (usually libtile_fwk_interface.so).
6. Test it under `python/tests/ut/ir/`, following `test_common.py::run_merge_pass` (compile -> verify -> pass -> verify) and the snapshot pattern.

### Default pipeline (python/pypto/pil/compile_pipeline.py)

`infer_token -> canonicalize+dce x2 -> canonicalize(merge_stmts) -> remove_redundant_token -> create_root_functions -> finalize`. When inserting a new pass, account for the IRProperty contracts before/after it (SSAForm etc.).

### Verifiers (src/interface/ir/verifier/)

- Python entry `ir.IRVerifier.create_default()`; print diagnostics with `IRVerifier.generate_report(diagnostic)`.
- verify_ssa: use-before-def (LogicalTensor defined before use), if/loop yield counts match iterArgs/returnVars, duplicate assignment. type_check: yield value types match returnVars.
- **Insert a verify after every transform during pass development** (the run_merge_pass pattern) — the fastest way to catch SSA breakage introduced by a pass.

## 8. Build / Test / Debug Workflow

```bash
python build_ci.py --no_iso --editable          # Build (fixed order; --editable makes python/pypto edits live)
pytest python/tests/ut/ir/test_xxx.py            # Single-file test (no PYTHONPATH needed)
pytest python/tests/ut/ir/ --forked -q           # Batch runs MUST use --forked (global state leaks otherwise)
export ASCEND_SLOG_PRINT_TO_STDOUT=1             # Flush C++ crash/segfault stacks to stdout
```

- **Snapshot tests**: `test_common.py::check_snapshot`; goldens are `.pypto` text files (grammar in io_text.md). Update a golden with `PYPTO_RENDER_IR=1 pytest ...`, then **review the generated .pypto by hand before committing**.
- **IR dump**: `SetPassDefaultConfig(KEY_PRINT_GRAPH, True)` -> files under `LogTopFolder()/TensorGraph/IR/ir_dump_after_*.txt`.
- **.so double-tree trap**: after a C++ change a plain pytest may load the stale `python/pypto/pypto_impl*.so` (editable tree). On a `has no attribute` error run `python -c "import pypto.pypto_impl as i; print(i.__file__)"`, compare timestamps of both .so trees, and re-run build_ci when stale.
- Frontend debugging: `print(func.body)` dumps PIL (Block/Call have dump support); DispatchError carries a source-location stack.

## 9. Test-Writing Patterns (python/tests/ut/ir/)

### 9.1 Directory and naming conventions

```
python/tests/ut/ir/
|-- test_common.py              # Shared helpers (check_snapshot / ssa_verify / run_merge_pass)
|-- test_ir.py                  # Infrastructure: dtype/Span/error types
|-- test_parser.py              # L1 parser unit tests (inspect PIL Calls after ast2pil)
|-- test_pil.py                 # L2/L3 frontend: pil.compile behavior, loop carries
|-- test_fa*.py / test_incr_fa.py             # End-to-end kernels (FA etc.)
|-- test_merge_pass/            # Per-pass subdirectory: test_xxx.py + matching .pypto golden
|   |-- test_merge_pass1.py + test_merge_pass1.pypto
|   ...
|-- token/                      # Token-pass subdirectory (same py + pypto pairing)
```

Rules: **a golden `.pypto` file lives next to its test with the same base name**; gather many goldens into a `<test>_data/` subdirectory; start files with the copyright header + `# -*- coding: utf-8 -*-`; add ruff exemptions only when needed (`# ruff: noqa`).

## 10. Task Quick Reference

### Add an IR node (stmt/expr)
Add the ObjectKind in core.h -> define the class in stmt.h/expr.h (full-field constructor + span; GetKind/TypeName/GetFieldDescriptors) -> add the virtual to functor.h -> default implementations in visitor.cpp/mutator.cpp -> printing in printer.cpp -> serialization in io_text_dumper/loader -> structural_equal/hash need no manual change (they are descriptor-driven; confirm the new field appears in GetFieldDescriptors, which is the single source both use) -> binding in bindings/ir/ir.cpp -> verifier rules -> UT.

### Add a frontend op (user-visible API)
1. Add the API function in `python/pypto/__init__.py` (or the matching module).
2. Add `@impl(the_func)` in ops.py (pure-Python semantics that can execute directly need no registration).
3. Implementation: SymbolicScalar/Tensor operations end in `ctx.create_tensor_op_stmt` + `ctx.emit`; side-effect writes (`d.a=`, `t[:]=`) pair with `slot_write_inplace`/store_names.
4. Put IR-level tests under `python/tests/ut/ir/` (default); use `ut/interface/` only for final regression tests.

### Change the control-flow carry set (yield/carry)
The full chain of touch points: parser store_names -> ops.py `carriable_names`/`_loop_unroll`/`_if_else_stmt` yield set -> C++ canonicalize carry-liveness -> merge_stmts splice/clone. Missing one spot shows up as values lost after a branch, or canonicalize pruning a still-referenced carry (ori_shape / undefined-variable errors).

### Diagnose a pass-introduced problem
Run IRVerifier after each pass (run_merge_pass pattern) -> get per-stage dumps with KEY_PRINT_GRAPH and diff adjacent ones -> reproduce with minimal hand-built IR (`ir.IRBuilder()` + create_*, bypassing the frontend — see test_ir_builder.py and pattern C in section 9).

### Diagnose a frontend tracing problem
`print(Block)` to inspect PIL -> log `scope.resolve` values inside the impl -> mind the static/dynamic mode difference (a concrete cond traces only one branch) -> the DispatchError stack points at the source line.

## 11. Known Pitfalls

- IfStmt else is a `std::optional<SeqStmtsPtr>`: test with `has_value()`, rebuild through the nullopt split (section 6).
- `SeqStmts::Make` flattens; keep control-flow bodies as `SeqStmts` (`Wrap`) when rebuilding.
- A scalar reference inside dynvalidshape is a use: every new use-collection site must account for it.
- `VarExprMap` keys are **old-Var shared_ptrs** (pointer identity); update the map after cloning or later substitutions miss.
- `Var::Clone()` generates `name_clone_N`; SSA identity is the pointer, never the name.
- IRContext is a global singleton: the frontend calls `pypto_impl.Reset()` to clean it; mind concurrent compiles.
- Batch test runs without `--forked` produce false failures from global-state leaks (single runs pass).
