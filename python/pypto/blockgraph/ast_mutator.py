#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import ast
import inspect
import textwrap
from typing import Optional, List


class AstMutator(ast.NodeTransformer):
    """
    Transforms Python AST to use low-level block API calls.

    Major transformations:
    1. Change Python native `for`, `if`, `else` to context manager `with block.for_scope`,
       `with block.if_then_scope`, `block.if_else_scope` (always need one `block.exit_if(ifs)`
       for single `if` or `if` + `else` pair)
    2. Change type annotations to `FunctionSignature` creation.
    3. Change `return` statement to `block.create_return` call.
    """

    def __init__(self):
        super().__init__()
        self.if_counter = 0  # Counter for unique if node variable names
        self.for_counter = 0  # Counter for unique for node variable names
        self.loop_vars = (
            set()
        )  # Track loop variables (ir.Scalar) created from block.loop
        self.func_params = (
            {}
        )  # Track function parameters and their types from annotations

    @classmethod
    def mutate_ast(cls, func, dump_source: Optional[str] = None) -> ast.FunctionDef:
        """
        Transform a function AST to use low-level block API calls.

        Args:
            func: The function to transform
            dump_source: Optional path to dump transformed source code

        Returns:
            The transformed AST wrapper function node
        """
        # Get source code and parse AST
        source = inspect.getsource(func)
        # Dedent the source code to remove surrounding indentation
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Extract function definition (first statement should be the function)
        func_node = tree.body[0]
        if not isinstance(func_node, ast.FunctionDef):
            raise ValueError("Expected function definition")

        # Transform AST
        mutator = cls()

        # Create wrapper function that accepts metadata
        wrapper = mutator._create_wrapper(func_node)

        # Fix missing locations for AST nodes (required for ast.unparse and compilation)
        ast.fix_missing_locations(wrapper)

        # Dump transformed source if requested
        if dump_source:
            mutator._dump_source(wrapper, dump_source)

        return wrapper

    def _is_scalar_annotation(self, annotation: ast.AST) -> bool:
        """
        Check if an annotation is ir.Scalar(...).

        Args:
            annotation: AST node representing the type annotation

        Returns:
            True if the annotation is ir.Scalar(...), False otherwise
        """
        if isinstance(annotation, ast.Call):
            func = annotation.func
            if isinstance(func, ast.Attribute):
                # Check for ir.Scalar(...)
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "ir"
                    and func.attr == "Scalar"
                ):
                    return True
        return False

    def _transform_statement(self, stmt: ast.AST) -> List[ast.AST]:
        """Transform a statement, potentially returning multiple statements."""
        if isinstance(stmt, ast.For):
            return self._transform_for(stmt)
        elif isinstance(stmt, ast.If):
            return self._transform_if(stmt)
        elif isinstance(stmt, ast.Return):
            return [self._transform_return(stmt)]
        else:
            # Visit other statements normally
            visited = self.visit(stmt)
            return [visited] if visited is not None else []

    def _create_wrapper(self, func_node: ast.FunctionDef) -> ast.FunctionDef:
        """
        Create a wrapper function that:
        1. Extracts function signature from annotations
        2. Creates FunctionSignature
        3. Creates function using block.create_function
        4. Wraps body in block.function_scope
        """
        # Create wrapper function body
        wrapper_body = []

        # Note: Closure variables are injected into the namespace before exec,
        # so they'll be available in __globals__ when the function executes

        # Create argument variables from annotations (they are constructor calls)
        arg_names = []
        for arg in func_node.args.args:
            if arg.annotation:
                # The annotation is a constructor call like ir.Tensor(...)
                # NOTE: insert code: `arg_name = annotation``
                wrapper_body.append(
                    ast.Assign(
                        targets=[ast.Name(id=arg.arg, ctx=ast.Store())],
                        value=arg.annotation,
                    )
                )
                arg_names.append(ast.Name(id=arg.arg, ctx=ast.Load()))

                # Track if this parameter is an ir.Scalar
                if self._is_scalar_annotation(arg.annotation):
                    self.func_params[arg.arg] = True

        # Extract return types from return annotation
        return_types = []
        if func_node.returns:
            if isinstance(func_node.returns, ast.Tuple):
                return_types = func_node.returns.elts
            else:
                return_types = [func_node.returns]

        # Create FunctionSignature
        wrapper_body.append(
            ast.Assign(
                targets=[ast.Name(id="sig", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="ir", ctx=ast.Load()),
                        attr="FunctionSignature",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                ),
            )
        )

        # Set sig.arguments
        wrapper_body.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="sig", ctx=ast.Load()),
                        attr="arguments",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.List(elts=arg_names, ctx=ast.Load()),
            )
        )

        # Set sig.returns
        wrapper_body.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="sig", ctx=ast.Load()),
                        attr="returns",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.List(elts=return_types, ctx=ast.Load()),
            )
        )

        # Create function: func = block.create_function(metadata["name"], metadata["function_kind"], sig)
        wrapper_body.append(
            ast.Assign(
                targets=[ast.Name(id="func", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="block", ctx=ast.Load()),
                        attr="create_function",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Subscript(
                            value=ast.Name(id="metadata", ctx=ast.Load()),
                            slice=ast.Constant(value="name"),
                            ctx=ast.Load(),
                        ),
                        ast.Subscript(
                            value=ast.Name(id="metadata", ctx=ast.Load()),
                            slice=ast.Constant(value="function_kind"),
                            ctx=ast.Load(),
                        ),
                        ast.Name(id="sig", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            )
        )

        # Transform function body - expand control flow nodes
        transformed_body = []
        for stmt in func_node.body:
            transformed_body.extend(self._transform_statement(stmt))

        # Wrap function body in block.function_scope(func)
        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="block", ctx=ast.Load()),
                            attr="function_scope",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id="func", ctx=ast.Load())],
                        keywords=[],
                    ),
                    optional_vars=None,
                )
            ],
            body=transformed_body,
        )
        wrapper_body.append(with_stmt)

        # Return func
        wrapper_body.append(ast.Return(value=ast.Name(id="func", ctx=ast.Load())))

        # Create wrapper function
        wrapper = ast.FunctionDef(
            name=func_node.name,
            args=ast.arguments(
                args=[ast.arg(arg="metadata", annotation=None)],
                posonlyargs=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=wrapper_body,
            decorator_list=[],
            returns=None,
        )

        return wrapper

    def _transform_for(self, node: ast.For) -> List[ast.AST]:
        """
        Transform: for i in block.loop(start, end, step, **kwargs):
        To:
            i = block.scalar(ir.DataType.int32, "i")
            fs = block.for_node(i, start, end, step, **kwargs)
            with block.for_scope(fs):
                ...
        """
        # Check if iter is block.loop call
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and isinstance(node.iter.func.value, ast.Name)
            and node.iter.func.value.id == "block"
            and node.iter.func.attr == "loop"
        ):
            # Not a block.loop, keep as is
            return [self.generic_visit(node)]

        # Extract loop variable name
        loop_var_name = node.target.id if isinstance(node.target, ast.Name) else "i"

        # Track this loop variable as an ir.Scalar
        self.loop_vars.add(loop_var_name)

        # Extract arguments from block.loop(start, end, step, **kwargs)
        loop_args = node.iter.args
        loop_kwargs = {kw.arg: kw.value for kw in node.iter.keywords}

        # Determine start, end, step
        start = (
            self.visit(loop_args[0]) if len(loop_args) > 0 else ast.Constant(value=0)
        )
        end = (
            self.visit(loop_args[1])
            if len(loop_args) > 1
            else (
                self.visit(loop_args[0])
                if len(loop_args) > 0
                else ast.Constant(value=0)
            )
        )
        step = self.visit(loop_args[2]) if len(loop_args) > 2 else ast.Constant(value=1)

        # NOTE: insert code `i = block.scalar(ir.DataType.int32, "i")`
        scalar_assign = ast.Assign(
            targets=[ast.Name(id=loop_var_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="block", ctx=ast.Load()),
                    attr="scalar",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="ir", ctx=ast.Load()),
                            attr="DataType",
                            ctx=ast.Load(),
                        ),
                        attr="int32",
                        ctx=ast.Load(),
                    ),
                    ast.Constant(value=loop_var_name),
                ],
                keywords=[],
            ),
        )

        # Generate unique variable name for for node
        for_var_name = f"fs_{self.for_counter}"
        self.for_counter += 1

        # NOTE: insert code `fs = block.for_node(i, start, end, step, **kwargs)`
        fornode_kwargs = []
        for key, value in loop_kwargs.items():
            fornode_kwargs.append(ast.keyword(arg=key, value=self.visit(value)))

        fornode_assign = ast.Assign(
            targets=[ast.Name(id=for_var_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="block", ctx=ast.Load()),
                    attr="for_node",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=loop_var_name, ctx=ast.Load()), start, end, step],
                keywords=fornode_kwargs,
            ),
        )

        # Transform body - expand nested control flow
        transformed_body = []
        for stmt in node.body:
            transformed_body.extend(self._transform_statement(stmt))

        # Create: with block.for_scope(fs):
        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="block", ctx=ast.Load()),
                            attr="for_scope",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=for_var_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                    optional_vars=None,
                )
            ],
            body=transformed_body,
        )

        return [scalar_assign, fornode_assign, with_stmt]

    def _is_scalar_condition(self, test: ast.AST) -> bool:
        """
        Check if the condition in an if statement is an ir.Scalar.

        Only transforms when we're certain it's an ir.Scalar:
        - Loop variables (from block.loop)
        - Function parameters that are ir.Scalar (from type annotations)

        Does NOT transform:
        - Python bool constants (True/False)
        - Other variables (assumed to be Python bools unless proven otherwise)

        Args:
            test: AST node representing the condition

        Returns:
            True if the condition is definitely an ir.Scalar, False otherwise
        """
        # If it's a Python constant (True/False), it's not an ir.Scalar
        if isinstance(test, ast.Constant):
            if isinstance(test.value, bool):
                return False

        # If it's a Name, only transform if it's a known ir.Scalar
        if isinstance(test, ast.Name):
            # Check if it's a loop variable (ir.Scalar)
            if test.id in self.loop_vars:
                return True
            # Check if it's a function parameter that's ir.Scalar
            if test.id in self.func_params:
                return True
            # Otherwise, assume it's a Python bool and don't transform
            return False

        # For other cases (attribute access, function calls, etc.),
        # be conservative and don't transform unless we're certain
        # This prevents transforming Python bool expressions
        return False

    def _transform_if(self, node: ast.If) -> List[ast.AST]:
        """
        Transform: if cond: ... else: ...
        To:
            ifs = block.if_node(cond)
            with block.if_then_scope(ifs):
                ...
            with block.if_else_scope(ifs):
                ...
            block.exit_if(ifs)

        Only transforms when cond is an ir.Scalar, not a Python bool.
        """
        # Check if condition is an ir.Scalar
        if not self._is_scalar_condition(node.test):
            # It's a Python bool, don't transform - just visit normally
            return [self.generic_visit(node)]

        # Generate unique variable name for if node
        if_var_name = f"ifs_{self.if_counter}"
        self.if_counter += 1

        # Transform condition
        cond = self.visit(node.test)

        # NOTE: Insert code `ifs = block.if_node(cond)`
        ifnode_assign = ast.Assign(
            targets=[ast.Name(id=if_var_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="block", ctx=ast.Load()),
                    attr="if_node",
                    ctx=ast.Load(),
                ),
                args=[cond],
                keywords=[],
            ),
        )

        # Transform then body - expand nested control flow
        then_body = []
        for stmt in node.body:
            then_body.extend(self._transform_statement(stmt))

        # Create: with block.if_then_scope(ifs):
        then_with = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="block", ctx=ast.Load()),
                            attr="if_then_scope",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=if_var_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                    optional_vars=None,
                )
            ],
            body=then_body,
        )

        # Transform else body if exists
        statements = [ifnode_assign, then_with]

        if node.orelse:
            else_body = []
            for stmt in node.orelse:
                else_body.extend(self._transform_statement(stmt))
            # Create: with block.if_else_scope(ifs):
            else_with = ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="block", ctx=ast.Load()),
                                attr="if_else_scope",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Name(id=if_var_name, ctx=ast.Load())],
                            keywords=[],
                        ),
                        optional_vars=None,
                    )
                ],
                body=else_body,
            )
            statements.append(else_with)

        # NOTE: insert code `block.exit_if(ifs)``
        exit_if = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="block", ctx=ast.Load()),
                    attr="exit_if",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=if_var_name, ctx=ast.Load())],
                keywords=[],
            )
        )
        statements.append(exit_if)

        return statements

    def _transform_return(self, node: ast.Return) -> ast.AST:
        """
        Transform: return (value1, value2, ...)
        To: block.create_return([value1, value2, ...])
        """
        if node.value is None:
            # return without value
            return ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="block", ctx=ast.Load()),
                        attr="create_return",
                        ctx=ast.Load(),
                    ),
                    args=[ast.List(elts=[], ctx=ast.Load())],
                    keywords=[],
                )
            )

        # Transform return value
        value = self.visit(node.value)

        # If it's a tuple, extract elements; otherwise wrap in list
        if isinstance(value, ast.Tuple):
            return_values = value.elts
        else:
            return_values = [value]

        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="block", ctx=ast.Load()),
                    attr="create_return",
                    ctx=ast.Load(),
                ),
                args=[ast.List(elts=return_values, ctx=ast.Load())],
                keywords=[],
            )
        )

    def _dump_source(self, func_node: ast.FunctionDef, dump_path: str):
        """Dump transformed source code to file for inspection."""
        source = ast.unparse(func_node)
        with open(dump_path, "w") as f:
            f.write(source)


def ast_to_ir(metadata=None, closure_vars=None, verbose=False):
    """
    Decorator that transforms a function's AST and returns an IR function.

    Args:
        metadata: Dictionary containing function metadata (name, function_kind, etc.)
        closure_vars: Dictionary of closure variables needed by the function (e.g., tensor_shape, block, etc.)

    Returns:
        The IR function (ir.Function) instead of the original Python function.
    """
    # Capture the parameters to avoid closure issues
    provided_metadata = metadata
    provided_closure_vars = closure_vars or {}

    def decorator(func):
        # Get transformed AST (without executing)
        transformed_ast = AstMutator.mutate_ast(func)
        module_ast = ast.Module(body=[transformed_ast], type_ignores=[])
        ast.fix_missing_locations(module_ast)

        if verbose:
            print("Low-level builder calls:\n", ast.unparse(module_ast))

        code = compile(module_ast, filename="<ast>", mode="exec")

        # Get the closure variables from the function
        func_closure_vars = inspect.getclosurevars(func)

        # The transformed function needs access to:
        # - Module-level globals (ir, BlockBuilderHelper, etc.) from func.__globals__
        # - Closure variables (explicitly provided + those from function closure)
        exec_namespace = {
            **func.__globals__,  # Include global imports (ir, BlockBuilderHelper, etc.)
            **func_closure_vars.nonlocals,  # Include closure vars from function
            **provided_closure_vars,  # Include explicitly provided closure vars
        }

        exec(code, exec_namespace)
        my_kernel_transformed = exec_namespace[func.__name__]

        # Use provided metadata or default
        if provided_metadata is None:
            func_metadata = {
                "name": func.__name__,
                "function_kind": ir.FunctionKind.ControlFlow,
            }
        else:
            func_metadata = provided_metadata

        func_ir = my_kernel_transformed(func_metadata)
        return func_ir

    return decorator
