#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scope-based liveness analysis for automatic variable deletion.

This module implements scope-aware liveness analysis for the PTO Script Parser.
It analyzes the abstract syntax tree (AST) to determine when variables are last
used and can be safely deleted, with proper handling of scope boundaries
and scope lifting for cross-scope variable usage.

Unified Deletion Rules:
    Rule A: Variable deletion point should be at the same scope level as definition.
            Delete after last use within that scope.
    Rule B: If use scope and definition scope have no nesting relationship,
            lift definition scope to common ancestor, then apply Rule A.

Note:
    This analyzer records ALL variables regardless of type. The actual tensor
    type filtering is handled by parser.py's _filter_tensor_vars() method
    during execution, which uses runtime type information (isinstance checks).
"""

import ast
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List, Tuple

from pypto.error import FeError


@dataclass
class Scope:
    """Represents a code block scope in the AST.

    A scope corresponds to a code block such as:
    - Function body (root scope)
    - For loop body
    - If body / Else body

    Each scope maintains information about:
    - Variables defined within it
    - Entry and exit statement IDs for deletion timing
    - Parent-child relationships forming a scope tree
    """
    scope_id: int
    scope_type: str
    parent: Optional['Scope'] = None
    children: List['Scope'] = field(default_factory=list)
    tensor_defs: Set[str] = field(default_factory=set)
    exit_stmt_id: Optional[int] = None
    entry_stmt_id: Optional[int] = None
    stmt_ids: List[int] = field(default_factory=list)

    def is_nested_in(self, other: 'Scope') -> bool:
        """Check if this scope is nested within another scope."""
        current = self.parent
        while current:
            if current.scope_id == other.scope_id:
                return True
            current = current.parent
        return False

    def find_common_ancestor(self, other: 'Scope') -> Optional['Scope']:
        """Find the common ancestor scope of two scopes."""
        ancestors: Dict[int, Scope] = {}
        current = self
        while current:
            ancestors[current.scope_id] = current
            current = current.parent

        current = other
        while current:
            if current.scope_id in ancestors:
                return ancestors[current.scope_id]
            current = current.parent
        return None

    def depth(self) -> int:
        """Calculate the depth of this scope in the scope tree."""
        d = 0
        current = self.parent
        while current:
            d += 1
            current = current.parent
        return d


@dataclass
class VarInfo:
    """Information about a variable for liveness analysis."""
    var_name: str
    def_scope_id: int
    def_stmt_id: int
    use_points: List[Tuple[int, int]] = field(default_factory=list)
    needs_scope_lift: bool = False
    lift_target_scope_id: Optional[int] = None
    delete_after_stmt_id: Optional[int] = None
    delete_after_scope_exit: bool = False
    delete_scope_id: Optional[int] = None


@dataclass
class LivenessResult:
    """Structured result from scope-based liveness analysis."""
    var_info: Dict[str, VarInfo] = field(default_factory=dict)
    scope_lift_suggestions: List[Tuple[str, int, int]] = field(default_factory=list)
    delete_points: Dict[int, Set[str]] = field(default_factory=dict)
    scope_tree: Optional[Scope] = None
    scope_map: Dict[int, Scope] = field(default_factory=dict)


class ScopeLivenessAnalyzer(ast.NodeVisitor):
    """Scope-based liveness analyzer for variables.

    This analyzer builds a scope tree from the AST, collects variable
    definitions and uses, and applies unified deletion rules.

    Type filtering is handled by parser.py during execution using
    _filter_tensor_vars() which checks isinstance(value, pypto.Tensor).
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.result: LivenessResult = LivenessResult()
        self.exempt_vars: Set[str] = set()
        self.scope_stack: List[Scope] = []
        self.current_scope: Optional[Scope] = None
        self.scope_counter: int = 0
        self.current_stmt_id: Optional[int] = None

    def analyze(
        self,
        node: ast.AST,
        exempt_vars: Optional[Set[str]] = None
    ) -> LivenessResult:
        """Analyze the AST and return structured liveness result.

        Parameters
        ----------
        node : ast.AST
            The AST node to analyze (typically FunctionDef).
        exempt_vars : Optional[Set[str]]
            Variables that should not be auto-deleted (e.g., function arguments).

        Returns
        -------
        LivenessResult
            Structured analysis result containing deletion points,
            scope lifting suggestions, and detailed variable info.
        """
        if exempt_vars:
            self.exempt_vars = set(exempt_vars)

        self.visit(node)
        self._apply_deletion_rules()

        return self.result

    # -------------------------------------------------------------------------
    # Statement Visitors
    # -------------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition - creates root scope."""
        for arg in node.args.args:
            self.exempt_vars.add(arg.arg)

        root_scope = self._create_scope(
            scope_type='root',
            entry_stmt_id=self._get_node_id(node)
        )
        self.result.scope_tree = root_scope
        self._push_scope(root_scope)

        for stmt in node.body:
            self.visit(stmt)

        root_scope.exit_stmt_id = self._get_node_id(node)
        self._pop_scope()

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        self.visit(node.value)

        for target in node.targets:
            self._visit_assign_target(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignment."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        if node.value:
            self.visit(node.value)

        self._visit_assign_target(node.target, node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        self.visit(node.value)
        self.visit(node.target)

    def visit_For(self, node: ast.For) -> None:
        """Visit for loop - creates loop scope."""
        stmt_id = self._get_node_id(node)
        self.current_stmt_id = stmt_id

        if self.current_scope:
            self.current_scope.stmt_ids.append(stmt_id)

        self.visit(node.iter)

        loop_scope = self._create_scope(
            scope_type='for',
            entry_stmt_id=stmt_id,
            exit_stmt_id=stmt_id
        )
        self._push_scope(loop_scope)

        if isinstance(node.target, ast.Name):
            loop_var = node.target.id
            self._record_var_def(loop_var, stmt_id)
            self.exempt_vars.add(loop_var)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self._record_var_def(elt.id, stmt_id)
                    self.exempt_vars.add(elt.id)

        for stmt in node.body:
            self.visit(stmt)

        self._pop_scope()

    def visit_While(self, node: ast.While) -> None:
        """Visit while loop - not supported."""
        raise FeError(NotImplementedError("While loop is not supported yet."))

    def visit_If(self, node: ast.If) -> None:
        """Visit if statement - creates if scope with body/else sub-scopes."""
        stmt_id = self._get_node_id(node)
        self.current_stmt_id = stmt_id

        if self.current_scope:
            self.current_scope.stmt_ids.append(stmt_id)

        self.visit(node.test)

        if_scope = self._create_scope(
            scope_type='if',
            entry_stmt_id=stmt_id,
            exit_stmt_id=stmt_id
        )
        self._push_scope(if_scope)

        if_body_scope = self._create_scope(scope_type='if_body')
        self._push_scope(if_body_scope)

        for stmt in node.body:
            self.visit(stmt)

        self._pop_scope()

        if node.orelse:
            else_body_scope = self._create_scope(scope_type='else_body')
            self._push_scope(else_body_scope)

            for stmt in node.orelse:
                self.visit(stmt)

            self._pop_scope()

        self._pop_scope()

    def visit_Return(self, node: ast.Return) -> None:
        """Visit return statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        if node.value:
            self.visit(node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Visit explicit delete statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        for target in node.targets:
            if isinstance(target, ast.Name):
                self.exempt_vars.add(target.id)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Visit expression statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

        self.visit(node.value)

    def visit_Pass(self, node: ast.Pass) -> None:
        """Visit pass statement."""
        self.current_stmt_id = self._get_node_id(node)

        if self.current_scope:
            self.current_scope.stmt_ids.append(self.current_stmt_id)

    # -------------------------------------------------------------------------
    # Expression Visitors
    # -------------------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        """Visit name node - record variable uses."""
        if isinstance(node.ctx, ast.Load):
            self._record_var_use(node.id, self.current_stmt_id)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call."""
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit attribute access."""
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Visit subscript access."""
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Visit binary operation."""
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Visit unary operation."""
        self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Visit comparison."""
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        """Visit tuple."""
        for elt in node.elts:
            self.visit(elt)

    def visit_List(self, node: ast.List) -> None:
        """Visit list."""
        for elt in node.elts:
            self.visit(elt)

    def visit_Slice(self, node: ast.Slice) -> None:
        """Visit slice."""
        if node.lower:
            self.visit(node.lower)
        if node.upper:
            self.visit(node.upper)
        if node.step:
            self.visit(node.step)

    # -------------------------------------------------------------------------
    # Deletion Rule Application (Unified)
    # -------------------------------------------------------------------------

    def _apply_deletion_rules(self) -> None:
        """Apply unified deletion rules to compute deletion points.

        Algorithm:
        1. Check Rule B - if non-nested usage exists, perform scope lifting
        2. Map all use points to def_scope's stmt_ids:
           - Use in def_scope → use stmt_id directly
           - Use in nested scope → climb to direct child scope, use its entry_stmt_id
        3. Find last stmt_id with use in def_scope.stmt_ids list (execution order)
        4. Set deletion point after that stmt_id
        """
        for var_name, var_info in self.result.var_info.items():
            if var_name in self.exempt_vars:
                continue

            if not var_info.use_points:
                self.result.delete_points.setdefault(var_info.def_stmt_id, set()).add(var_name)
                continue

            self._compute_delete_point_unified(var_info)

        self._build_delete_points_from_var_info()

    def _compute_delete_point_unified(self, var_info: VarInfo) -> None:
        """Compute deletion point using unified rules."""
        def_scope = self.result.scope_map.get(var_info.def_scope_id)
        if not def_scope:
            return

        use_scope_ids = [scope_id for scope_id, _ in var_info.use_points]
        use_scopes = [self.result.scope_map.get(sid) for sid in use_scope_ids]
        use_scopes = [s for s in use_scopes if s]

        if not use_scopes:
            return

        effective_def_scope = self._check_scope_lift(var_info, def_scope, use_scopes)

        self._set_delete_point(var_info, effective_def_scope)

    def _check_scope_lift(
        self,
        var_info: VarInfo,
        def_scope: Scope,
        use_scopes: List[Scope]
    ) -> Scope:
        """Check Rule B: Non-nested usage requires scope lifting.

        Returns effective definition scope (may be lifted).
        """
        non_nested_uses = []
        for use_scope in use_scopes:
            is_nested = use_scope.is_nested_in(def_scope)
            is_same = use_scope.scope_id == def_scope.scope_id
            if not is_nested and not is_same:
                non_nested_uses.append(use_scope)

        if not non_nested_uses:
            return def_scope

        common_ancestor = None
        for use_scope in use_scopes:
            ancestor = def_scope.find_common_ancestor(use_scope)
            if ancestor:
                if common_ancestor is None or ancestor.depth() > common_ancestor.depth():
                    common_ancestor = ancestor

        if common_ancestor:
            var_info.needs_scope_lift = True
            var_info.lift_target_scope_id = common_ancestor.scope_id

            self.result.scope_lift_suggestions.append(
                (var_info.var_name, var_info.def_scope_id, common_ancestor.scope_id)
            )

            return common_ancestor

        return def_scope

    def _set_delete_point(
        self,
        var_info: VarInfo,
        def_scope: Scope
    ) -> None:
        """Set deletion point based on representative scope.
        
        Unified logic for both same-scope and cross-scope usage.
        Maps all use points to def_scope's stmt_ids and finds the last one
        in execution order to ensure deletion happens after all uses complete.
        """
        stmt_ids_info = {}

        for scope_id, stmt_id in var_info.use_points:
            use_scope = self.result.scope_map.get(scope_id)
            if not use_scope:
                continue

            if scope_id == def_scope.scope_id:
                stmt_ids_info[stmt_id] = False
            elif use_scope.is_nested_in(def_scope):
                current = use_scope
                while current.parent and current.parent.scope_id != def_scope.scope_id:
                    current = current.parent

                if current.entry_stmt_id:
                    stmt_ids_info[current.entry_stmt_id] = True

        last_stmt_id = None
        last_from_nested = False
        for sid in def_scope.stmt_ids:
            if sid in stmt_ids_info:
                last_stmt_id = sid
                last_from_nested = stmt_ids_info[sid]

        if last_stmt_id:
            var_info.delete_after_stmt_id = last_stmt_id
            var_info.delete_scope_id = def_scope.scope_id
            var_info.delete_after_scope_exit = last_from_nested
        else:
            var_info.delete_after_stmt_id = var_info.def_stmt_id
            var_info.delete_scope_id = def_scope.scope_id
            var_info.delete_after_scope_exit = False

    def _build_delete_points_from_var_info(self) -> None:
        """Build delete_points dict from VarInfo."""
        for var_name, var_info in self.result.var_info.items():
            if var_name in self.exempt_vars:
                continue

            if var_info.delete_after_stmt_id:
                self.result.delete_points.setdefault(
                    var_info.delete_after_stmt_id, set()
                ).add(var_name)

    def _next_scope_id(self) -> int:
        """Generate next scope ID."""
        self.scope_counter += 1
        return self.scope_counter

    def _get_node_id(self, node: ast.AST) -> int:
        """Get unique ID for an AST node."""
        return id(node)

    def _create_scope(
        self,
        scope_type: str,
        entry_stmt_id: Optional[int] = None,
        exit_stmt_id: Optional[int] = None
    ) -> Scope:
        """Create a new scope and add to tree."""
        scope = Scope(
            scope_id=self._next_scope_id(),
            scope_type=scope_type,
            parent=self.current_scope,
            entry_stmt_id=entry_stmt_id,
            exit_stmt_id=exit_stmt_id
        )

        if self.current_scope:
            self.current_scope.children.append(scope)

        self.result.scope_map[scope.scope_id] = scope

        return scope

    def _push_scope(self, scope: Scope) -> None:
        """Push a scope onto the stack and set as current."""
        self.scope_stack.append(scope)
        self.current_scope = scope

    def _pop_scope(self) -> Optional[Scope]:
        """Pop the current scope from stack."""
        if self.scope_stack:
            popped = self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1] if self.scope_stack else None
            return popped
        return None

    def _record_var_def(self, var_name: str, stmt_id: Optional[int]) -> None:
        """Record a variable definition.

        For redefinitions, keep the original def_scope_id (first definition)
        and only update def_stmt_id to track the last definition statement.
        This ensures deletion happens at the outermost definition scope level.
        """
        if not self.current_scope:
            return

        if stmt_id is None:
            return

        self.current_scope.tensor_defs.add(var_name)

        if var_name not in self.result.var_info:
            self.result.var_info[var_name] = VarInfo(
                var_name=var_name,
                def_scope_id=self.current_scope.scope_id,
                def_stmt_id=stmt_id
            )
        else:
            info = self.result.var_info[var_name]
            info.def_stmt_id = stmt_id

    def _record_var_use(self, var_name: str, stmt_id: Optional[int]) -> None:
        """Record a variable use."""
        if stmt_id is None:
            return

        if var_name in self.exempt_vars:
            return

        if var_name not in self.result.var_info:
            return

        scope_id = self.current_scope.scope_id if self.current_scope else 0
        self.result.var_info[var_name].use_points.append((scope_id, stmt_id))

    def _visit_assign_target(
        self,
        target: ast.expr,
        value_node: Optional[ast.AST] = None,
        ann_type: Optional[str] = None
    ) -> None:
        """Visit assignment target and record variable definitions."""
        if isinstance(target, ast.Name):
            var_name = target.id

            if var_name in self.exempt_vars:
                return

            self._record_var_def(var_name, self.current_stmt_id)

        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._visit_assign_target(elt, value_node, ann_type)

        elif isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)