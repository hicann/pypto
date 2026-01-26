#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Liveness analysis for automatic variable deletion.

This module implements liveness analysis for the PTO Script Parser, which
analyzes the abstract syntax tree (AST) to determine when variables are last
used and can be safely deleted. This optimization helps reduce memory usage
during execution by automatically cleaning up variables that are no longer needed.

Key Features:
    - Variable Use Tracking: Records all uses of each variable across statements
    - Last-Use Detection: Identifies the final statement where each variable is referenced
    - Automatic Deletion: Marks variables for deletion after their last use
    - Loop-Aware Analysis: Handles variables defined and used within loops appropriately
    - Exempt Variables: Allows certain variables (e.g., function parameters) to be excluded

The analyzer walks the AST using the visitor pattern, tracking variable definitions
and uses. After analysis, it generates a mapping from statement IDs to sets of
variable names that should be deleted after that statement executes.

Example:
    analyzer = LivenessAnalyzer()
    delete_after = analyzer.analyze(ast_node, exempt_vars={'input', 'output'})
    # delete_after = {stmt_id: {'temp1', 'temp2'}, ...}
"""

from typing import Optional

from . import doc


class LivenessAnalyzer(doc.NodeVisitor):
    """Analyzes variable liveness in the AST.

    Tracks variable uses and determines after which statements variables
    should be automatically deleted.
    """

    def __init__(self):
        # Maps statement id to set of variable names to delete after that statement
        self.delete_after: dict[int, set[str]] = {}
        # Maps variable name to list of statement IDs where it's used (in order)
        self.var_uses: dict[str, list[int]] = {}
        # Maps variable name to the statement ID where it's defined
        self.var_defs: dict[str, int] = {}
        # Current statement being analyzed
        self.current_stmt_id: Optional[int] = None
        # Variables that should never be auto-deleted
        self.exempt_vars: set[str] = set()
        # Counter for statement IDs
        self.stmt_counter = 0
        # Stack of loop statement IDs to handle nested loops
        self.loop_scope_stack: list[int] = []
        # Set of variables defined inside the current loop (cleared when entering/exiting loops)
        self.vars_defined_in_loop: set[str] = set()

    def analyze(
        self, node: doc.AST, exempt_vars: Optional[set[str]] = None
    ) -> dict[int, set[str]]:
        """Analyze the AST and return deletion points.

        Parameters
        ----------
        node : doc.AST
            The AST node to analyze.
        exempt_vars : Optional[set[str]]
            Variables that should not be auto-deleted (e.g., function arguments).

        Returns
        -------
        delete_after : dict[int, set[str]]
            Map from statement ID to variables to delete after that statement.
        """
        if exempt_vars:
            self.exempt_vars = exempt_vars
        self.visit(node)
        self._compute_deletion_points()
        return self.delete_after

    def visit(self, node: doc.AST):
        """Visit a node."""
        if isinstance(node, (list, tuple)):
            for item in node:
                self.visit(item)
            return
        if not isinstance(node, doc.AST):
            return
        return super().visit(node)

    # Statement visitors
    def visit_function_def(self, node: doc.FunctionDef):
        """Visit function definition."""
        # Mark function arguments as exempt
        for arg in node.args.args:
            self.exempt_vars.add(arg.arg)
        # Visit body
        for stmt in node.body:
            self.visit(stmt)

    def visit_assign(self, node: doc.Assign):
        """Visit assignment statement."""
        self.current_stmt_id = _get_node_id(node)
        # Visit RHS first to record uses
        self.visit(node.value)
        # Then record definitions on LHS
        for target in node.targets:
            self._visit_assign_target(target, is_def=True)

    def visit_ann_assign(self, node: doc.AnnAssign):
        """Visit annotated assignment."""
        self.current_stmt_id = _get_node_id(node)
        if node.value:
            self.visit(node.value)
        self._visit_assign_target(node.target, is_def=True)

    def visit_aug_assign(self, node: doc.AugAssign):
        """Visit augmented assignment."""
        self.current_stmt_id = _get_node_id(node)
        # Visit the value expression first to record uses
        self.visit(node.value)
        # Visit the target to record uses (for reading the current value)
        self.visit(node.target)
        # Then record definition on the target (for writing the new value)
        self._visit_assign_target(node.target, is_def=True)

    def visit_for(self, node: doc.For):
        """Visit for loop."""
        stmt_id = _get_node_id(node)
        self.current_stmt_id = stmt_id

        # Visit iterator expression (outside the loop scope)
        self.visit(node.iter)

        # Save variables defined in outer loop scope
        saved_vars_defined_in_loop = self.vars_defined_in_loop.copy()

        # Enter loop scope - variables used inside will be deleted after loop exits
        self.loop_scope_stack.append(stmt_id)
        # Reset for this loop body
        self.vars_defined_in_loop = set()

        # Loop variable is defined here
        # Support both single variable and tuple unpacking
        if isinstance(node.target, doc.Name):
            self._record_var_def(node.target.id)
            self.exempt_vars.add(node.target.id)  # Loop vars not auto-deleted
        elif isinstance(node.target, (doc.Tuple, doc.List)):
            # Tuple unpacking: for x, y in iterator
            for elt in node.target.elts:
                if isinstance(elt, doc.Name):
                    self._record_var_def(elt.id)
                    self.exempt_vars.add(elt.id)  # Loop vars not auto-deleted
                # Note: nested tuples in loop targets are not supported
                # (e.g., for (x, (y, z)) in iterator is not allowed)

        # Visit body
        for stmt in node.body:
            self.visit(stmt)

        # Exit loop scope and restore outer loop's vars
        self.loop_scope_stack.pop()
        self.vars_defined_in_loop = saved_vars_defined_in_loop

    def visit_while(self, node: doc.While):
        """Visit while loop."""
        raise NotImplementedError("While loop is not supported yet.")

    def visit_if(self, node: doc.If):
        """Visit if statement."""
        self.current_stmt_id = _get_node_id(node)
        # Visit condition
        self.visit(node.test)
        # Visit both branches
        for stmt in node.body:
            self.visit(stmt)
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_return(self, node: doc.Return):
        """Visit return statement."""
        self.current_stmt_id = _get_node_id(node)
        if node.value:
            self.visit(node.value)

    def visit_delete(self, node: doc.Delete):
        """Visit delete statement."""
        self.current_stmt_id = _get_node_id(node)
        # Explicit delete - mark these vars as exempt from auto-delete
        for target in node.targets:
            if isinstance(target, doc.Name):
                self.exempt_vars.add(target.id)

    def visit_expr(self, node: doc.Expr):
        """Visit expression statement."""
        self.current_stmt_id = _get_node_id(node)
        self.visit(node.value)

    def visit_pass(self, node: doc.Pass):
        """Visit pass statement."""
        self.current_stmt_id = _get_node_id(node)

    # Expression visitors
    def visit_name(self, node: doc.Name):
        """Visit name (variable reference)."""
        if isinstance(node.ctx, doc.Load):
            # This is a use/read
            self._record_var_use(node.id)

    def visit_call(self, node: doc.Call):
        """Visit function call."""
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_attribute(self, node: doc.Attribute):
        """Visit attribute access."""
        self.visit(node.value)

    def visit_subscript(self, node: doc.Subscript):
        """Visit subscript."""
        self.visit(node.value)
        self.visit(node.slice)

    def visit_bin_op(self, node: doc.BinOp):
        """Visit binary operation."""
        self.visit(node.left)
        self.visit(node.right)

    def visit_unary_op(self, node: doc.UnaryOp):
        """Visit unary operation."""
        self.visit(node.operand)

    def visit_compare(self, node: doc.Compare):
        """Visit comparison."""
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_tuple(self, node: doc.Tuple):
        """Visit tuple."""
        for elt in node.elts:
            self.visit(elt)

    def visit_list(self, node: doc.List):
        """Visit list."""
        for elt in node.elts:
            self.visit(elt)

    def visit_slice(self, node: doc.Slice):
        """Visit slice."""
        if node.lower:
            self.visit(node.lower)
        if node.upper:
            self.visit(node.upper)
        if node.step:
            self.visit(node.step)

    def _compute_deletion_points(self):
        """Compute where each variable should be deleted based on usage."""
        for var_name, use_ids in self.var_uses.items():
            if var_name in self.exempt_vars:
                continue
            if not use_ids:
                # Variable defined but never used - delete after definition
                if var_name in self.var_defs:
                    stmt_id = self.var_defs[var_name]
                    if stmt_id not in self.delete_after:
                        self.delete_after[stmt_id] = set()
                    self.delete_after[stmt_id].add(var_name)
            else:
                # Delete after last use
                last_use_id = use_ids[-1]
                if last_use_id not in self.delete_after:
                    self.delete_after[last_use_id] = set()
                self.delete_after[last_use_id].add(var_name)

    def _next_stmt_id(self) -> int:
        """Generate next statement ID."""
        self.stmt_counter += 1
        return self.stmt_counter

    def _record_var_use(self, var_name: str):
        """Record a variable use at the current statement.

        If inside a loop scope, records the use at the loop level to ensure
        variables aren't deleted inside the loop body, UNLESS the variable
        was defined inside the loop (in which case it can be deleted per-iteration).
        """
        if self.current_stmt_id is not None:
            if var_name not in self.var_uses:
                self.var_uses[var_name] = []
            # If we're inside a loop and the variable was defined OUTSIDE the loop,
            # use the loop's statement ID to delete after loop exits.
            # If defined INSIDE the loop, use actual statement ID to delete per-iteration.
            if self.loop_scope_stack and var_name not in self.vars_defined_in_loop:
                stmt_id = self.loop_scope_stack[-1]  # Use innermost loop scope
            else:
                stmt_id = self.current_stmt_id
            self.var_uses[var_name].append(stmt_id)

    def _record_var_def(self, var_name: str):
        """Record a variable definition at the current statement."""
        if self.current_stmt_id is not None:
            self.var_defs[var_name] = self.current_stmt_id
            # Track if this variable is defined inside a loop
            if self.loop_scope_stack:
                self.vars_defined_in_loop.add(var_name)

    def _visit_assign_target(self, target: doc.expr, is_def: bool = False):
        """Visit assignment target."""
        if isinstance(target, doc.Name):
            if is_def:
                self._record_var_def(target.id)
        elif isinstance(target, (doc.Tuple, doc.List)):
            for elt in target.elts:
                self._visit_assign_target(elt, is_def)
        elif isinstance(target, doc.Subscript):
            # a[i] = x means 'a' is used, not defined
            self.visit(target.value)
            self.visit(target.slice)


def _get_node_id(node: doc.AST) -> int:
    """Get a unique ID for a node."""
    return id(node)
