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

"""Variable table and context management for PTO Script Parser.

This module provides context management and variable scoping utilities for
parsing PTO scripts. It implements a stack-based variable table that tracks
variables across different scopes and blocks during parsing.

The main components are:
- VarTableFrame: Represents a single scope/block of variables
- VarTable: Stack of frames managing variable lifetime and shadowing
- Context managers for automatic frame cleanup
"""


from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from pypto.frontend.parser.doc_core import AST
from pypto.frontend.parser.error import ParserError


def _deferred(cleanup_function: Callable[[], None]) -> Iterator[None]:
    """Generate a context manager that executes a cleanup function on exit.

    Parameters
    ----------
    cleanup_function : Callable[[], None]
        The cleanup function to invoke when exiting the context manager.

    Returns
    -------
    result : Iterator[None]
        A context manager that yields None and executes the cleanup on exit.
    """

    @contextmanager
    def context_manager():
        try:
            yield
        finally:
            cleanup_function()

    return context_manager()


class ContextFrame:
    """Represents a single scope frame for variable storage.

    Each frame maintains a collection of variable names that were introduced
    within a specific block or scope during parsing.
    """

    vars: set[str]

    def __init__(self):
        self.vars = set()

    def add(self, variable_name: str, node: Optional[AST] = None) -> None:
        """Register a new variable name in this context frame.

        Parameters
        ----------
        variable_name : str
            The identifier of the variable to register.
        node : Optional[AST]
            The AST node associated with this variable, used for error reporting.
        """
        if variable_name in self.vars:
            if node is None:
                error_message = "Variable '{}' already exists in the current scope".format(
                    variable_name
                )
                raise NameError(error_message)
            else:
                error_message = "Variable '{}' already exists in the current scope".format(
                    variable_name
                )
                raise ParserError(node, NameError(error_message))
        self.vars.add(variable_name)

    def pop_all(self, removal_handler: Callable[[str], None]):
        """Remove all variables from this frame using the provided handler.

        Parameters
        ----------
        removal_handler : Callable[[str], None]
            The callback function to invoke for each variable being removed.
        """
        for identifier in self.vars:
            removal_handler(identifier)
        self.vars.clear()


class Context:
    """Manages variable scoping and context during PTO script parsing.

    Maintains a stack of context frames, each representing a different scope,
    and tracks variable values across these scopes.

    Parameters
    ----------
    frames : list[ContextFrame]
        The stack of context frames representing nested scopes.
    """

    frames: list[ContextFrame]
    name2value: dict[str, list[Any]]
    marked_for_deletion: set[str]

    def __init__(self):
        self.frames = []
        self.name2value = defaultdict(list)
        self.marked_for_deletion = set()

    def with_frame(self) -> Iterator[None]:
        """Establish a new context frame that can be used with a with statement.

        Returns
        -------
        result : Iterator[None]
            A context manager that creates a new frame and automatically
            cleans it up when the context exits.
        """

        def remove_frame() -> None:
            current_frame = self.frames.pop()
            current_frame.pop_all(
                lambda identifier: self.name2value[identifier].pop()
            )

        self.frames.append(ContextFrame())
        return _deferred(remove_frame)

    def add(
        self,
        var: str,
        value: Any,
        node: Optional[AST] = None,
        allow_update: bool = True,
    ) -> None:
        """Register or update a variable in the current context frame.

        Parameters
        ----------
        var : str
            The variable identifier.
        value : Any
            The value to associate with the variable.
        node : Optional[AST]
            The AST node for this variable, used for error reporting.
        allow_update : bool
            Whether updates to existing variables in the current frame are permitted.
        """
        if allow_update and var in self.frames[-1].vars:
            # Modify existing variable value
            self.name2value[var][-1] = value
        else:
            self.frames[-1].add(var, node)
            self.name2value[var].append(value)

    def get(self) -> dict[str, Any]:
        """Retrieve a dictionary containing the most recent value for each variable.

        Returns
        -------
        result : dict[str, Any]
            A dictionary mapping variable names to their latest values,
            including only variables that have values.
        """
        result_dict = {}
        for var_key, value_list in self.name2value.items():
            if value_list:
                result_dict[var_key] = value_list[-1]
        return result_dict

    def delete(self, var: str) -> None:
        """Remove a variable from the topmost context frame.

        Parameters
        ----------
        var : str
            The identifier of the variable to remove.
        """
        if len(self.frames) == 0:
            raise ValueError("Cannot delete variable outside of a frame")
        if var not in self.frames[-1].vars:
            error_msg = "Variable '{}' is not defined in the current scope".format(var)
            raise NameError(error_msg)

        # Eliminate from the top frame's variable collection
        self.frames[-1].vars.discard(var)
        # Eliminate the most recent value from the value stack
        if var in self.name2value:
            if self.name2value[var]:
                self.name2value[var].pop()

    def mark_for_deletion(self, var_names: set[str]) -> None:
        """Record variable names that should be deleted later.

        Parameters
        ----------
        var_names : set[str]
            A collection of variable identifiers to be marked for removal.
        """
        self.marked_for_deletion.update(var_names)

    def cleanup_marked(self) -> None:
        """Remove all variables that have been marked for deletion."""
        marked_list = list(self.marked_for_deletion)
        for identifier in marked_list:
            # Verify the variable exists and has associated values
            if identifier in self.name2value:
                if self.name2value[identifier]:
                    # Locate the frame containing this variable
                    for frame in reversed(self.frames):
                        if identifier in frame.vars:
                            # Eliminate from the frame
                            frame.vars.discard(identifier)
                            # Eliminate from the value stack
                            self.name2value[identifier].pop()
                            break
        # Reset the marked set
        self.marked_for_deletion.clear()
