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

"""PTO Script Parser Source and Diagnostics.

This module provides source code management and diagnostic reporting utilities
for the PTO Script Parser. It handles:

1. Source Code Management:
   - Reading and caching source code from functions, strings, or files
   - Converting source to Python AST and PTO doc AST
   - Tracking line numbers and source locations

2. Diagnostic Reporting:
   - Error, warning, and info message formatting
   - Pretty-printing of errors with source context
   - Color-coded terminal output for better readability

Key Classes:
    - Source: Represents source code with location information
    - Diagnostics: Manages diagnostic messages and error reporting
    - DiagnosticLevel: Enumeration of diagnostic severity levels

The diagnostic system provides rich error messages with source context,
making it easier for users to identify and fix issues in their PTO scripts.
"""

import inspect
import enum
import linecache
import sys
from typing import NoReturn, Union

from . import doc
from .error import ParserError, RenderedParserError

PRIOR_CONTEXT_LINES = 2
SUBSEQUENT_CONTEXT_LINES = 4


class DiagnosticLevel(enum.IntEnum):
    """Severity levels for diagnostic messages, corresponding to diagnostic.h definitions."""

    BUG = 10
    ERROR = 20
    WARNING = 30
    INFO = 40
    DEBUG = 50


class Source:
    """Represents source code for PTO Script parsing.

    Can be instantiated from either a source code string or a doc AST tree object.

    Parameters
    ----------
    source_name : str
        The file path where the source code resides.

    start_line : int
        The initial line number of the source code segment.

    start_column : int
        The initial column position on the first line of the source code.

    source : str
        The actual source code content string.

    full_source : str
        The entire source code content of the file containing this source segment.
    """

    source_name: str
    start_line: int
    start_column: int
    source: str
    full_source: str

    def __init__(self, program: Union[str, doc.AST]):
        if isinstance(program, str):
            self.source_name = "<str>"
            self.start_line = 1
            self.start_column = 0
            self.source = program
            self.full_source = program
            return

        self.source_name = inspect.getsourcefile(program)  # type: ignore
        source_lines, self.start_line = getsourcelines(program)  # type: ignore
        if source_lines:
            self.start_column = len(source_lines[0]) - len(source_lines[0].lstrip())
        else:
            self.start_column = 0
        if self.start_column and source_lines:
            self.source = "\n".join(
                [line_content[self.start_column:].rstrip() for line_content in source_lines]
            )
        else:
            self.source = "".join(source_lines)
        try:
            # Handling Jupyter Notebook compatibility issue.
            # When running in Jupyter, `mod` becomes <module '__main__'>, a built-in module
            # which causes `getsource` to raise a TypeError
            module_obj = inspect.getmodule(program)
            if module_obj:
                self.full_source = inspect.getsource(module_obj)
            else:
                self.full_source = self.source
        except TypeError:
            # Fallback approach for Jupyter compatibility.
            # Using `findsource` as an alternative since it's an internal inspect API
            # that can handle this edge case.
            source_content, _ = inspect.findsource(program)  # type: ignore
            self.full_source = "".join(source_content)

    def as_ast(self) -> doc.AST:
        """Convert the source code string into an AST representation.

        Returns
        -------
        result : doc.AST
            The abstract syntax tree representation of the source code.
        """
        return doc.parse(self.source)


_original_getfile = inspect.getfile  # pylint: disable=invalid-name
_original_findsource = inspect.findsource  # pylint: disable=invalid-name


def _custom_inspect_getfile(target_obj):
    """Determine the source file or compiled file location where an object was defined."""
    if not inspect.isclass(target_obj):
        return _original_getfile(target_obj)
    module_name = getattr(target_obj, "__module__", None)
    if module_name is not None:
        file_path = getattr(sys.modules[module_name], "__file__", None)
        if file_path is not None:
            return file_path
    for _, method_member in inspect.getmembers(target_obj):
        if inspect.isfunction(method_member):
            if (
                target_obj.__qualname__ + "." + method_member.__name__
                == method_member.__qualname__
            ):
                return inspect.getfile(method_member)
    raise TypeError("Source for {!r} not found".format(target_obj))


def findsource(target_obj):
    """Retrieve the complete source file content and the starting line number for an object."""

    if not inspect.isclass(target_obj):
        return _original_findsource(target_obj)

    file_path = inspect.getsourcefile(target_obj)
    if file_path:
        linecache.checkcache(file_path)
    else:
        file_path = inspect.getfile(target_obj)
        if not (file_path.startswith("<") and file_path.endswith(">")):
            raise OSError("source code not available")

    module_obj = inspect.getmodule(target_obj, file_path)
    if module_obj:
        file_lines = linecache.getlines(file_path, module_obj.__dict__)
    else:
        file_lines = linecache.getlines(file_path)
    if not file_lines:
        raise OSError("could not get source code")
    qualified_name_parts = target_obj.__qualname__.replace(".<locals>", "<locals>").split(".")
    comment_state = 0
    nesting_stack = []
    indentation_map = {}
    for line_index, current_line in enumerate(file_lines):
        docstring_count = current_line.count('"""')
        if docstring_count:
            # Toggle comment state based on docstring markers
            comment_state = comment_state ^ (docstring_count & 1)
            continue
        if comment_state:
            # Ignore lines that are inside multi-line docstrings
            continue
        line_indent = len(current_line) - len(current_line.lstrip())
        line_tokens = current_line.split()
        if len(line_tokens) > 1:
            identifier = None
            if line_tokens[0] == "def":
                identifier = (
                    line_tokens[1].split(":")[0].split("(")[0] + "<locals>"
                )
            elif line_tokens[0] == "class":
                identifier = line_tokens[1].split(":")[0].split("(")[0]
            # Remove scopes that are at equal or greater indentation
            while nesting_stack and indentation_map[nesting_stack[-1]] >= line_indent:
                nesting_stack.pop()
            if identifier:
                nesting_stack.append(identifier)
                indentation_map[identifier] = line_indent
                if nesting_stack == qualified_name_parts:
                    return file_lines, line_index

    raise OSError("could not find class definition")


def getsourcelines(target_obj):
    """Extract the code block starting from the top of the provided lines list."""
    unwrapped_obj = inspect.unwrap(target_obj)
    source_lines, line_number = findsource(unwrapped_obj)
    return inspect.getblock(source_lines[line_number:]), line_number + 1


inspect.getfile = _custom_inspect_getfile


class Span:
    """Represents a source code location span for diagnostic messages."""

    source_name: str
    line: int
    end_line: int
    column: int
    end_column: int

    def __init__(
        self,
        source_name: str,
        lineno: int,
        end_lineno: int,
        col_offset: int,
        end_col_offset: int,
    ):
        self.source_name = source_name
        self.line = lineno
        self.end_line = end_lineno
        self.column = col_offset
        self.end_column = end_col_offset


class DiagnosticItem:
    """Represents a single diagnostic message item.

    Parameters
    ----------
    level : DiagnosticLevel
        The severity level of this diagnostic message.
    """

    level: DiagnosticLevel
    span: Span
    message: str

    def __init__(self, level: DiagnosticLevel, span: Span, message: str):
        self.level = level
        self.span = span
        self.message = message

    def render_to_console(self) -> None:
        """Output the diagnostic item to the console with formatting."""
        # ANSI escape sequences for terminal color coding by severity
        color_map = {
            DiagnosticLevel.BUG: "\033[95m",  # Magenta
            DiagnosticLevel.ERROR: "\033[91m",  # Red
            DiagnosticLevel.WARNING: "\033[93m",  # Yellow
            DiagnosticLevel.INFO: "\033[94m",  # Blue
            DiagnosticLevel.DEBUG: "\033[92m",  # Green
        }
        level_labels = {
            DiagnosticLevel.BUG: "INTERNAL BUG",
            DiagnosticLevel.ERROR: "ERROR",
            DiagnosticLevel.WARNING: "WARNING",
            DiagnosticLevel.INFO: "INFO",
            DiagnosticLevel.DEBUG: "DEBUG",
        }
        color_reset = "\033[0m"
        bold_style = "\033[1m"

        selected_color = color_map.get(self.level, "")
        severity_label = level_labels.get(self.level, "")

        # Output formatted header with color coding
        header_parts = [
            bold_style,
            selected_color,
            severity_label,
            color_reset,
            " ",
            bold_style,
            self.span.source_name,
            ":",
            str(self.span.line),
            ":",
            str(self.span.column),
            ":",
            color_reset,
            " ",
            self.message,
        ]
        print("".join(header_parts))

        # Load and present source code context
        with open(self.span.source_name, "r", encoding="utf-8") as source_file:
            file_lines = source_file.readlines()

        preceding_context = PRIOR_CONTEXT_LINES
        following_context = SUBSEQUENT_CONTEXT_LINES
        context_start = max(0, self.span.line - 1 - preceding_context)
        context_end = min(len(file_lines), self.span.end_line + following_context)

        # Determine padding width for line number display
        max_line_digits = len(str(context_end))

        # Display surrounding source lines
        for line_idx in range(context_start, context_end):
            current_line_num = line_idx + 1
            line_text = file_lines[line_idx].rstrip("\n")

            # Determine if this line is part of the error span
            is_span_line = (
                self.span.line <= current_line_num <= self.span.end_line
            )

            if is_span_line:
                formatted_line = (
                    selected_color
                    + str(current_line_num).rjust(max_line_digits)
                    + " |"
                    + color_reset
                    + " "
                    + line_text
                )
                print(formatted_line)
            else:
                line_display = (
                    str(current_line_num).rjust(max_line_digits) + " | " + line_text
                )
                print(line_display)

            # Add visual indicator for error span lines
            if is_span_line:
                # Compute indentation for caret positioning
                if current_line_num == self.span.line:
                    caret_start = self.span.column - 1
                else:
                    caret_start = 0

                if current_line_num == self.span.end_line:
                    caret_end = self.span.end_column - 1
                else:
                    caret_end = len(line_text)

                # Output caret indicator line
                padding_spaces = " " * (max_line_digits + 3 + caret_start)
                caret_chars = "^" * max(1, caret_end - caret_start)
                caret_line = selected_color + padding_spaces + caret_chars + color_reset
                print(caret_line)

        print()  # Blank line for visual separation


class DiagnosticContext:
    """Container for managing diagnostic messages within a source context.

    Parameters
    ----------
    source : Source
        The source code object associated with these diagnostics.
    """

    source: Source
    diagnostics: list[DiagnosticItem]

    def __init__(self, source: Source):
        self.source = source
        self.diagnostics = []

    def emit(self, diagnostic: DiagnosticItem) -> None:
        """Add a diagnostic message to the collection.

        Parameters
        ----------
        diagnostic : DiagnosticItem
            The diagnostic message to add.
        """
        self.diagnostics.append(diagnostic)

    def render(self) -> None:
        """Display all collected diagnostics to the console and clear the list."""
        for diagnostic_item in self.diagnostics:
            diagnostic_item.render_to_console()
        self.diagnostics.clear()


class Diagnostics:
    """Manages diagnostic message generation and reporting for the parser.

    Parameters
    ----------
    source : Source
        The source code being analyzed.

    ctx : DiagnosticContext
        The diagnostic context container for managing messages.
    """

    source: Source
    context: DiagnosticContext

    def __init__(self, source: Source):
        self.source = source
        self.context = DiagnosticContext(source)

    def __del__(self) -> None:
        """Output all diagnostics and perform cleanup."""
        self._render()

    def emit(
        self, node: doc.AST, message: str, level: DiagnosticLevel = DiagnosticLevel.INFO
    ) -> None:
        """Generate and record a diagnostic message.

        Parameters
        ----------
        node : doc.AST
            The AST node containing location information for the diagnostic.

        message : str
            The diagnostic message text.

        level : DiagnosticLevel
            The severity level of the diagnostic.
        """
        line_number = getattr(node, "lineno", 1)
        column_position = getattr(node, "col_offset", self.source.start_column)
        ending_line = getattr(node, "end_lineno", line_number)
        ending_column = getattr(node, "end_col_offset", column_position)
        line_number = line_number + (self.source.start_line - 1)
        ending_line = ending_line + (self.source.start_line - 1)
        column_position = column_position + self.source.start_column + 1
        ending_column = ending_column + self.source.start_column + 1
        self.context.emit(
            DiagnosticItem(
                level=level,
                span=Span(
                    source_name=self.source.source_name,
                    lineno=line_number,
                    end_lineno=ending_line,
                    col_offset=column_position,
                    end_col_offset=ending_column,
                ),
                message=message,
            )
        )

    def bug(self, node: doc.AST, message: str) -> NoReturn:
        """Generate a bug-level diagnostic and raise an exception.

        Parameters
        ----------
        node : doc.AST
            The AST node associated with the bug.

        message : str
            The bug description message.

        Raises
        ------
        ParserError
            Always raises RenderedParserError after displaying the diagnostic.
        """
        self.emit(node, message, DiagnosticLevel.BUG)
        self._render()
        raise RenderedParserError(node, message)

    def error(self, node: doc.AST, message: str) -> NoReturn:
        """Generate an error-level diagnostic and raise an exception.

        Parameters
        ----------
        node : doc.AST
            The AST node associated with the error.

        message : str
            The error description message.

        Raises
        ------
        ParserError
            Always raises RenderedParserError after displaying the diagnostic.
        """
        self.emit(node, message, DiagnosticLevel.ERROR)
        self._render()
        raise RenderedParserError(node, message)

    def warning(self, node: doc.AST, message: str) -> None:
        """Generate a warning-level diagnostic message.

        Parameters
        ----------
        node : doc.AST
            The AST node associated with the warning.
        """
        self.emit(node, message, DiagnosticLevel.WARNING)

    def info(self, node: doc.AST, message: str) -> None:
        """Generate an info-level diagnostic message.

        Parameters
        ----------
        node : doc.AST
            The AST node associated with the informational message.

        message : str
            The informational message text.
        """
        self.emit(node, message, DiagnosticLevel.INFO)

    def debug(self, node: doc.AST, message: str) -> None:
        """Generate a debug-level diagnostic message.

        Parameters
        ----------
        node : doc.AST
            The AST node associated with the debug message.

        message : str
            The debug message text.
        """
        self.emit(node, message, DiagnosticLevel.DEBUG)

    def _render(self) -> None:
        """Output all diagnostics to the console."""
        self.context.render()
