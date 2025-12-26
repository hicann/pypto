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

"""PTO Script Parser doc AST - Bidirectional AST Conversion System.

This module provides a registry system for converting between Python's standard
AST nodes and the custom "doc AST" nodes used by the PTO frontend parser. The
doc AST provides a stable interface that isolates the parser from Python version
changes while maintaining full compatibility with Python's AST structure.

Key Features
------------
- **Bidirectional Conversion**: Convert between Python AST and doc AST in both directions
- **Registration System**: Extensible decorator-based registration for new node types
- **Version Independence**: Stable interface across Python versions (3.9+)
- **Visitor Patterns**: Support for NodeVisitor and NodeTransformer patterns
- **Automatic Conversion**: Transparent conversion during parsing and code generation

Architecture
------------
The conversion system uses a singleton Registry that maps Python AST node type
names to Entry objects. Each Entry contains two conversion functions:
- `to_doc`: Converts Python AST node → doc AST node
- `from_doc`: Converts doc AST node → Python AST node

Main Components
---------------
- `Entry`: Stores conversion functions for a single node type
- `Registry`: Singleton registry mapping node types to conversion functions
- `register_to_doc`: Decorator to register Python → doc conversion
- `register_from_doc`: Decorator to register doc → Python conversion
- `to_doc()`: Convert Python AST to doc AST
- `from_doc()`: Convert doc AST to Python AST
- `parse()`: Parse source string directly to doc AST

Usage Example
-------------
```python
from pypto.frontend.parser import doc

# Parse Python source to doc AST
source = "def foo(x): return x + 1"
ast_node = doc.parse(source)

# Convert Python AST to doc AST
import ast
py_ast = ast.parse(source)
doc_ast = doc.to_doc(py_ast)

# Convert doc AST back to Python AST
py_ast_again = doc.from_doc(doc_ast)
```

Extending the System
--------------------
To add support for a new AST node type:

```python
@doc.register_to_doc("MyNewNode")
def convert_to_doc(node: ast.MyNewNode) -> doc.MyNewNode:
    return doc.MyNewNode(
        field1=doc.to_doc(node.field1),
        field2=node.field2,
    )

@doc.register_from_doc("MyNewNode")
def convert_from_doc(node: doc.MyNewNode) -> ast.MyNewNode:
    return ast.MyNewNode(
        field1=doc.from_doc(node.field1),
        field2=node.field2,
    )
```
"""


import ast
import inspect
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional, Union

from . import doc_core as doc
from .doc_core import *  # pylint: disable=unused-wildcard-import,wildcard-import

FnToDoc = Callable[[ast.AST], doc.AST]
FnFromDoc = Callable[[doc.AST], ast.AST]


class Entry:
    """Mapping entry between Python AST node type string and doc AST.

    This class stores the bidirectional conversion functions for a specific
    AST node type, allowing transformation between Python's standard AST
    representation and the custom doc AST representation.

    Attributes
    ----------
    to_doc : Optional[FnToDoc]
        Callable that converts a Python AST node to a doc AST node.
        None if no conversion is registered.

    from_doc : Optional[FnFromDoc]
        Callable that converts a doc AST node to a Python AST node.
        None if no conversion is registered.
    """

    to_doc: Optional[FnToDoc]
    from_doc: Optional[FnFromDoc]

    def __init__(self) -> None:
        """Initialize an Entry with no conversion functions registered."""
        self.to_doc = None
        self.from_doc = None


class Registry:
    """Registration map from Python AST node type string to conversion methods.

    This singleton class maintains a registry of conversion functions between
    Python AST nodes and doc AST nodes. It uses a table to map node type names
    to their corresponding Entry objects containing conversion functions.

    Attributes
    ----------
    _inst : Optional[Registry]
        The singleton instance of the Registry class. None until initialized.

    table : dict[str, Entry]
        Mapping from Python AST node type name to Entry containing conversion
        functions. Uses defaultdict to automatically create Entry objects.
    """

    _inst: Optional["Registry"] = None
    table: dict[str, Entry]

    def __init__(self) -> None:
        """Initialize the Registry with an empty table using defaultdict."""
        self.table = defaultdict(Entry)


def register_to_doc(name: str) -> Callable[[FnToDoc], FnToDoc]:
    """Register a conversion function from Python AST to doc AST for a node type.

    This decorator registers a function that converts a specific Python AST node
    type to its corresponding doc AST representation.

    Parameters
    ----------
    name : str
        The Python AST node type name (e.g., 'FunctionDef', 'Assign').

    Returns
    -------
    Callable[[FnToDoc], FnToDoc]
        A decorator function that registers the conversion function and returns it.

    Examples
    --------
    >>> @register_to_doc("FunctionDef")
    ... def convert_func_def(node):
    ...     return doc.FunctionDef(...)
    """

    def f(to_doc_fn: FnToDoc) -> FnToDoc:
        reg = Registry._inst  # pylint: disable=protected-access
        reg.table[name].to_doc = to_doc_fn
        return to_doc_fn

    return f


def register_from_doc(name: str) -> Callable[[FnFromDoc], FnFromDoc]:
    """Register a conversion function from doc AST to Python AST for a node type.

    This decorator registers a function that converts a specific doc AST node
    type to its corresponding Python AST representation.

    Parameters
    ----------
    name : str
        The Python AST node type name (e.g., 'FunctionDef', 'Assign').

    Returns
    -------
    Callable[[FnFromDoc], FnFromDoc]
        A decorator function that registers the conversion function and returns it.

    Examples
    --------
    >>> @register_from_doc("FunctionDef")
    ... def convert_from_func_def(node):
    ...     return ast.FunctionDef(...)
    """

    def f(from_doc_fn: FnFromDoc) -> FnFromDoc:
        reg = Registry._inst  # pylint: disable=protected-access
        reg.table[name].from_doc = from_doc_fn
        return from_doc_fn

    return f


def _is_atomic_type(node: Any) -> bool:
    """Check if a node is an atomic (primitive) type that doesn't need conversion.

    Atomic types include None, Ellipsis, booleans, and basic Python primitives
    like int, float, str, bytes, and complex numbers.

    Parameters
    ----------
    node : Any
        The node to check.

    Returns
    -------
    bool
        True if the node is an atomic type, False otherwise.
    """
    return (
        node is None
        or node in [..., True, False]
        or isinstance(node, (int, float, str, bool, bytes, complex))
    )


def _get_registry_entry(
    clsname: str, attr_name: str
) -> Optional[Union[FnToDoc, FnFromDoc]]:
    """Retrieve a conversion function from the registry for a given class.

    Parameters
    ----------
    cls_name : str
        The fully qualified or simple class name of the AST node.
        Will be normalized to simple name by taking the last component.
    attr : str
        The attribute name to retrieve ('to_doc' or 'from_doc').

    Returns
    -------
    Optional[Union[FnToDoc, FnFromDoc]]
        The requested conversion function if registered, None otherwise.
    """
    clsname = clsname.split(".")[-1]
    reg = Registry._inst  # pylint: disable=protected-access
    if clsname in reg.table:
        entry = reg.table[clsname]
        return getattr(entry, attr_name, None)
    return None


def from_doc(node: Any) -> Any:
    """Convert a doc AST node to its corresponding Python AST node.

    This function recursively converts doc AST nodes to Python AST nodes,
    handling atomic types, tuples, lists, and custom doc AST nodes.

    Parameters
    ----------
    node : Any
        The doc AST node to convert. Can be atomic types, tuples, lists,
        or doc.AST instances.

    Returns
    -------
    Any
        The corresponding Python AST node.

    Raises
    ------
    NotImplementedError
        If no conversion function is registered for the node type.

    Examples
    --------
    >>> doc_node = doc.FunctionDef(...)
    >>> ast_node = from_doc(doc_node)
    >>> isinstance(ast_node, ast.FunctionDef)
    True
    """
    if _is_atomic_type(node):
        return node
    if isinstance(node, tuple):
        return tuple(from_doc(item) for item in node)
    if isinstance(node, list):
        return [from_doc(item) for item in node]
    function_from_doc = _get_registry_entry(node.__class__.__name__, "from_doc")
    if not function_from_doc:
        raise NotImplementedError(
            f"from_doc is not implemented for node type: {node.__class__.__name__}"
        )
    return function_from_doc(node)


def to_doc(node: Any) -> Any:
    """Convert a Python AST node to its corresponding doc AST node.

    This function recursively converts Python AST nodes to doc AST nodes,
    handling atomic types, tuples, lists, and standard AST nodes.

    Parameters
    ----------
    node : Any
        The Python AST node to convert. Can be atomic types, tuples, lists,
        or ast.AST instances.

    Returns
    -------
    Any
        The corresponding doc AST node.

    Raises
    ------
    NotImplementedError
        If no conversion function is registered for the node type.

    Examples
    --------
    >>> ast_node = ast.FunctionDef(...)
    >>> doc_node = to_doc(ast_node)
    >>> isinstance(doc_node, doc.FunctionDef)
    True
    """
    if _is_atomic_type(node):
        return node
    # Handle tuple
    if isinstance(node, tuple):
        return tuple(to_doc(item) for item in node)
    # Handle list
    if isinstance(node, list):
        return [to_doc(item) for item in node]
    # Get the conversion function for the node type
    function_to_doc = _get_registry_entry(node.__class__.__name__, "to_doc")
    if not function_to_doc:
        raise NotImplementedError(
            f"to_doc is not implemented for node type: {node.__class__.__name__}"
        )
    return function_to_doc(node)


def parse(
    source: str,
    filename: str = "<unknown>",
    mode: str = "exec",
) -> doc.AST:
    """Parse Python source code string to doc AST.

    This function parses Python source code into a doc AST representation.
    It attempts to parse using Python 3.8 feature version for consistency,
    falling back to the current Python version if that fails.

    The interface is consistent with Python's built-in ast.parse function.

    Parameters
    ----------
    source : str
        The Python source code to parse.

    filename : str, default="<unknown>"
        Optional filename indicating where the source code is from.
        Used for error messages and debugging.

    mode : str, default="exec"
        The parsing mode. Can be:
        - "exec": Parse file input (multiple statements)
        - "eval": Parse a single expression
        - "single": Parse a single interactive statement

    Returns
    -------
    doc.AST
        The parsed doc AST representation of the source code.

    Examples
    --------
    >>> source = "def foo(): pass"
    >>> tree = parse(source)
    >>> isinstance(tree, doc.Module)
    True
    """
    program = ast.parse(source=source, filename=filename, mode=mode)
    return to_doc(program)


class NodeVisitor:
    """Node visitor for doc AST using the visitor pattern.

    This class implements the visitor pattern for traversing doc AST trees.
    Subclasses can override specific visit_* methods to customize behavior
    for particular node types.

    The visitor pattern allows you to perform operations on AST nodes without
    modifying their classes. Each node type can have a corresponding visit_*
    method, with a fallback to generic_visit.

    Examples
    --------
    >>> class MyVisitor(NodeVisitor):
    ...     def visit_FunctionDef(self, node):
    ...         print(f"Found function: {node.name}")
    ...         self.generic_visit(node)
    >>> visitor = MyVisitor()
    >>> visitor.visit(tree)
    """

    def visit(self, node: doc.AST) -> None:
        """Visit a node and dispatch to the appropriate visitor method.

        Parameters
        ----------
        node : doc.AST
            The node to visit. Can be a single node, list, or tuple of nodes.
        """
        if isinstance(node, (list, tuple)):
            for item in node:
                self.visit(item)
            return
        assert isinstance(node, doc.AST), f"Expected doc.AST, got {type(node)}"

        # Convert CamelCase to snake_case for method names
        class_name = node.__class__.__name__.split(".")[-1]
        snake_case_name = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        method = getattr(self, f"visit_{snake_case_name}", self.generic_visit)
        method(node)

    def generic_visit(self, node: doc.AST) -> None:
        """Generic visit method that visits all child nodes.

        This is called when no specific visit_* method exists for a node type.
        It recursively visits all fields of the node.

        Parameters
        ----------
        node : doc.AST
            The node whose children should be visited.
        """
        for field in get_cls_fields(node.__class__):
            value = getattr(node, field, None)
            if value is None:
                pass
            elif isinstance(value, (doc.AST, list, tuple)):
                self.visit(value)


def get_cls_fields(cls: type) -> set[str]:
    """Get the fields of a class.

    Parameters
    ----------
    cls : type
        The class to get the fields of.

    Returns
    -------
    set[str]
        The fields of the class.
    """
    fields = set()
    for base in cls.__mro__:
        if hasattr(base, "__annotations__"):
            fields.update(base.__annotations__.keys())
    return fields


def _register_default() -> None:
    """Register default conversion functions for all standard AST node types.

    This function initializes the Registry singleton and automatically registers
    bidirectional conversion functions for all AST node types that exist in both
    the doc module and Python's standard ast module.

    The default translators handle the common case where fields can be directly
    mapped between Python AST and doc AST nodes.
    """

    class DefaultTranslator:
        """Default translator for standard AST node type conversions.

        Attributes
        ----------
        doc_cls : type
            The doc AST class to convert to/from.
        func : Callable
            The conversion function (to_doc or from_doc).
        fields : tuple[str, ...]
            The field names to convert.
        """

        def __init__(
            self,
            doc_cls: type,
            func: Callable[[Any], Any],
            fields: tuple[str, ...],
        ) -> None:
            """Initialize the default translator.

            Parameters
            ----------
            doc_cls : type
                The target class for conversion.
            func : Callable[[Any], Any]
                The conversion function to apply to fields.
            fields : tuple[str, ...]
                The field names to extract and convert.
            """
            self.doc_cls = doc_cls  # getattr(doc, name)
            self.func = func
            self.fields = fields

        def __call__(self, node: Any) -> Any:
            """Convert a node by applying the conversion function to all fields.

            Parameters
            ----------
            node : Any
                The node to convert.

            Returns
            -------
            Any
                A new instance of doc_cls with converted fields.
            """
            kv = {attr: self.func(getattr(node, attr, None)) for attr in self.fields}
            return self.doc_cls(**kv)

    Registry._inst = Registry()  # pylint: disable=protected-access
    
    # Register default conversion functions for all standard AST node types
    for cls_name_ in dir(doc):
        doccls = getattr(doc, cls_name_)
        # Skip if the class is not in the ast module
        if not hasattr(ast, cls_name_):
            continue
        # Skip if the class is not a subclass of doc.AST
        if inspect.isclass(doccls) and issubclass(doccls, doc.AST):
            # Skip if the class name contains a dot
            assert "." not in cls_name_
            # Collect annotations from parent classes as well
            cls_fields_names = get_cls_fields(doccls)
            register_to_doc(cls_name_)(
                DefaultTranslator(getattr(doc, cls_name_), to_doc, cls_fields_names)
            )
            register_from_doc(cls_name_)(
                DefaultTranslator(getattr(ast, cls_name_), from_doc, cls_fields_names)
            )


_register_default()
