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
"""The author-facing typed operator-attribute declaration: :class:`AttrSpec`.

A frozen, JSON-serializable record of one declared operator attribute: ``name``, ``type``, ``default``.
Construction validates the name and the default, so a malformed attr is rejected where the author wrote
it rather than at code generation.

``ATTR_NAME_RE`` and ``ATTR_TYPES`` are public: the build-time code generator re-checks both at its own
boundary.
"""
from dataclasses import dataclass
import math
import re

__all__ = ("AttrSpec", "ATTR_NAME_RE", "ATTR_TYPES")


# The typed attribute kinds an op may declare. The token is used verbatim as the GE REG_OP type token.
ATTR_TYPES = ("Int", "Float", "String", "ListInt")

# An attr name is emitted verbatim as a C++ identifier in the generated REG_OP block and is also the ONNX
# node attribute key, so it has to be an identifier in both.
ATTR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class AttrSpec:
    """One declared compute attribute of an operator: ordered, typed, ``name``/``type``/``default``.

    ``type`` is one of :data:`ATTR_TYPES`.

    ``default=None`` means the attribute is REQUIRED, not "defaults to None". A non-None default is a
    GE-side emission default only; it never becomes a Python call default.
    """

    name: str
    type: str
    default: object = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"AttrSpec.name must be a non-empty string, got {self.name!r}")
        if not ATTR_NAME_RE.match(self.name):
            raise ValueError(
                f"AttrSpec.name must be an identifier matching {ATTR_NAME_RE.pattern} -- it is emitted "
                f"verbatim into the generated C++ and used as the ONNX attribute key, got {self.name!r}"
            )
        if self.type not in ATTR_TYPES:
            raise ValueError(
                f"AttrSpec(name={self.name!r}).type must be one of {ATTR_TYPES}, got {self.type!r}"
            )
        # Exact-type checks (``type(x) is int``) rather than isinstance: bool is an int subclass, and a
        # wrongly-typed default would be silently coerced by the emitters rather than rejected.
        if self.type == "Int" and self.default is not None and type(self.default) is not int:
            raise ValueError(
                f"AttrSpec(name={self.name!r}) Int default must be an int (bool excluded; a float would be "
                f"truncated when emitted as a C++ integer literal), got {self.default!r}"
            )
        # The Int default is emitted verbatim into ``.ATTR(name, Int, <literal>)``; a value wider than int64
        # is not representable there (10**20 would become 7766279631452241920). Reject at construction.
        if self.type == "Int" and self.default is not None and not -2**63 <= self.default <= 2**63 - 1:
            raise ValueError(
                f"AttrSpec(name={self.name!r}) Int default must fit in int64 [-2**63, 2**63 - 1] (a wider "
                f"value is silently truncated by the compiler when emitted as a C++ integer literal), got "
                f"{self.default!r}"
            )
        # A non-finite Float default has no valid C++ literal form: repr() yields "inf"/"nan", which the
        # generated .ATTR(...) default would carry verbatim. Reject at construction.
        if self.type == "Float" and self.default is not None:
            if type(self.default) not in (int, float) or not math.isfinite(self.default):
                raise ValueError(
                    f"AttrSpec(name={self.name!r}) Float default must be a finite int or float (it is "
                    f"emitted as a C++ float literal); inf/-inf/nan, bool or a non-number are invalid, "
                    f"got {self.default!r}"
                )
        if self.type == "String" and self.default is not None:
            if not isinstance(self.default, str):
                raise ValueError(
                    f"AttrSpec(name={self.name!r}) String default must be a str (any other value would be "
                    f"stringified into the emitted C++ string literal), got {self.default!r}"
                )
            # The default is emitted verbatim inside a C++ string literal, so reject anything that would
            # break or silently shorten it. NUL is the dangerous one: it ends the emitted C string early,
            # so the generated operator would carry a truncated default with nothing to flag it.
            if any(c in self.default for c in ('"', "\\", "\n", "\r", "\t", "\v", "\f", "\0", "\x1b")):
                raise ValueError(
                    f"AttrSpec(name={self.name!r}) String default may not contain a quote, backslash, or "
                    f"control character (it is emitted as a C++ string literal), got {self.default!r}"
                )
        # A ListInt default is emitted as a C++ brace-init list of integer literals, element by element, so
        # a scalar is not iterable there and a non-int element would be coerced the same way an Int is.
        if self.type == "ListInt" and self.default is not None:
            if not isinstance(self.default, (list, tuple)) or not all(type(v) is int for v in self.default):
                raise ValueError(
                    f"AttrSpec(name={self.name!r}) ListInt default must be a list or tuple of ints "
                    f"(bool excluded; it is emitted as a C++ brace-init list of integer literals), "
                    f"got {self.default!r}"
                )

    def to_json_dict(self) -> dict:
        """Serialize to the plain ``{"name","type","default"}`` dict stored in ``op_export_record``."""
        return {"name": self.name, "type": self.type, "default": self.default}
