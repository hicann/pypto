# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the real frontend pytest loop with CPU-only stand-ins for JIT/launch."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

_FAKE_JIT = """
import sys
from pathlib import Path
from types import ModuleType


def record(event):
    with Path("events.txt").open("a") as stream:
        stream.write(event + "\\n")


class CompiledKernel:
    def __init__(self, **kwargs):
        pass


class Kernel:
    compiled = None

    def _compile_variant(self, *args):
        record("compile")
        self.compiled = CompiledKernel()
        return self.compiled

    def __call__(self):
        compiled = self.compiled or self._compile_variant(None, (None, None, None, 0), (), ())
        return jit._launch(compiled)


jit = ModuleType("pypto_pro.runtime.jit")
jit._TileJitKernel = Kernel
jit.CompiledKernel = CompiledKernel
jit._launch = lambda compiled: 37
jit.record = record
jit.kernel = Kernel()
for name in ("pypto_pro", "pypto_pro.runtime"):
    package = ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
sys.modules[jit.__name__] = jit
"""


def _run_suite(tmp_path, monkeypatch, source, *, parallel=True, args=()):
    conftest = Path(__file__).resolve().parents[3] / "st/pypto_pro/frontend/conftest.py"
    (tmp_path / "conftest.py").write_text(conftest.read_text() + _FAKE_JIT)
    (tmp_path / "test_sample.py").write_text(textwrap.dedent(source))
    monkeypatch.setenv("PARALLEL_COMPILE", "1" if parallel else "0")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_sample.py", "-q", *args],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    events = tmp_path / "events.txt"
    return result, events.read_text().splitlines() if events.exists() else []


@pytest.mark.parametrize("parallel", [False, True])
def test_resource_fixture_runs_once_and_kernel_compiles_once(tmp_path, monkeypatch, parallel):
    result, events = _run_suite(tmp_path, monkeypatch, """
        import pytest
        from pypto_pro.runtime.jit import kernel, record

        @pytest.fixture
        def resource():
            record("setup")
            yield
            record("teardown")

        def test_kernel():
            record("kernel_body")
            assert kernel() == 37

        @pytest.mark.skip_jit_discovery(reason="external resource")
        def test_resource(resource):
            record("resource_body")
    """, parallel=parallel)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert events.count("kernel_body") == (2 if parallel else 1)
    assert events.count("compile") == 1
    assert events.count("setup") == events.count("teardown") == events.count("resource_body") == 1


def test_module_marker_excludes_only_discovery(tmp_path, monkeypatch):
    result, events = _run_suite(tmp_path, monkeypatch, """
        import pytest
        from pypto_pro.runtime.jit import record

        pytestmark = pytest.mark.skip_jit_discovery(reason="subprocess suite")

        def test_external():
            record("external")
    """)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "WARNING -- discovered no kernels" not in result.stdout
    assert events == ["external"]


def test_collection_never_runs_test_bodies(tmp_path, monkeypatch):
    result, events = _run_suite(tmp_path, monkeypatch, """
        from pypto_pro.runtime.jit import record

        def test_collect():
            record("unexpected")
    """, args=("--collect-only",))
    assert result.returncode == 0, result.stdout + result.stderr
    assert not events


def test_execution_failure_keeps_normal_traceback(tmp_path, monkeypatch):
    result, events = _run_suite(tmp_path, monkeypatch, """
        from pypto_pro.runtime.jit import record

        def test_real_failure():
            record("body")
            actual_value = 4
            assert actual_value == 17
    """)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed" in result.stdout
    assert ">       assert actual_value == 17" in result.stdout
    assert events == ["body", "body"]
