import logging
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def manager():
    with patch("pypto._build_online.BuildOnlinePyptoImplManager.__init__", lambda self: None):
        from pypto._build_online import BuildOnlinePyptoImplManager
        mgr = BuildOnlinePyptoImplManager()
        mgr._target_compiled = False
        mgr.pkg_lib_dir = Path("/nonexistent/pkg/lib")
        mgr.pkg_dir = Path("/nonexistent/pkg")
        mgr._lock_name = ".test.lock"
        mgr._compile_lock = __import__("threading").Lock()
        yield mgr


def test_src_dir_not_exists_logs_warning_and_returns(manager, caplog):
    with caplog.at_level(logging.WARNING, logger="pypto._build_online"):
        with patch.object(manager, "_find_pypto_impl_so", return_value=(False, None)):
            manager._ensure_pypto_impl_locked()

    assert not manager._target_compiled
    assert any("Can't get pypto_impl" in record.message for record in caplog.records)
