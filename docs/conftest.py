"""
Conftest for notebook integration tests (docs/ directory).

Notebooks are only executed when MARS_MF_TEST_DATA=1 is set.
This avoids running them in CI or environments without the large data files.
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip all notebook tests unless MARS_MF_TEST_DATA env var is set."""
    if os.environ.get("MARS_MF_TEST_DATA") == "1":
        return

    skip_marker = pytest.mark.skip(reason="MARS_MF_TEST_DATA env var not set")
    for item in items:
        if item.fspath.ext == ".ipynb":
            item.add_marker(skip_marker)
