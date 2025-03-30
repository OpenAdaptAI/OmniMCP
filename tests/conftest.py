# tests/conftest.py

"""Pytest configuration for OmniMCP tests."""

import sys
import os

# Add the 'tests' directory to the Python path for imports within tests
TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import pytest  # noqa


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests that may require external resources",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on command line options."""
    if not config.getoption("--run-e2e"):
        skip_e2e = pytest.mark.skip(reason="Need --run-e2e option to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)
