"""
Pytest configuration for marshsi tests.

This file contains pytest hooks and fixtures shared across all test modules.
"""

from unittest.mock import MagicMock

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon


# ── Shared infrastructure fixtures ──────────────────────────────────────


@pytest.fixture
def mock_logger():
    """Mock logger that supports standard logging interface."""
    logger = MagicMock()
    return logger


# ── Shared geometry fixtures ────────────────────────────────────────────


@pytest.fixture
def sample_polygon():
    """A 1x1 degree polygon centered on (-121.5, 37.5)."""
    return Polygon([(-122, 37), (-122, 38), (-121, 38), (-121, 37)])


@pytest.fixture
def sample_multipolygon(sample_polygon):
    return MultiPolygon([sample_polygon])


@pytest.fixture
def sample_point():
    return Point(-121.5, 37.5)
