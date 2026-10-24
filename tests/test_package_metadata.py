"""Tests for the public package metadata contract."""

from __future__ import annotations

from importlib.metadata import version

import kiln


def test_distribution_version_matches_package_version() -> None:
    assert version("kiln") == kiln.__version__
