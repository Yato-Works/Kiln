"""Enables ``python -m kiln ...`` as an alias for the ``kiln`` console script."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
