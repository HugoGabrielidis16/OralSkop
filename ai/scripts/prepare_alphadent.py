#!/usr/bin/env python3
"""Thin runner for the AlphaDent prepare step.

Usage (from ai/):
    uv run python scripts/prepare_alphadent.py
"""

from oralskop.data.prepare import main

if __name__ == "__main__":
    main()
