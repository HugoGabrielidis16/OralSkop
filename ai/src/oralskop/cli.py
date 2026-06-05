"""Console-script entry points (see pyproject [project.scripts])."""

from __future__ import annotations


def prepare_main() -> None:
    from oralskop.data.prepare import main

    main()
