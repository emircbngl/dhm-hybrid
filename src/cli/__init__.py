"""DHM headless command-line interface — v2.0.7 sprint, T1.

Sven's production gate (Lindqvist 2026-04-27): "we have a Linux GPU
box, we want to run 3000 holograms overnight without launching Dear
PyGui." This package is the answer.

Entry points:

* ``python -m cli.run_session run <manifest.json> --out <dir>``
* ``python -m cli.run_session inspect <manifest.json>``

The CLI imports the same core pipeline functions the GUI uses
(``offaxis``, ``autofocus``, ``reconstruction``) so headless and
interactive paths can never diverge.
"""
__all__: list[str] = []
