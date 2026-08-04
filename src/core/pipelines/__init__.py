"""Shared, framework-free DHM pipelines (reference-free reconstruction, ...).

Core-level so both frontends, the CLI scripts, and the DL inference module
import ONE implementation instead of copying the demod/propagate/unwrap
chain (2026-07-05 review: it had drifted into 3-4 copies, and
``src/recon_dl`` imported it from ``scripts/`` — a layering inversion).
"""
