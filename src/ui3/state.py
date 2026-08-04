"""ui3 application state + persistence.

Reuses the shared, typed reconstruction params (``core.drivers.reconstruction.ReconParams``
— Qt-free data with every field incl. the reference/reffree knobs) so ui3 and the
compute drivers never drift. UI-shell state (theme, recent files, window geometry,
last sample id, workflow mode) rides in :class:`Ui3State`, persisted as JSON under
``~/.dhm-reconstruction/ui3_state.json`` — independent of the ui2/Qt-v1 stores.

Qt-free: this module is plain dataclasses + json so state logic is unit-testable
without a QApplication.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

from core.drivers.reconstruction import ReconParams  # shared Qt-free params

STATE_VERSION = 1


def default_state_path() -> Path:
    root = Path(os.path.expanduser("~")) / ".dhm-reconstruction"
    return root / "ui3_state.json"


@dataclass
class Ui3State:
    """Everything the ui3 shell needs to feel continuous across launches."""
    version: int = STATE_VERSION
    theme: str = "dark"
    workflow_mode: str = "Reconstruct"
    sample_id: str = ""
    selected_preset: str = "Custom"
    recent: List[str] = field(default_factory=list)
    last_dir: str = ""
    last_hologram: str = ""
    last_reference: str = ""
    window_geometry_b64: str = ""     # QMainWindow.saveGeometry() base64
    window_state_b64: str = ""        # QMainWindow.saveState() base64
    # Dock-layout schema version the window_state_b64 was saved under. A saved
    # QMainWindow.saveState() only restores cleanly against the SAME set of
    # docks; restoring an older layout (docks added/removed/re-wrapped since)
    # leaves mismatched docks FLOATING over the central grid (2026-07-10 real-
    # screen bug). The shell only restores the layout when this matches its
    # current _LAYOUT_VERSION; otherwise it uses the clean programmatic layout.
    layout_version: int = 0
    onboarding_seen: bool = False
    # Live reconstruction params (flat dict of ReconParams for portability).
    params: Dict[str, Any] = field(default_factory=dict)

    # -- params helpers -------------------------------------------------
    def recon_params(self) -> ReconParams:
        """Build a ReconParams from the stored dict, tolerating unknown /
        missing keys (forward+backward compatible)."""
        valid = {f.name for f in fields(ReconParams)}
        clean = {k: v for k, v in (self.params or {}).items() if k in valid}
        # JSON has no tuple type: coerce list-valued tuple fields back so the
        # runtime type matches ReconParams' contract (af_roi is unpacked as a
        # 4-tuple downstream). See _TUPLE_PARAM_FIELDS.
        for key in _TUPLE_PARAM_FIELDS:
            v = clean.get(key)
            if isinstance(v, list):
                clean[key] = tuple(v)
        # Path fields round-trip through str — rebuild the Path (empty → None
        # so an unset reference doesn't become Path('.')). See _PATH_PARAM_FIELDS.
        for key in _PATH_PARAM_FIELDS:
            v = clean.get(key)
            if isinstance(v, str):
                clean[key] = Path(v) if v else None
        try:
            return ReconParams(**clean)
        except Exception:
            return ReconParams()

    def set_recon_params(self, p: ReconParams) -> None:
        self.params = _params_to_dict(p)

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ui3State":
        valid = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in valid}
        st = cls(**clean)
        st.recent = [str(r) for r in (st.recent or []) if r][:12]
        return st


# ReconParams fields that are logically tuples (round-tripped through JSON,
# which has no tuple type, so they must be coerced back on load).
_TUPLE_PARAM_FIELDS = ("af_roi", "offaxis_center")
# Path-valued fields: JSON has no Path type, so persist as str and coerce
# back on load. Blanket-skipping them (they were "non-serialisable") dropped
# reference_path while STILL persisting reference_mode="reference", so a
# relaunch re-armed reference mode with no path → a silently unreferenced
# reconstruction presented as reference-corrected (2026-07-08 review).
_PATH_PARAM_FIELDS = ("reference_path",)


def _params_to_dict(p: ReconParams) -> Dict[str, Any]:
    """Serialise ReconParams to a JSON-native dict.

    Scalars pass through; small tuples/lists of scalars (e.g. ``af_roi``) are
    stored as JSON lists; Path fields (``reference_path``) as their string
    form. Genuinely non-serialisable values (numpy arrays) are *skipped*
    rather than ``str()``-ified — a stringified tuple used to reload as the
    literal ``"(10, 20, ...)"``, silently corrupting the field (2026-07-06
    review). Skipped fields fall back to their ReconParams default on load.
    """
    out: Dict[str, Any] = {}
    for f in fields(ReconParams):
        val = getattr(p, f.name, None)
        if val is None or isinstance(val, (bool, int, float, str)):
            out[f.name] = val
        elif f.name in _PATH_PARAM_FIELDS:
            out[f.name] = str(val)
        elif isinstance(val, (tuple, list)) and all(
                isinstance(x, (bool, int, float, str)) for x in val):
            out[f.name] = list(val)
        # else: non-serialisable (ndarray) — skip; default on reload.
    return out


def load_state(path: Path | None = None) -> Ui3State:
    """Load state, returning defaults on any error (corrupt/missing file)."""
    p = path or default_state_path()
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return Ui3State.from_dict(data)
    except (OSError, ValueError, json.JSONDecodeError):
        return Ui3State()


def save_state(state: Ui3State, path: Path | None = None) -> None:
    """Atomically persist state (tempfile + os.replace) — never raises."""
    p = path or default_state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except OSError:
        pass  # persistence must never crash the app


def push_recent(state: Ui3State, path: str, cap: int = 12) -> None:
    """Move ``path`` to the front of the recent list (dedup, capped)."""
    path = str(path)
    state.recent = [path] + [r for r in state.recent if r != path]
    del state.recent[cap:]
