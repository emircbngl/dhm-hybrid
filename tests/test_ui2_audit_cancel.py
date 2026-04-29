"""Compliance + cancellation tests for the v2 pipeline (v2.0.4-A).

CEO complaints:
* v2 didn't emit audit events — lab traceability broken.
* Esc cleared the status line but kept the worker running.

These tests pin both:
* Every ScienceDriver job path calls :func:`core.audit.get_audit_log`
  with the right action + params (including sample_id).
* ``cancel()`` actually makes the core scan abort via the
  ``cancel_check`` parameter we threaded through in v2.0.4.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Fixtures — capture audit writes via a tmp directory monkeypatch
# ---------------------------------------------------------------------------

@pytest.fixture
def captured_audit(monkeypatch, tmp_path):
    """Redirect ``core.audit._AUDIT_DIR`` to tmp_path and reset the
    singleton so each test sees its own log file."""
    from core import audit as audit_mod
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    monkeypatch.setattr(audit_mod, "_AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit_mod, "_SINGLETON", None)
    return audit_dir


def _read_audit_records(audit_dir: Path) -> list[dict]:
    import json
    records = []
    for log_file in audit_dir.glob("*.jsonl"):
        for line in log_file.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _wait(predicate, timeout_s=3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# Audit — every science job writes a record
# ---------------------------------------------------------------------------

def test_reconstruction_emits_audit_record(captured_audit, tmp_path):
    """Running a single reconstruction must leave a ``reconstruction``
    action in today's audit log with wavelength + path + sample_id."""
    from ui2.reconstruction import ReconParams, ReconstructionDriver

    # Feed the driver a trivial synthetic hologram so the real
    # propagate() call goes through without error.
    size = 64
    y, x = np.mgrid[:size, :size].astype(np.float32)
    hol = (0.5 + 0.3 * np.cos(2 * np.pi * (0.15 * x + 0.0 * y))).astype(np.float32)
    hol_path = tmp_path / "mini.tif"
    import tifffile
    tifffile.imwrite(hol_path, hol)

    driver = ReconstructionDriver()
    done = [False]
    results: dict[str, object] = {}

    def _ok(r): results["result"] = r; done[0] = True
    def _err(e): results["error"] = e; done[0] = True

    driver.submit(
        hol_path, ReconParams(mask_radius=8),
        sample_id="LIMS-001",
        on_result=_ok, on_error=_err,
    )
    assert _wait(lambda: done[0], timeout_s=5.0), "recon never completed"
    driver.shutdown()

    records = _read_audit_records(captured_audit)
    assert records, "no audit records written"
    recon_rec = next(
        (r for r in records if r["action"] == "reconstruction"), None,
    )
    assert recon_rec is not None, (
        f"no 'reconstruction' action in: {[r['action'] for r in records]}")
    assert recon_rec["params"]["sample_id"] == "LIMS-001"
    assert recon_rec["params"]["method"] == "ASM"
    assert "effective_pixel_um" in recon_rec["params"]
    assert "runtime_ms" in recon_rec["result_summary"]


def test_autofocus_audit_captures_z_range_and_metric(captured_audit):
    """The science worker's audit record for autofocus must include
    the z range + step count + metric used — enough to reproduce."""
    from core.autofocus import FocusMetric
    from ui2.reconstruction import ReconParams
    from ui2.workers import ScienceDriver

    driver = ScienceDriver()
    done = [False]

    fake_loaded = MagicMock()
    fake_loaded.array = np.zeros((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    fake_result = MagicMock()
    fake_result.best_z_m = 0.012
    fake_result.scores = {0.012: 0.9}

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.zeros((16, 16), dtype=np.complex64),
                             (8, 8))), \
         patch("ui2.workers.autofocus_zscan", return_value=fake_result):
        driver.run_autofocus(
            Path("/dev/null"),
            ReconParams(autofocus_metric="TENENGRAD"),
            z_min_mm=-2.0, z_max_mm=5.0, n_steps=10,
            sample_id="LIMS-42",
            on_result=lambda r: done.__setitem__(0, True),
            on_error=lambda e: done.__setitem__(0, True),
        )
        assert _wait(lambda: done[0])
    driver.shutdown()

    records = _read_audit_records(captured_audit)
    af = next((r for r in records if r["action"] == "autofocus"), None)
    assert af is not None
    assert af["params"]["z_min_mm"] == pytest.approx(-2.0)
    assert af["params"]["z_max_mm"] == pytest.approx(5.0)
    assert af["params"]["n_steps"] == 10
    assert af["params"]["metric"] == "TENENGRAD"
    assert af["params"]["sample_id"] == "LIMS-42"
    assert af["result_summary"]["best_z_mm"] == pytest.approx(12.0)


def test_qpi_audit_captures_indices(captured_audit):
    from ui2.reconstruction import ReconParams
    from ui2.workers import ScienceDriver

    driver = ScienceDriver()
    done = [False]

    fake_loaded = MagicMock()
    fake_loaded.array = np.zeros((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    fake_qpi = MagicMock()
    fake_qpi.phase_stats = MagicMock(range_nm=250.0)
    fake_qpi.total_dry_mass_pg = 1.2
    fake_qpi.step_height_m = 1e-7

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.zeros((16, 16), dtype=np.complex64),
                             (8, 8))), \
         patch("ui2.workers.propagate",
               return_value=np.zeros((16, 16), dtype=np.complex64)), \
         patch("ui2.workers.unwrap_phase_advanced",
               return_value=np.zeros((16, 16), dtype=np.float32)), \
         patch("ui2.workers.compute_qpi", return_value=fake_qpi):
        driver.run_qpi(
            Path("/dev/null"),
            ReconParams(n_sample=1.42, n_medium=1.333),
            z_mm=3.5, sample_id="S-3",
            on_result=lambda r: done.__setitem__(0, True),
            on_error=lambda e: done.__setitem__(0, True),
        )
        assert _wait(lambda: done[0])
    driver.shutdown()

    records = _read_audit_records(captured_audit)
    qpi = next((r for r in records if r["action"] == "qpi"), None)
    assert qpi is not None
    assert qpi["params"]["n_sample"] == pytest.approx(1.42)
    assert qpi["params"]["n_medium"] == pytest.approx(1.333)
    assert qpi["params"]["z_mm"] == pytest.approx(3.5)
    assert qpi["params"]["sample_id"] == "S-3"
    assert qpi["result_summary"]["opd_range_nm"] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Cancel — cancel_check is threaded through the scan
# ---------------------------------------------------------------------------

def test_cancel_autofocus_raises_autofocus_cancelled(captured_audit):
    """When cancel() is called during a scan, the core function's
    cancel_check sees the flag and raises AutofocusCancelled — which
    the driver reports as 'Cancelled.' via on_error."""
    from core.autofocus import FocusMetric
    from ui2.reconstruction import ReconParams
    from ui2.workers import ScienceDriver

    driver = ScienceDriver()
    outcome: dict[str, object] = {}
    done = [False]

    fake_loaded = MagicMock()
    fake_loaded.array = np.zeros((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    # Simulate a slow scan that polls cancel_check.
    def slow_scan(field, base, zs, method, metric, *,
                  cancel_check=None, **_):
        for _ in range(100):
            time.sleep(0.01)
            if cancel_check and cancel_check():
                from core.autofocus import AutofocusCancelled
                raise AutofocusCancelled()
        raise AssertionError("scan should have been cancelled")

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.zeros((16, 16), dtype=np.complex64),
                             (8, 8))), \
         patch("ui2.workers.autofocus_zscan", side_effect=slow_scan):
        driver.run_autofocus(
            Path("/dev/null"), ReconParams(),
            z_min_mm=-1.0, z_max_mm=1.0, n_steps=10,
            on_result=lambda r: outcome.__setitem__("ok", r) or done.__setitem__(0, True),
            on_error=lambda e: outcome.__setitem__("err", e) or done.__setitem__(0, True),
        )
        # Give the worker a moment to enter the scan.
        time.sleep(0.05)
        assert driver.cancel(), "cancel() should report an in-flight job"
        assert _wait(lambda: done[0])
    driver.shutdown()

    assert "err" in outcome, f"expected cancel → error path, got {outcome}"
    assert "Cancelled" in outcome["err"].message


def test_cancel_with_no_inflight_returns_false():
    from ui2.workers import ScienceDriver
    d = ScienceDriver()
    try:
        assert d.cancel() is False
    finally:
        d.shutdown()


def test_driver_discards_late_result_after_cancel():
    """A job that finishes after cancel() was called must NOT call
    on_result — the user saw 'Cancelled' and a stale payload would
    confuse them."""
    from ui2.reconstruction import ReconParams
    from ui2.workers import ScienceDriver

    driver = ScienceDriver()
    events = {"result_called": False, "error_called": False}
    done = [False]

    fake_loaded = MagicMock()
    fake_loaded.array = np.zeros((16, 16), dtype=np.float32)
    fake_loaded.metadata = {}

    # autofocus_zscan returns normally — we ignore cancel_check on purpose
    # to simulate a job that completed just before cancel observation.
    fake_af = MagicMock()
    fake_af.best_z_m = 0.005
    fake_af.scores = {0.005: 0.7}

    def immediate_scan(*args, **kwargs):
        # Set the cancel flag from the worker thread before returning,
        # so ``_dispatch`` sees cancelled=True.
        driver._cancel_event.set()
        return fake_af

    with patch("ui2.workers.load_any", return_value=fake_loaded), \
         patch("ui2.workers.extract_complex_field_offaxis",
               return_value=(np.zeros((16, 16), dtype=np.complex64),
                             (8, 8))), \
         patch("ui2.workers.autofocus_zscan", side_effect=immediate_scan):
        driver.run_autofocus(
            Path("/dev/null"), ReconParams(),
            z_min_mm=-1.0, z_max_mm=1.0, n_steps=10,
            on_result=lambda r: events.__setitem__("result_called", True)
                                 or done.__setitem__(0, True),
            on_error=lambda e: events.__setitem__("error_called", True)
                                 or done.__setitem__(0, True),
        )
        assert _wait(lambda: done[0])
    driver.shutdown()

    assert events["error_called"] is True
    assert events["result_called"] is False
