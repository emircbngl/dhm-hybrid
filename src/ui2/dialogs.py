"""One-shot Dear PyGui dialogs: onboarding, multi-focus candidates,
QPI batch review, etc. Each dialog is a small builder — takes data +
callbacks, returns the window tag."""
from __future__ import annotations

from typing import Any, Callable, List, Sequence

import dearpygui.dearpygui as dpg

from .theme import ScientificTheme


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------

_ONBOARDING_TAG = "onboarding_modal"


def show_onboarding(*, on_close: Callable[[], None]) -> None:
    """Five-step walkthrough. Uses collapsing sections instead of
    QWizard — simpler, more native to DPG's layout model."""
    if dpg.does_item_exist(_ONBOARDING_TAG):
        dpg.configure_item(_ONBOARDING_TAG, show=True)
        return

    with dpg.window(label="Welcome to DHM Reconstruction",
                    tag=_ONBOARDING_TAG, modal=True,
                    width=620, height=440, no_collapse=True):
        dpg.add_text("A four-step tour of the workflow you'll use every day.",
                     color=ScientificTheme.TEXT_MUTED, wrap=580)
        dpg.add_separator()

        with dpg.collapsing_header(label="1 — Load a hologram",
                                   default_open=True):
            dpg.add_text(
                "Drag any TIFF / PNG onto the window, or use File → "
                "Load hologram… (Ctrl+O / ⌘O). The Input panel will "
                "preview the raw frame immediately.",
                wrap=560,
            )

        with dpg.collapsing_header(label="2 — Tune parameters"):
            dpg.add_text(
                "Set wavelength (nm), pixel pitch (µm), and the +1-order "
                "mask radius in the sidebar. Presets seed common setups "
                "(cell / thin-film / USAF).",
                wrap=560,
            )

        with dpg.collapsing_header(label="3 — Reconstruct + autofocus"):
            dpg.add_text(
                "Press Ctrl+R / ⌘R to reconstruct at the current z. "
                "Autofocus scans a z range and lands on the best plane. "
                "Find multiple focus planes is handy for layered samples.",
                wrap=560,
            )

        with dpg.collapsing_header(label="4 — Command palette + themes"):
            dpg.add_text(
                "Press Ctrl+K / ⌘K to search every action. Change the "
                "theme anytime via View → Theme.",
                wrap=560,
            )

        dpg.add_spacer(height=8)
        dpg.add_button(label="Got it — close", width=140,
                       callback=lambda: (
                           dpg.configure_item(_ONBOARDING_TAG, show=False),
                           on_close(),
                       ))


# ---------------------------------------------------------------------------
# Multi-focus candidate picker
# ---------------------------------------------------------------------------

_MF_TAG = "multi_focus_modal"


def show_focus_candidates(
    candidates: Sequence[Any],
    *,
    on_focus: Callable[[float], None],
) -> None:
    """One row per candidate: rank, z (mm), prominence, Focus-here button."""
    if dpg.does_item_exist(_MF_TAG):
        dpg.delete_item(_MF_TAG)
    with dpg.window(label="Focus candidates", tag=_MF_TAG, modal=False,
                    width=520, height=400, no_collapse=True):
        if not candidates:
            dpg.add_text("No candidate focus planes found — widen the z "
                         "range or drop the prominence threshold.",
                         wrap=480, color=ScientificTheme.TEXT_MUTED)
            return
        dpg.add_text(
            f"{len(candidates)} plausible focus plane(s) — click "
            "*Focus here* to reconstruct at that z.",
            color=ScientificTheme.TEXT_MUTED, wrap=480,
        )
        dpg.add_separator()
        with dpg.table(header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(label="#")
            dpg.add_table_column(label="z (mm)")
            dpg.add_table_column(label="Prominence")
            dpg.add_table_column(label="Action")
            for cand in candidates:
                z_mm = float(cand.z_m) * 1e3
                with dpg.table_row():
                    dpg.add_text(str(cand.rank + 1))
                    dpg.add_text(f"{z_mm:+.3f}")
                    dpg.add_text(f"{cand.prominence:.3f}")
                    dpg.add_button(
                        label="Focus here",
                        callback=lambda s, a, u=float(cand.z_m): (
                            dpg.configure_item(_MF_TAG, show=False),
                            on_focus(u * 1e3),  # pass mm
                        ),
                    )


# ---------------------------------------------------------------------------
# QPI batch review dialog
# ---------------------------------------------------------------------------

_QB_TAG = "qpi_batch_modal"


def show_qpi_batch_review(
    entries: Sequence[Any],
    *,
    on_focus: Callable[[float], None],
    on_export_csv: Callable[[], None],
) -> None:
    if dpg.does_item_exist(_QB_TAG):
        dpg.delete_item(_QB_TAG)
    with dpg.window(label="QPI batch — candidate comparison", tag=_QB_TAG,
                    modal=False, width=820, height=460, no_collapse=True):
        if not entries:
            dpg.add_text("No QPI batch entries — widen the z range or "
                         "drop the prominence threshold.",
                         wrap=780, color=ScientificTheme.TEXT_MUTED)
            return
        dpg.add_text(
            f"{len(entries)} candidate plane(s). Click *Focus here* to "
            "commit, *Export CSV…* to dump the whole table.",
            color=ScientificTheme.TEXT_MUTED, wrap=780,
        )
        dpg.add_separator()
        with dpg.table(header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp):
            for col in ("#", "z (mm)", "prom", "dry mass (pg)",
                        "area (µm²)", "OPD (nm)", "step (nm)",
                        "circularity", ""):
                dpg.add_table_column(label=col)
            for entry in entries:
                q = entry.qpi_result
                morph = getattr(q, "cell_morph", None)
                ph = getattr(q, "phase_stats", None)
                with dpg.table_row():
                    dpg.add_text(str(entry.candidate.rank + 1))
                    dpg.add_text(f"{entry.candidate.z_m * 1e3:+.3f}")
                    dpg.add_text(f"{entry.candidate.prominence:.3f}")
                    dpg.add_text(_fmt(getattr(q, "total_dry_mass_pg", None)))
                    dpg.add_text(_fmt(getattr(morph, "area_um2", None)
                                      if morph else None))
                    dpg.add_text(_fmt(getattr(ph, "range_nm", None)
                                      if ph else None))
                    sh = getattr(q, "step_height_m", None)
                    dpg.add_text(_fmt(sh * 1e9 if sh is not None else None))
                    dpg.add_text(_fmt(getattr(morph, "circularity", None)
                                      if morph else None))
                    dpg.add_button(
                        label="Focus here",
                        callback=lambda s, a, u=float(entry.candidate.z_m): (
                            dpg.configure_item(_QB_TAG, show=False),
                            on_focus(u * 1e3),
                        ),
                    )
        dpg.add_spacer(height=6)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Export CSV…", callback=on_export_csv)
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.configure_item(_QB_TAG, show=False),
            )


def _fmt(value, *, digits: int = 3, missing: str = "—") -> str:
    import math
    if value is None:
        return missing
    try:
        v = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(v):
        return missing
    return f"{v:.{digits}g}"


# ---------------------------------------------------------------------------
# v2.0.7 — Audit log viewer (T5)
# ---------------------------------------------------------------------------

_AUDIT_VIEWER_TAG = "audit_viewer_modal"
_AUDIT_TABLE_TAG = "audit_viewer_table"
_AUDIT_QUERY_TAG = "audit_viewer_query"
_AUDIT_OPERATOR_TAG = "audit_viewer_operator"
_AUDIT_ACTION_TAG = "audit_viewer_action"
_AUDIT_STATUS_TAG = "audit_viewer_status"


def show_audit_viewer(*, audit_dir=None) -> None:
    """Browse + filter the JSONL audit log.

    Sven's compliance ask: instead of `cat audit/*.jsonl | jq` for
    every "what did Erik run yesterday?" question, give the lab a
    point-and-click view with:

    * a free-text query box (substring match on full JSON record);
    * operator + action dropdowns populated from the log itself
      (so the filter list reflects actual usage, not a hardcoded
      enum that goes stale);
    * a results table with one row per record, newest first,
      capped at 500 entries to keep DPG responsive.

    Pure-data layer (read / parse / filter) lives in
    ``core.audit_viewer`` so this dialog can be reasoned about
    without a DPG context.
    """
    from core.audit_viewer import (
        AuditFilter, apply_filter, known_actions, known_operators,
        read_entries,
    )

    if dpg.does_item_exist(_AUDIT_VIEWER_TAG):
        dpg.delete_item(_AUDIT_VIEWER_TAG)

    operators = known_operators(audit_dir)
    actions = known_actions(audit_dir)
    operator_choices = ["(any)"] + operators
    action_choices = ["(any)"] + actions

    def _refresh():
        op_sel = dpg.get_value(_AUDIT_OPERATOR_TAG)
        ac_sel = dpg.get_value(_AUDIT_ACTION_TAG)
        q = (dpg.get_value(_AUDIT_QUERY_TAG) or "").strip() or None
        filt = AuditFilter(
            operators=([op_sel]
                       if op_sel and op_sel != "(any)" else None),
            actions=([ac_sel]
                     if ac_sel and ac_sel != "(any)" else None),
            query=q,
        )
        # Cap at 500 to keep table render snappy. The lab's typical
        # one-day log is a few dozen records; 500 covers a heavy
        # batch session.
        all_entries = read_entries(audit_dir, limit=500)
        rows = apply_filter(all_entries, filt)
        # Clear + repopulate.
        if dpg.does_item_exist(_AUDIT_TABLE_TAG):
            children = dpg.get_item_children(_AUDIT_TABLE_TAG, 1) or []
            for c in children:
                dpg.delete_item(c)
            for e in rows:
                with dpg.table_row(parent=_AUDIT_TABLE_TAG):
                    dpg.add_text(e.timestamp[:19])  # YYYY-MM-DDTHH:MM:SS
                    dpg.add_text(e.operator or e.user or "—")
                    dpg.add_text(e.action)
                    summary_str = (
                        ", ".join(f"{k}={v}"
                                  for k, v in e.params.items())
                        if e.params else ""
                    )
                    dpg.add_text(summary_str[:120])
        if dpg.does_item_exist(_AUDIT_STATUS_TAG):
            dpg.set_value(
                _AUDIT_STATUS_TAG,
                f"{len(rows)} of {len(all_entries)} entries",
            )

    with dpg.window(label="Audit log",
                    tag=_AUDIT_VIEWER_TAG, modal=True,
                    width=900, height=520, no_collapse=True):
        with dpg.group(horizontal=True):
            dpg.add_text("Operator")
            dpg.add_combo(operator_choices, default_value="(any)",
                          tag=_AUDIT_OPERATOR_TAG, width=140,
                          callback=lambda *_: _refresh())
            dpg.add_text("Action")
            dpg.add_combo(action_choices, default_value="(any)",
                          tag=_AUDIT_ACTION_TAG, width=160,
                          callback=lambda *_: _refresh())
            dpg.add_text("Search")
            dpg.add_input_text(tag=_AUDIT_QUERY_TAG, width=240,
                               on_enter=True,
                               callback=lambda *_: _refresh())
            dpg.add_button(label="Refresh",
                           callback=lambda: _refresh())
        dpg.add_separator()
        with dpg.table(tag=_AUDIT_TABLE_TAG, header_row=True,
                       resizable=True, scrollY=True, height=380,
                       borders_innerH=True, borders_outerV=True):
            dpg.add_table_column(label="When (UTC)", width_fixed=True,
                                 init_width_or_weight=160)
            dpg.add_table_column(label="Operator", width_fixed=True,
                                 init_width_or_weight=110)
            dpg.add_table_column(label="Action", width_fixed=True,
                                 init_width_or_weight=140)
            dpg.add_table_column(label="Params (preview)")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text("", tag=_AUDIT_STATUS_TAG,
                         color=ScientificTheme.TEXT_MUTED)
            dpg.add_spacer(width=20)
            dpg.add_button(
                label="Close",
                callback=lambda: dpg.configure_item(
                    _AUDIT_VIEWER_TAG, show=False,
                ),
            )

    _refresh()


# ---------------------------------------------------------------------------
# v2.1.y P1 — Line-profile click-drag dialog
# ---------------------------------------------------------------------------
#
# Thin DPG wrapper around :class:`ui2.line_profile_state.LineProfileEditor`.
# The state machine + bilinear sampler live in ``core.line_profile`` /
# ``ui2.line_profile_state`` so this file stays render-only.
#
# Operator flow:
# 1. Open dialog.
# 2. Two clicks on the input panel → first_click / second_click → a
#    saved profile lands in the editor's list.
# 3. Sampled values render as overlay plots; the dialog table shows
#    label, length, min/max/mean stats, and per-row colour swatch.
# 4. Drop, rename, or recolour from the table; clear all wipes both
#    state + plot.
#
# Image source is supplied by the caller via ``image_provider()`` —
# the dialog never reaches into ``DhmApp`` directly. That keeps the
# wrapper testable and lets a future "compare across reconstructions"
# mode pass two image providers without forking.

_LP_DIALOG_TAG = "line_profile_dialog"
_LP_PLOT_TAG = "line_profile_plot"
_LP_PLOT_X_TAG = "line_profile_plot_x"
_LP_PLOT_Y_TAG = "line_profile_plot_y"
_LP_TABLE_TAG = "line_profile_table"
_LP_STATUS_TAG = "line_profile_status"


def show_line_profiles(*,
                       editor=None,
                       image_provider: Callable = None,
                       on_close: Callable[[], None] = None) -> None:
    """Open the multi-line profile dialog.

    Parameters
    ----------
    editor
        :class:`ui2.line_profile_state.LineProfileEditor` instance.
        ``None`` → a fresh editor is constructed; the caller can
        retrieve it via ``editor=getattr(dpg, '_line_profile_editor',
        None)`` after the call.
    image_provider
        ``Callable[[], Optional[np.ndarray]]`` returning the current
        2-D image (phase / amplitude / OPD) the profiles sample.
        ``None`` (or returning None) → the dialog opens with empty
        plot + no-image hint.
    on_close
        Optional callback when the dialog hides. Useful for
        clearing input-panel click handlers the caller installed
        when opening.
    """
    from ui2.line_profile_state import LineProfileEditor
    from core.line_profile import sample_line, stats_for

    if editor is None:
        editor = LineProfileEditor()
    # Stash on dpg so subsequent calls can find the same editor —
    # avoids losing saved profiles if the caller forgot to thread
    # the instance through.
    dpg._line_profile_editor = editor  # type: ignore[attr-defined]

    if dpg.does_item_exist(_LP_DIALOG_TAG):
        dpg.delete_item(_LP_DIALOG_TAG)

    def _refresh_table() -> None:
        if not dpg.does_item_exist(_LP_TABLE_TAG):
            return
        children = dpg.get_item_children(_LP_TABLE_TAG, 1) or []
        for c in children:
            dpg.delete_item(c)
        # Pull image once for stats — bilinear sampler is the same
        # whether we redo per-row or batch; per-row keeps memory low.
        img = image_provider() if image_provider else None
        for idx, prof in enumerate(editor.profiles):
            with dpg.table_row(parent=_LP_TABLE_TAG):
                dpg.add_text(prof.label or f"line-{idx + 1}")
                dpg.add_text(f"{prof.length_px:.1f}")
                if img is not None:
                    sampled = sample_line(img, prof)
                    s = stats_for(sampled)
                    dpg.add_text(
                        f"{s['min']:.3g}" if s["min"] is not None
                        else "—",
                    )
                    dpg.add_text(
                        f"{s['max']:.3g}" if s["max"] is not None
                        else "—",
                    )
                    dpg.add_text(
                        f"{s['mean']:.3g}" if s["mean"] is not None
                        else "—",
                    )
                else:
                    dpg.add_text("—")
                    dpg.add_text("—")
                    dpg.add_text("—")
                # Drop button — capture idx by default-arg.
                dpg.add_button(
                    label="Drop",
                    callback=lambda s, a, u=idx: _on_drop(u),
                )
        if dpg.does_item_exist(_LP_STATUS_TAG):
            dpg.set_value(
                _LP_STATUS_TAG,
                f"{len(editor.profiles)} profile(s)  ·  state="
                f"{editor.state.value}",
            )

    def _refresh_plot() -> None:
        if not dpg.does_item_exist(_LP_PLOT_TAG):
            return
        # Wipe existing series — DPG accumulates; clear each refresh.
        kids = dpg.get_item_children(_LP_PLOT_Y_TAG, 1) or []
        for k in kids:
            dpg.delete_item(k)
        img = image_provider() if image_provider else None
        if img is None:
            return
        for prof in editor.profiles:
            try:
                sp = sample_line(img, prof)
            except Exception:
                continue
            # NaN samples drop naturally — DPG add_line_series
            # treats NaN as gaps when the platform supports it.
            label = prof.label or "(unlabelled)"
            dpg.add_line_series(
                list(sp.distance_px), list(sp.values),
                label=label, parent=_LP_PLOT_Y_TAG,
            )

    def _on_drop(idx: int) -> None:
        # Drop the *idx*-th profile, not the most recent — mirrors
        # the table column the user clicks.
        if 0 <= idx < len(editor.profiles):
            del editor.profiles[idx]
        _refresh_table()
        _refresh_plot()

    def _on_clear() -> None:
        editor.clear_all()
        _refresh_table()
        _refresh_plot()

    def _on_close() -> None:
        try:
            dpg.configure_item(_LP_DIALOG_TAG, show=False)
        except Exception:
            pass
        if on_close is not None:
            try:
                on_close()
            except Exception:
                pass

    with dpg.window(label="Line profiles", modal=False,
                    tag=_LP_DIALOG_TAG, no_collapse=True,
                    width=720, height=520):
        dpg.add_text(
            "Click two points on the input panel to define a "
            "profile. Repeat to layer more profiles. The plot + "
            "table update automatically.",
            color=ScientificTheme.TEXT_MUTED, wrap=680,
        )
        dpg.add_separator()
        with dpg.plot(tag=_LP_PLOT_TAG, height=220, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(
                dpg.mvXAxis, label="Distance (px)",
                tag=_LP_PLOT_X_TAG,
            )
            dpg.add_plot_axis(
                dpg.mvYAxis, label="Value",
                tag=_LP_PLOT_Y_TAG,
            )
        dpg.add_separator()
        with dpg.table(tag=_LP_TABLE_TAG, header_row=True,
                       resizable=True, scrollY=True, height=180,
                       borders_innerH=True, borders_outerV=True):
            dpg.add_table_column(label="Label",
                                 init_width_or_weight=120)
            dpg.add_table_column(label="Length px",
                                 init_width_or_weight=80)
            dpg.add_table_column(label="Min",
                                 init_width_or_weight=80)
            dpg.add_table_column(label="Max",
                                 init_width_or_weight=80)
            dpg.add_table_column(label="Mean",
                                 init_width_or_weight=80)
            dpg.add_table_column(label="",
                                 init_width_or_weight=60)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text("", tag=_LP_STATUS_TAG,
                         color=ScientificTheme.TEXT_MUTED)
            dpg.add_spacer(width=12)
            dpg.add_button(label="Clear all", callback=_on_clear)
            dpg.add_button(label="Close", callback=_on_close)

    _refresh_table()
    _refresh_plot()


__all__ = [
    "show_onboarding",
    "show_focus_candidates",
    "show_qpi_batch_review",
    "show_audit_viewer",
    "show_line_profiles",
]
