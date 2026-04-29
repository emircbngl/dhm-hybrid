"""First-run onboarding wizard (v1.4 UI Redesign).

Four short pages that walk a first-time operator through the workflow
they'll actually use: load a hologram, reconstruct, autofocus, and
discover the command palette / themes. Kept deliberately brief — the
wizard's job is to point at the lever, not to replace the user manual.

UX rules
--------
* Can be dismissed at any time (``Skip`` button via ``QWizard``'s
  built-in ``NoBackButtonOnStartPage`` doesn't block the user).
* Never shows automatically after the first successful dismissal —
  the host flips ``ui/onboarding_seen`` when the wizard reaches any
  terminal state (``Finish``, ``Cancel``, or ``close``).
* Re-openable from the command palette (``help.show_onboarding``)
  so the lab can review the workflow after a long break.

The wizard is pure presentation — it emits no signals into the
pipeline. Text lives inline (English only; the language guard would
reject anything else) and references features the app already ships
in v1.0–v1.3.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)


def _body_label(text: str) -> QLabel:
    """Word-wrapped muted body copy — used on every page."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextFormat(Qt.TextFormat.RichText)
    lbl.setOpenExternalLinks(False)
    return lbl


class _IntroPage(QWizardPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Welcome to DHM Reconstruction")
        self.setSubTitle(
            "A four-step tour of the workflow you'll use every day."
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(_body_label(
            "This wizard points at the pieces of the app you'll reach for "
            "first: loading a hologram, running a reconstruction, finding "
            "the focus plane, and using the command palette."
            "<br><br>"
            "You can close it at any time with <b>Cancel</b> or the "
            "window close button. Re-open it later from the command "
            "palette &mdash; search for <b>Show onboarding</b>."
        ))


class _LoadPage(QWizardPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("1 — Load a hologram")
        self.setSubTitle(
            "Point at the file or the camera feed."
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(_body_label(
            "Click <b>Load File</b> on the toolbar and pick a TIFF / PNG. "
            "The dialog remembers the last folder, so subsequent loads "
            "open next to your data."
            "<br><br>"
            "For live acquisition, switch <b>Mode</b> (toolbar) to "
            "<i>Live</i>; the sidebar then shows Camera + Record tabs."
        ))


class _ReconstructPage(QWizardPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("2 — Reconstruct + autofocus")
        self.setSubTitle(
            "Tune parameters in the sidebar, run from the toolbar."
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(_body_label(
            "Set wavelength, pixel size, and mask radius in the "
            "<b>Recon</b> tab. Click <b>⬢ Reconstruct</b> (toolbar, far "
            "right) or press <b>Ctrl+R</b>."
            "<br><br>"
            "For focus, the <b>Focus</b> tab carries a one-shot autofocus "
            "button and a z-range / metric selector. The command palette "
            "&mdash; press <b>Ctrl+K</b> &mdash; also hosts "
            "<b>Find multiple focus planes</b> for multi-depth scenes."
        ))


class _DiscoveryPage(QWizardPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("3 — Command palette + themes")
        self.setSubTitle(
            "Everything the app does lives in one searchable list."
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(_body_label(
            "Press <b>Ctrl+K</b> to open the command palette. Type a few "
            "letters of what you want &mdash; <i>export</i>, <i>theme</i>, "
            "<i>depth</i> &mdash; and the matching actions appear."
            "<br><br>"
            "Appearance: search for <b>Theme</b> and pick Light, Dark, or "
            "System to follow your OS. The choice persists across "
            "sessions."
            "<br><br>"
            "Errors surface as toasts (top-right) and in a dedicated "
            "drawer &mdash; <b>Show error log</b> in the palette re-opens "
            "it anytime."
        ))


class _OutroPage(QWizardPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("4 — You're set")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        heading = QLabel("Happy hologramming.")
        f = heading.font()
        f.setPointSize(f.pointSize() + 2)
        f.setWeight(QFont.Weight.DemiBold)
        heading.setFont(f)
        layout.addWidget(heading)
        layout.addWidget(_body_label(
            "Questions: the <b>Help &rarr; About</b> menu has the version "
            "and support address; <b>docs/ROADMAP.md</b> shows what's "
            "coming next."
            "<br><br>"
            "Click <b>Finish</b> to dismiss this tour."
        ))


class OnboardingWizard(QWizard):
    """Four-page first-run walkthrough."""

    WIZARD_TITLE = "DHM Reconstruction — welcome"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.WIZARD_TITLE)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setOption(QWizard.WizardOption.HaveHelpButton, False)
        self.setMinimumSize(540, 380)

        self.addPage(_IntroPage(self))
        self.addPage(_LoadPage(self))
        self.addPage(_ReconstructPage(self))
        self.addPage(_DiscoveryPage(self))
        self.addPage(_OutroPage(self))


__all__ = ["OnboardingWizard"]
