"""OnboardingDialog — first-run walkthrough (ui3).

Behaviour parity with ``ui2.dialogs.show_onboarding`` (a modal DPG window
with four collapsing sections: load a hologram, tune parameters,
reconstruct + autofocus, command palette + themes) — reproduced here as a
paged ``QStackedWidget`` with Back/Next navigation instead of DPG collapsing
headers, plus a fifth page ui2 doesn't have: AI panel + reference-free
tips (this app grew an AI/vision panel and a reference-free reconstruction
mode since the ui2 dialog was written, and new users benefit from a pointer
to both up front).

The dialog never touches persistence itself — it is a pure Qt widget that
reports the final "don't show again" checkbox state via ``seen_requested``
(bool) when closed. The caller (``MainWindow`` integration, mount_hint below)
is responsible for writing that into ``Ui3State.onboarding_seen`` and calling
``save_state``. This keeps the dialog importable/testable without the state
module's file-system side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui3.context import PanelContext
from ui3.design import Space, Type


@dataclass(frozen=True)
class OnboardingPage:
    title: str
    body: str


PAGES: List[OnboardingPage] = [
    OnboardingPage(
        "Welcome",
        "A short tour of the workflow you'll use every day — five quick "
        "steps, skippable anytime.",
    ),
    OnboardingPage(
        "1 — Load a hologram",
        "Drag any TIFF / PNG onto the window, or use File → Load "
        "hologram… (Ctrl+O / ⌘O). The Input panel previews the raw "
        "frame immediately.",
    ),
    OnboardingPage(
        "2 — Tune parameters",
        "Set wavelength (nm), pixel pitch (µm), and the +1-order mask "
        "radius in the Reconstruct panel. Presets seed common setups "
        "(cell / thin-film / USAF).",
    ),
    OnboardingPage(
        "3 — Reconstruct + autofocus",
        "Press Ctrl+R / ⌘R to reconstruct at the current z. Autofocus "
        "scans a z range and lands on the best plane; Find focus "
        "candidates is handy for layered samples.",
    ),
    OnboardingPage(
        "4 — Command palette + themes",
        "Press Ctrl+K / ⌘K to search every action. Change the theme "
        "anytime via View → Theme.",
    ),
    OnboardingPage(
        "5 — AI panel + reference-free",
        "No reference hologram on hand? Switch Reference mode to "
        "Reference-free in the Reconstruct panel to fit and subtract a "
        "numerical background instead. The AI panel can inspect your "
        "current reconstruction and suggest next steps — open it "
        "anytime from the AI dock.",
    ),
]


class OnboardingDialog(QDialog):
    """Paged first-run wizard.

    ``finished_with_choice`` fires once, right before the dialog closes
    (via Finish, Skip, or the window close button), carrying the final
    "don't show again" checkbox state so the caller can persist it.
    """

    finished_with_choice = Signal(bool)  # dont_show_again

    def __init__(self, ctx: Optional[PanelContext] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self.setWindowTitle("Welcome to DHM Reconstruction")
        self.setModal(True)
        self.resize(560, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(Space.lg, Space.lg, Space.lg, Space.lg)
        root.setSpacing(Space.md)

        self._stack = QStackedWidget()
        self._page_widgets: List[QWidget] = []
        for page in PAGES:
            self._page_widgets.append(self._build_page(page))
            self._stack.addWidget(self._page_widgets[-1])
        root.addWidget(self._stack, 1)

        self._dots = QLabel()
        self._dots.setProperty("role", "muted")
        self._dots.setAlignment(Qt.AlignCenter)
        root.addWidget(self._dots)

        self.cb_dont_show = QCheckBox("Don't show again")
        root.addWidget(self.cb_dont_show)

        nav = QHBoxLayout()
        self.btn_skip = QPushButton("Skip")
        self.btn_skip.clicked.connect(self._on_skip)
        self.btn_back = QPushButton("Back")
        self.btn_back.clicked.connect(self._go_back)
        self.btn_next = QPushButton("Next")
        self.btn_next.setProperty("role", "primary")
        self.btn_next.clicked.connect(self._go_next)
        nav.addWidget(self.btn_skip)
        nav.addStretch(1)
        nav.addWidget(self.btn_back)
        nav.addWidget(self.btn_next)
        root.addLayout(nav)

        self._refresh_nav()

    # ------------------------------------------------------------------
    def _build_page(self, page: OnboardingPage) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(Space.sm)
        title = QLabel(page.title)
        title.setProperty("role", "heading")
        lay.addWidget(title)
        body = QLabel(page.body)
        body.setWordWrap(True)
        body.setProperty("role", "muted")
        lay.addWidget(body)
        lay.addStretch(1)
        return w

    def _refresh_nav(self) -> None:
        idx = self._stack.currentIndex()
        last = idx == len(PAGES) - 1
        self.btn_back.setEnabled(idx > 0)
        self.btn_next.setText("Finish" if last else "Next")
        self._dots.setText(f"{idx + 1} / {len(PAGES)}")

    def _go_next(self) -> None:
        idx = self._stack.currentIndex()
        if idx >= len(PAGES) - 1:
            self._finish()
            return
        self._stack.setCurrentIndex(idx + 1)
        self._refresh_nav()

    def _go_back(self) -> None:
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
            self._refresh_nav()

    def _on_skip(self) -> None:
        self._finish()

    def _finish(self) -> None:
        dont_show = bool(self.cb_dont_show.isChecked())
        self.finished_with_choice.emit(dont_show)
        self.accept()

    # ------------------------------------------------------------------
    def current_page_index(self) -> int:
        return self._stack.currentIndex()


__all__ = ["OnboardingDialog", "OnboardingPage", "PAGES"]
