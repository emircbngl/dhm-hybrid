"""Compact floating progress panel shown during autofocus."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor


class AutofocusOverlay(QWidget):
    """Small floating panel — shows progress bar, %, ETA, and cancel button."""

    cancel_requested = Signal()

    _PANEL_W = 340
    _PANEL_H = 130

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFixedSize(self._PANEL_W, self._PANEL_H)
        self.setStyleSheet(
            "AutofocusOverlay {"
            "  background: #2b2b2b; border: 1px solid #555;"
            "  border-radius: 10px;"
            "}"
        )

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        # Title row
        self._label = QLabel("Auto-focusing...")
        self._label.setStyleSheet("color: #eee; font-size: 13px; font-weight: bold; background: transparent;")
        layout.addWidget(self._label)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        self._bar.setStyleSheet(
            "QProgressBar { background: #444; border: none; border-radius: 5px; }"
            "QProgressBar::chunk { background: #4a9eff; border-radius: 5px; }"
        )
        layout.addWidget(self._bar)

        # Info row: pct + eta on left, cancel on right
        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #aaa; font-size: 11px; background: transparent;")
        info_row.addWidget(self._info_label, 1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedSize(70, 26)
        self._cancel_btn.setStyleSheet(
            "QPushButton { background: #cc3333; color: white; border: none; "
            "border-radius: 4px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background: #ee4444; }"
            "QPushButton:disabled { background: #666; }"
        )
        self._cancel_btn.clicked.connect(self._on_cancel)
        info_row.addWidget(self._cancel_btn)

        layout.addLayout(info_row)

        # Dot animation
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)

        self.hide()

    # ── public API ──

    def show_overlay(self, message: str = "Auto-focusing"):
        self._base_msg = message
        self._dots = 0
        self._label.setText(message + "...")
        self._info_label.setText("")
        self._bar.setValue(0)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("Cancel")
        self._reposition()
        self.raise_()
        self.show()
        self._timer.start(400)

    def hide_overlay(self):
        self._timer.stop()
        self.hide()

    def set_status(self, text: str):
        self._label.setText(text)

    def update_progress(self, pct: int, elapsed_s: float, eta_s: float):
        """Update progress bar, percentage, and ETA."""
        self._bar.setValue(pct)
        parts = [f"{pct}%"]
        parts.append(f"Elapsed: {self._fmt_time(elapsed_s)}")
        if eta_s >= 0:
            parts.append(f"ETA: {self._fmt_time(eta_s)}")
        self._info_label.setText("  |  ".join(parts))

    # ── internals ──

    def _on_cancel(self):
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("...")
        self._label.setText("Cancelling...")
        self.cancel_requested.emit()

    def _animate(self):
        self._dots = (self._dots + 1) % 4
        if self._cancel_btn.isEnabled():
            self._label.setText(self._base_msg + "." * self._dots)

    def _reposition(self):
        """Center horizontally at the top of the parent."""
        p = self.parent()
        if p:
            x = (p.width() - self.width()) // 2
            self.move(x, 16)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        s = max(0, int(seconds))
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        return f"{m}m {s:02d}s"

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._on_cancel()
        event.accept()
