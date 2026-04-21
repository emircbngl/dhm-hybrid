"""About dialog for DHM Reconstruction.

Shows app name, version, git SHA, build date, vendor, and exposes a button
to copy the `version_string()` to the clipboard (handy for support tickets).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDialogButtonBox,
)

from __version__ import (
    __app_name__,
    __build_date__,
    __git_sha__,
    __vendor__,
    __version__,
    version_string,
)


class AboutDialog(QDialog):
    """Modal dialog with version info + copy-version button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {__app_name__}")
        self.setModal(True)
        self.resize(420, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel(f"<h2 style='margin:0'>{__app_name__}</h2>")
        title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(title)

        # Version line
        version_label = QLabel(f"<b>Version:</b> {__version__}")
        version_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(version_label)

        # Git SHA + build date
        build_label = QLabel(
            f"<b>Git SHA:</b> {__git_sha__}<br>"
            f"<b>Build date:</b> {__build_date__}"
        )
        build_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(build_label)

        # Vendor
        vendor_label = QLabel(f"<b>Vendor:</b> {__vendor__}")
        vendor_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(vendor_label)

        # Changelog reference (plain text path, not clickable — matches spec)
        changelog = QLabel("<b>Changelog:</b> CHANGELOG.md")
        changelog.setTextFormat(Qt.TextFormat.RichText)
        changelog.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(changelog)

        layout.addStretch(1)

        # Buttons row: Copy version + Close
        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy version")
        copy_btn.setToolTip("Copy the full version string to the clipboard")
        copy_btn.clicked.connect(self._on_copy_version)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch(1)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        close_box.accepted.connect(self.accept)
        btn_row.addWidget(close_box)
        layout.addLayout(btn_row)

        self._copy_btn = copy_btn

    def _on_copy_version(self) -> None:
        """Place version_string() on the system clipboard."""
        try:
            QGuiApplication.clipboard().setText(version_string())
            # Transient feedback on the button itself
            self._copy_btn.setText("Copied!")
        except Exception:
            self._copy_btn.setText("Copy failed")
