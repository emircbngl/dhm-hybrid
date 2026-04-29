"""Modal settings dialog for the AI assistant.

Reads/writes :class:`core.settings_schema.AIDefaults`. The panel hands
the dialog the current settings; on Accept the dialog returns a fresh
``AIDefaults`` with the user-entered values and the panel writes them
back through ``settings_store.save``.

Kept small — most fields are line edits or spin boxes, no clever
validation. Out-of-range values are caught at clamp time inside the
tool handlers, not here, so the user can save aspirational values and
fix them later.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from core.settings_schema import AIDefaults


# Suggestions only — the ComboBox stays editable so any model name
# the local server reports works. Order matches first-shot tool-call
# reliability on Ollama as of 2026-Q1: qwen2.5 lands cleanly the most
# often; llama3.1 is a close runner-up; llama3.2 is faster to boot
# but misses tool calls more.
_MODEL_SUGGESTIONS = (
    "qwen2.5:7b-instruct",
    "llama3.1:8b-instruct",
    "mistral-nemo:12b-instruct",
    "llama3.2",
    "phi3.5:3.8b",
)


class AISettingsDialog(QDialog):
    """Configure the local LLM connection."""

    def __init__(self, settings: AIDefaults, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Assistant Settings")
        self.setObjectName("ai_settings_dialog")
        self._initial = settings
        self._result: Optional[AIDefaults] = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._enabled = QCheckBox("Enabled", self)
        self._enabled.setChecked(bool(settings.enabled))
        form.addRow(self._enabled)

        self._endpoint = QLineEdit(settings.endpoint_url, self)
        self._endpoint.setPlaceholderText("http://localhost:11434  ·  http://localhost:1234")
        form.addRow("Endpoint URL", self._endpoint)

        self._model = QComboBox(self)
        self._model.setEditable(True)
        for name in _MODEL_SUGGESTIONS:
            self._model.addItem(name)
        self._model.setEditText(settings.model_name)
        form.addRow("Model", self._model)

        self._temperature = QDoubleSpinBox(self)
        self._temperature.setRange(0.0, 2.0)
        self._temperature.setSingleStep(0.05)
        self._temperature.setDecimals(2)
        self._temperature.setValue(float(settings.temperature))
        form.addRow("Temperature", self._temperature)

        self._max_tokens = QSpinBox(self)
        self._max_tokens.setRange(64, 32_768)
        self._max_tokens.setSingleStep(64)
        self._max_tokens.setValue(int(settings.max_tokens))
        form.addRow("Max tokens", self._max_tokens)

        self._max_iter = QSpinBox(self)
        self._max_iter.setRange(1, 32)
        self._max_iter.setValue(int(settings.max_iterations))
        form.addRow("Max tool-call iterations", self._max_iter)

        self._timeout = QDoubleSpinBox(self)
        self._timeout.setRange(5.0, 1800.0)
        self._timeout.setSuffix(" s")
        self._timeout.setSingleStep(5.0)
        self._timeout.setValue(float(settings.request_timeout_s))
        form.addRow("Request timeout", self._timeout)

        self._restrict = QCheckBox("Restrict file access to home directory", self)
        self._restrict.setChecked(bool(settings.restrict_to_home))
        form.addRow(self._restrict)

        self._confirm = QCheckBox("Confirm before irreversible tool calls", self)
        self._confirm.setChecked(bool(settings.confirm_irreversible))
        form.addRow(self._confirm)

        self._redact = QCheckBox("Redact operator + paths in audit tail to LLM", self)
        self._redact.setChecked(bool(settings.audit_redact_for_llm))
        form.addRow(self._redact)

        layout.addLayout(form)

        hint = QLabel(
            "<small>The AI assistant talks to a local LLM (Ollama or LM Studio). "
            "Endpoint must be reachable on this machine. Model is loaded by the "
            "LLM runtime, not by this app.</small>",
            self,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self._result = AIDefaults(
            enabled=bool(self._enabled.isChecked()),
            endpoint_url=str(self._endpoint.text().strip() or self._initial.endpoint_url),
            model_name=str(self._model.currentText().strip() or self._initial.model_name),
            temperature=float(self._temperature.value()),
            max_tokens=int(self._max_tokens.value()),
            max_iterations=int(self._max_iter.value()),
            request_timeout_s=float(self._timeout.value()),
            restrict_to_home=bool(self._restrict.isChecked()),
            confirm_irreversible=bool(self._confirm.isChecked()),
            audit_redact_for_llm=bool(self._redact.isChecked()),
        )
        self.accept()

    def result_settings(self) -> Optional[AIDefaults]:
        return self._result


__all__ = ["AISettingsDialog"]
