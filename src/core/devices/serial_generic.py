"""Generic serial-port device — RS-232 / USB-CDC line protocol.

Lab hardware that doesn't have a vendor SDK usually speaks one of
two patterns:

1. **ASCII command/reply**: write ``MOVE 100,200<LF>``, read
   ``OK<CR><LF>``. Most Arduinos + lab controllers.
2. **Binary framed**: SLIP / COBS / vendor-specific. Rare in lab
   bench top instruments; use a vendor backend module instead.

This module covers (1) — a thin pyserial wrapper exposing
:meth:`send_command` (write line + read reply with timeout). It
satisfies the bare :class:`Device` Protocol (connect / disconnect
/ is_connected) plus a generic :meth:`send_command` that vendor-
specific stage / shutter / LED wrappers can compose against.

pyserial is optional. ``is_available()`` probes import. Without
it, ``make()`` raises a clear RuntimeError pointing at
``pip install pyserial``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from . import DeviceBackendInfo, DeviceKind


_LOG = logging.getLogger(__name__)


BACKEND = DeviceBackendInfo(
    name="serial_generic",
    display_name="Serial port (generic)",
    kind=DeviceKind.GENERIC,
    summary="ASCII command/reply over RS-232 / USB-CDC. Use as "
            "the base for vendor stages / shutters / LEDs that "
            "don't ship a Python SDK.",
    requires_sdk=("pyserial",),
    capabilities={
        "ascii": True,
        "configurable_baud": True,
    },
)


def is_available() -> bool:
    try:
        import serial  # noqa: F401  (pyserial)
        return True
    except Exception:
        return False


@dataclass
class SerialGenericDevice:
    """Thin pyserial wrapper. Connection settings frozen at
    construction; reconfiguration goes through disconnect →
    re-init."""
    name: str = "serial-generic"
    port: str = "/dev/tty.usbserial"
    baudrate: int = 115200
    timeout_s: float = 1.0
    line_terminator: str = "\n"

    _connected: bool = field(default=False, init=False)
    _ser: Any = field(default=None, init=False)

    # ── lifecycle ─────────────────────────────────────────────
    def connect(self) -> None:
        if self._connected:
            return
        try:
            import serial  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                "serial_generic backend needs pyserial — "
                "pip install pyserial",
            ) from exc
        self._ser = serial.Serial(
            self.port,
            baudrate=self.baudrate,
            timeout=self.timeout_s,
        )
        self._connected = True
        _LOG.info("serial_generic[%s]: opened %s @ %d",
                  self.name, self.port, self.baudrate)

    def disconnect(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                _LOG.debug("serial_generic close raised", exc_info=True)
            self._ser = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── command/reply ────────────────────────────────────────
    def send_command(self, command: str,
                     *, expect_reply: bool = True) -> str:
        """Write ``command`` followed by ``line_terminator``,
        optionally read one reply line, return the trimmed reply.

        Returns ``""`` when ``expect_reply=False``."""
        if not self._connected or self._ser is None:
            raise RuntimeError("send_command() before connect()")
        payload = f"{command}{self.line_terminator}".encode("ascii")
        self._ser.write(payload)
        self._ser.flush()
        if not expect_reply:
            return ""
        raw = self._ser.readline()
        return raw.decode("ascii", errors="replace").strip()


def make(*,
         name: str = "serial-generic",
         port: str = "/dev/tty.usbserial",
         baudrate: int = 115200,
         timeout_s: float = 1.0,
         line_terminator: str = "\n",
         ) -> SerialGenericDevice:
    return SerialGenericDevice(
        name=name, port=port, baudrate=baudrate,
        timeout_s=timeout_s, line_terminator=line_terminator,
    )
