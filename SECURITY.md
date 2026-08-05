# Security Statement — DHM Reconstruction

**Version:** 2.0.0
**Last updated:** 2026-08-05
**Vendor:** Hybrid Optics

This document describes the security posture of the DHM Reconstruction
desktop application — what it does on your machine, what it does not do,
and how to report a vulnerability.

---

## Network behavior

The application makes **zero outbound network connections** for its core
off-file analysis workflow (ingestion, reconstruction, autofocus, QPI,
masking, profile management, reporting, audit logging).

Verified by source inspection of the optical pipeline: no imports of `socket`,
`urllib`, `http`, `requests`, `httpx`, `aiohttp`, or `websockets`. The PySide6
dependency provides `QtNetwork`, but no part of the optical pipeline imports
or uses it.

The optional **AI assistant** is outside that offline claim. When a user
explicitly enables it and starts a health check or chat, `src/core/ai/client.py`
uses `requests` to contact the endpoint configured in AI Settings (Ollama and
LM Studio default to `localhost`; a user may deliberately configure another
reachable host). No request is made merely by importing or opening the app.

Caveats where we cannot guarantee "zero":

- **Qt platform integration.** Qt / PySide6 may perform host-level
  operations (font discovery, display enumeration) that on some
  platforms involve local IPC. These are not network connections.
- **Third-party native libraries** (NumPy BLAS, OpenCV, pyFFTW, VisPy,
  PyOpenGL) link against system libraries whose behavior is determined
  by the OS, not by this application.
- **Camera acquisition path** (optional, NI-IMAQdx). See "Open ports"
  below.

## Data locality

All user data lives under the user's home directory, in
`~/.dhm-reconstruction/`:

| Subdirectory | Contents                                              |
|--------------|-------------------------------------------------------|
| `audit/`     | Append-only JSONL audit log, one file per day         |
| `profiles/`  | Saved acquisition / reconstruction profiles           |
| `crash/`     | Local crash dumps written by the crash handler        |
| `reports/`   | Generated PDF / HTML reports (if configured)          |

Input holograms are read from wherever the user selects on disk. Output
artifacts (reconstructions, phase maps, reports) are written to paths
the user explicitly chooses. **Nothing leaves the machine** without the
user manually exporting a file.

## Telemetry

**None.** The application contains no analytics, no usage reporting, no
auto-update checks, no error-reporting SaaS integration (Sentry,
Rollbar, Bugsnag, etc.), and no crash-reporting service. Crash dumps
are written to local disk only — see `src/core/crash_handler.py`.

## Open ports

The application opens **no listening ports** by default.

If the optional camera acquisition path is enabled (NI-IMAQdx /
equivalent vendor SDK), the SDK and its kernel driver may bind
platform-specific sockets to communicate with the camera hardware.
That is a device-driver concern governed by the SDK vendor's
documentation and the host OS, not by this application. No ports are
opened by application code.

## Dependencies

Top-level runtime dependencies from `requirements.txt`:

| Package               | Purpose                                 | License family |
|-----------------------|-----------------------------------------|----------------|
| PySide6               | Qt GUI bindings                         | LGPL           |
| mlx                   | Apple Silicon GPU arrays                | MIT            |
| numpy                 | Numerics                                | BSD            |
| scikit-image          | Image processing                        | BSD            |
| opencv-python         | Image processing                        | Apache 2.0     |
| pyqtgraph             | Qt plotting                             | MIT            |
| vispy                 | GPU visualization                       | BSD            |
| tifffile              | TIFF I/O                                | BSD            |
| imagecodecs           | Image codec backends                    | BSD            |
| h5py                  | HDF5 I/O                                | BSD            |
| qtpy                  | Qt API abstraction                      | MIT            |
| scipy                 | Scientific computing                    | BSD            |
| pyfftw                | FFTW bindings                           | BSD / LGPL     |
| matplotlib            | 2D plotting                             | PSF-based      |
| PyOpenGL              | OpenGL bindings                         | BSD            |
| PyOpenGL-accelerate   | OpenGL C accelerators                   | BSD            |

License classifications above are best-effort and **a formal license
audit is pending for Faz 2**. Transitive dependencies are not listed
here and will be covered in that audit.

## Reporting a vulnerability

Please report suspected vulnerabilities through [GitHub Private Vulnerability
Reporting](https://github.com/emircbngl/dhm-hybrid/security/advisories/new).

Reports are private to repository maintainers. This repository must keep Private
Vulnerability Reporting enabled in its GitHub security settings.

Include:
- Affected version (`About` dialog → version + git SHA)
- Reproduction steps or proof-of-concept
- Your contact info for follow-up

**Service levels (target):**
- Acknowledgement: within **7 days**
- Initial triage and fix or mitigation plan: within **30 days**

Please do not disclose publicly before we have had a chance to
investigate and release a fix.

## Update channel

Updates are **manual**: the customer fetches signed builds from a
vendor-supplied URL (to be determined). The application performs no
automatic update checks and phones no home server on startup or
during use.

## Code signing / notarization

macOS code signing and Apple notarization are planned for a post-2.0
release build. The current v2.0.0 distribution is unsigned and intended
for internal / lab use. Enterprise deployment and signed distribution
are tracked separately.

## Scope

This document covers the `DHM Reconstruction` desktop application
source tree in this repository. It does not cover:

- Camera hardware or vendor SDKs (NI-IMAQdx and similar)
- The host operating system
- User-supplied hologram files or downstream analysis tools

---

For anything not addressed here, contact the vendor at the address
above.
