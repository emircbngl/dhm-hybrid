"""Keep DHM's release-facing version declarations aligned.

The project has no packaging metadata yet, so ``src/__version__.py`` is the
source of truth.  This gate prevents a release from claiming one version in
the application and another in its README, changelog, or security policy.

Run without arguments on development branches.  ``--post`` additionally
requires a matching local git tag (``vX.Y.Z``) after the release tag has been
fetched or created.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POST = "--post" in sys.argv[1:]
errors: list[str] = []


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def capture(pattern: str, text: str, label: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        errors.append(f"{label}: expected declaration not found")
        return None
    return match.group(1)


source = read("src/__version__.py")
version = capture(r'^__version__\s*=\s*"([^"]+)"', source, "src/__version__.py")
if version is None:
    print("RELEASE CONSISTENCY FAIL (no source version)")
    raise SystemExit(1)

print(f"source version: {version}")

changelog = read("CHANGELOG.md")
if not re.search(rf"^## \[{re.escape(version)}\]", changelog, re.MULTILINE):
    errors.append(f"CHANGELOG.md: no [{version}] section")

readme = read("README.md")
if f"Version {version}" not in readme:
    errors.append(f"README.md: does not identify Version {version}")

security = read("SECURITY.md")
security_version = capture(r"^\*\*Version:\*\*\s*([^\s]+)", security, "SECURITY.md")
if security_version and security_version != version:
    errors.append(f"SECURITY.md version {security_version} != source {version}")

if POST:
    tag = f"v{version}"
    result = subprocess.run(
        ["git", "-C", str(ROOT), "tag", "-l", tag],
        capture_output=True,
        text=True,
        check=False,
    )
    if tag not in result.stdout.split():
        errors.append(f"git tag {tag} does not exist")

if errors:
    for error in errors:
        print(f"MISMATCH: {error}")
    print(f"RELEASE CONSISTENCY FAIL ({len(errors)})")
    raise SystemExit(1)

print(f"RELEASE CONSISTENCY PASS ({'post' if POST else 'pre'} mode, version {version})")
