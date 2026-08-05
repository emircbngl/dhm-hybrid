# Notes for AI agents working in this repo

## Never guess how long something takes — wait for it

The failure this file exists to prevent: an agent starts a long command, has no
idea when it ends, and invents a duration. It sleeps 60 s, polls, sleeps again;
it burns turns on a job that finished in five seconds, or declares success on
one still running. Worst of all, a command that *never* exits looks exactly like
one that is merely slow.

Use the bundled runner. The contract is a file, not a timer.

```bash
python3 tools/job.py run <name> -- <command...>   # returns immediately
python3 tools/job.py wait <name>                  # blocks until it ends
```

`wait` exits with the job's own exit code and prints its duration and last
output. One call. Correct whether the job takes 2 s or 2 h — no interval to
guess, no polling loop to write.

```bash
python3 tools/job.py wait <name> --timeout 600     # distinguish hung from slow
python3 tools/job.py tail <name>                   # stream the log live
python3 tools/job.py list                          # what ran, what is running
python3 tools/job.py stop <name>                   # kill a runaway
```

Only `wait --timeout` can tell "hung" from "slow": on timeout it reports
`still_running`, so a blocking prompt or an unclosed plot window is a loud,
distinguishable failure instead of a silent wait. Logs land in `.jobs/`
(gitignored) and stream as the job produces them.

## Write the full command; do not use a shell variable

```bash
CLI="python3 tools/job.py"; $CLI list      # BROKEN
```

zsh — the macOS default — does not word-split unquoted expansions, so the whole
string is taken as one command name and you get `command not found: python3
tools/job.py`. It reads like a missing file, not a quoting bug. Each agent tool
call also starts a fresh shell, so the variable would be gone next call anyway.
Type the full path every time.

## Running things here

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 tools/job.py run tests -- python3 -m pytest -q
python3 tools/job.py wait tests
```

Measured on an M4: the full suite is **~160-200 s** (1271 passed, 11 skipped).
That is the number to expect — do not poll it every 30 s, and do not assume it
died because two minutes passed.

Note `pytest.ini` sets `pythonpath = src`, so `from core.x import y` works from
the repo root without PYTHONPATH.

## FFT backend / accelerators

`fft_backend: "auto"` selects **PyFFTW -> Scipy -> NumPy**. MLX and Torch are
deliberately NOT in that chain: importing `mlx.core` can abort the interpreter
with SIGABRT on a machine whose Metal device is unusable, and an abort cannot be
caught. Ask for them by name (`fft_backend: "mlx"` / `"torch"`) — each falls
back to NumPy if its runtime does not initialise.

## Drift estimation

`estimate_drift` uses partial phase normalisation (`phase_exponent`, default
0.6), not textbook phase correlation (1.0). 1.0 is measurably fragile: it locks
onto wrong correlation peaks under noise. `drift_track_session` sums each step,
so one bad estimate offsets the whole remaining track. Do not "simplify" it back
to 1.0 — a test pins this.
