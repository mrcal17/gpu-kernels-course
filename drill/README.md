# Drills — keep it from evaporating

Three study methods for material you've already covered once. All commands run
from the repo root.

## 1. Spaced quiz (`quiz`)

Flashcards from `drill/bank.json` (falls back to `bank.sample.json` if missing),
scheduled with Leitner boxes: answer right and the question moves out a box
(review in 1 / 3 / 10 / 30 days); answer wrong and it drops back to box 0.

```bash
python drill/drill.py quiz                      # today's due questions (max 12)
python drill/drill.py quiz --n 6                # shorter session
python drill/drill.py quiz --family layout-strides
python drill/drill.py quiz --new                # only never-seen questions
```

Grading is honor-system: read the answer, then say y/n. Progress saves after
every answer, so Ctrl+C loses nothing. When more is due than fits in a session,
the families in `WEAK_FAMILIES` (top of `drill.py`) get served first — edit that
list as your weak spots change.

Bank entries are `{id, family, type, q, a, why, source}` in a JSON array.

## 2. Blank-page redo (`redo`)

The real test of whether you can still write a kernel is writing it again from
the stub. Solved `kernel.py` files are skip-worktree'd, so git HEAD still holds
the pristine stub even though your solution sits in the working tree.

```bash
python drill/drill.py redo e07            # back up your solution, restore the stub
python -m harness.runner e07 --watch      # now rebuild it from scratch
python drill/drill.py redo e07 --restore  # put your latest backup back
```

Backups go to `solutions/redo_backups/<exercise>_kernel_<k>.py` (numbered, never
overwritten). The stub is verified retrievable from git *before* anything is
touched.

## 3. Predict-then-run (`predict`)

Calibration training: commit to a number before you measure. Reads the
exercise's metric from `spec.py`, asks for your predicted GB/s or TFLOP/s (plus
a one-line reason), runs `python -m harness.runner <ex>`, and logs
predicted vs. measured to `drill/.state/predictions.csv`.

```bash
python drill/drill.py predict e05
```

Within 10% = calibrated. Way off = the interesting case — figure out why before
moving on.

## Progress

```bash
python drill/drill.py stats     # box distribution + accuracy per family,
                                # prediction history + mean abs % error
```

State lives in `drill/.state/` (gitignored).

## Web UI

All four methods also have a local web front end (pure stdlib, no installs):

```bash
python drill/server.py            # http://127.0.0.1:8177, opens the browser
python drill/server.py --port 9000
python drill/server.py --no-browser
```

Tabs: **Quiz** (flashcards with keyboard shortcuts — Space to reveal, Y/G got
it, N/M missed; every answer is saved immediately, same as the CLI), **Stats**
(box distribution, accuracy, prediction history + MAPE), **Redo** (blank-page
reset with type-the-name confirmation, restore backups), **Predict** (commit to
a number, run the real benchmark, log the error).

The server binds `127.0.0.1` only and shares all state with the CLI
(`drill/.state/`, `solutions/redo_backups/`) — use whichever is closer to hand.
