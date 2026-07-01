"""Self-study drills for the gpu-kernels course.

Run from the repo root (or anywhere -- paths resolve from this file):

    python drill/drill.py quiz              # spaced-repetition flashcards
    python drill/drill.py quiz --family layout-strides --n 6
    python drill/drill.py redo e07          # wipe kernel.py back to the stub
    python drill/drill.py redo e07 --restore
    python drill/drill.py predict e05       # guess the number, then measure
    python drill/drill.py stats

Three study methods:
  quiz    -- Leitner-box flashcards from drill/bank.json.
  redo    -- blank-page rebuild: back up your solution, restore the pristine
             stub from git (works because solved kernel.py files are
             skip-worktree'd, so HEAD still holds the clean stub).
  predict -- calibration: predict GB/s or TFLOP/s, run the harness, log the
             error to drill/.state/predictions.csv.

The core logic lives in plain functions (no input()/print()) so that
drill/server.py can import and reuse it; the CLI subcommands below are thin
wrappers around those functions.
"""
import argparse
import csv
import json
import random
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRILL = REPO / "drill"
STATE_DIR = DRILL / ".state"
STATE_FILE = STATE_DIR / "state.json"
PRED_FILE = STATE_DIR / "predictions.csv"
BACKUP_DIR = REPO / "solutions" / "redo_backups"
EXDIR = REPO / "exercises"

# Families to prioritize when more questions are due than fit in a session.
# Edit this as your weak spots change.
WEAK_FAMILIES = ["layout-strides", "numerics-precision"]

# Leitner box -> days until next review.
INTERVALS = {0: 0, 1: 1, 2: 3, 3: 10, 4: 30}
MAX_BOX = 4


class DrillError(Exception):
    """Recoverable error: the CLI prints it, the server maps it to a 4xx."""


class ExerciseNotFound(DrillError):
    """Unknown exercise prefix -- the server maps this to a 404."""


# --------------------------------------------------------------------------
# core: bank + state
# --------------------------------------------------------------------------

def using_sample_bank():
    return not (DRILL / "bank.json").exists()


def load_bank():
    path = DRILL / "bank.json"
    if not path.exists():
        sample = DRILL / "bank.sample.json"
        if not sample.exists():
            raise DrillError("No drill/bank.json or drill/bank.sample.json found.")
        path = sample
    with path.open(encoding="utf-8") as f:
        bank = json.load(f)
    seen = set()
    for q in bank:
        if q["id"] in seen:
            raise DrillError(f"Duplicate question id in bank: {q['id']}")
        seen.add(q["id"])
    return bank


def load_state():
    if STATE_FILE.exists():
        with STATE_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    tmp.replace(STATE_FILE)


def find_exercise(prefix):
    matches = sorted(d for d in EXDIR.iterdir()
                     if d.is_dir() and d.name.startswith(prefix))
    if not matches:
        raise ExerciseNotFound(f"No exercise matches '{prefix}' in {EXDIR}")
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise DrillError(f"'{prefix}' is ambiguous: {names}")
    return matches[0]


# --------------------------------------------------------------------------
# core: quiz
# --------------------------------------------------------------------------

def _entry(state, qid):
    return state.setdefault(qid, {"box": 0, "due": date.today().isoformat(),
                                  "seen": 0, "correct": 0})


def select_due_questions(bank, state, n, family=None, new_only=False):
    today = date.today().isoformat()
    pool = []
    for q in bank:
        if family and q["family"] != family:
            continue
        st = state.get(q["id"])
        if new_only:
            if st is None or st["seen"] == 0:
                pool.append(q)
        elif st is None or st["due"] <= today:
            pool.append(q)

    weak = [q for q in pool if q["family"] in WEAK_FAMILIES]
    rest = [q for q in pool if q["family"] not in WEAK_FAMILIES]
    random.shuffle(weak)
    random.shuffle(rest)
    return (weak + rest)[:n]


def grade_question(state, qid, correct):
    """Apply a grade to one question and persist the state atomically.

    Returns a copy of the updated entry ({box, due, seen, correct}).
    """
    st = _entry(state, qid)
    st["seen"] += 1
    if correct:
        st["correct"] += 1
        st["box"] = min(st["box"] + 1, MAX_BOX)
    else:
        st["box"] = 0
    st["due"] = (date.today()
                 + timedelta(days=INTERVALS[st["box"]])).isoformat()
    save_state(state)
    return dict(st)


def due_counts(bank, state):
    """Counts of questions due by tomorrow / within 3 days."""
    today = date.today()
    tomorrow = (today + timedelta(days=1)).isoformat()
    in3 = (today + timedelta(days=3)).isoformat()
    due_tom = due_3d = 0
    for q in bank:
        st = state.get(q["id"])
        due = st["due"] if st else today.isoformat()
        if due <= tomorrow:
            due_tom += 1
        if due <= in3:
            due_3d += 1
    return {"due_tomorrow": due_tom, "due_3d": due_3d}


def quiz_stats(bank, state):
    """Per-family box distribution + accuracy, and the due-today count."""
    today = date.today().isoformat()
    fams = {}
    due_today = 0
    for q in bank:
        st = state.get(q["id"])
        box = st["box"] if st else 0
        seen = st["seen"] if st else 0
        correct = st["correct"] if st else 0
        due = st["due"] if st else today
        if due <= today:
            due_today += 1
        f = fams.setdefault(q["family"],
                            {"boxes": [0] * (MAX_BOX + 1), "seen": 0, "correct": 0})
        f["boxes"][box] += 1
        f["seen"] += seen
        f["correct"] += correct
    for f in fams.values():
        f["accuracy"] = (100 * f["correct"] / f["seen"]) if f["seen"] else None
    return {"families": fams, "due_today": due_today, "total": len(bank)}


# --------------------------------------------------------------------------
# core: redo
# --------------------------------------------------------------------------

def _git_show_bytes(rel):
    """Contents of a repo-relative path at HEAD, or None if git can't."""
    proc = subprocess.run(["git", "show", f"HEAD:{rel}"],
                          cwd=str(REPO), capture_output=True)
    if proc.returncode != 0:
        return None
    return proc.stdout  # bytes, so line endings survive round-trip


def git_show_stub(folder):
    rel = f"exercises/{folder.name}/kernel.py"
    proc = subprocess.run(["git", "show", f"HEAD:{rel}"],
                          cwd=str(REPO), capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise DrillError(f"Could not read the pristine stub from git "
                         f"(git show HEAD:{rel} failed):\n{err}\n"
                         "Nothing was touched.")
    return proc.stdout  # bytes, so line endings survive round-trip


def backups_for(folder):
    if not BACKUP_DIR.exists():
        return []
    pat = re.compile(rf"^{re.escape(folder.name)}_kernel_(\d+)\.py$")
    found = []
    for p in BACKUP_DIR.iterdir():
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def redo_backup(prefix):
    """Back up kernel.py and reset it to the pristine stub from git HEAD.

    Verifies git can hand us the stub BEFORE touching anything.
    Returns {exercise, backup, kernel, runner_cmd}.
    """
    folder = find_exercise(prefix)
    kernel = folder / "kernel.py"
    if not kernel.exists():
        raise DrillError(f"{kernel} does not exist.")

    # Safety first: make sure git can hand us the stub before touching anything.
    stub = git_show_stub(folder)

    backups = backups_for(folder)
    k = backups[-1][0] + 1 if backups else 1
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"{folder.name}_kernel_{k}.py"
    backup.write_bytes(kernel.read_bytes())

    kernel.write_bytes(stub)
    return {"exercise": folder.name,
            "backup": str(backup),
            "kernel": str(kernel),
            "runner_cmd": f"python -m harness.runner "
                          f"{folder.name.split('_')[0]} --watch"}


def redo_restore(prefix):
    """Copy the latest backup back over kernel.py.

    Returns {exercise, restored_from, kernel}.
    """
    folder = find_exercise(prefix)
    kernel = folder / "kernel.py"
    backups = backups_for(folder)
    if not backups:
        raise DrillError(f"No backups for {folder.name} in {BACKUP_DIR}")
    _, latest = backups[-1]
    kernel.write_bytes(latest.read_bytes())
    return {"exercise": folder.name,
            "restored_from": str(latest),
            "kernel": str(kernel)}


def latest_backup(prefix):
    """Path of the newest backup for an exercise, or None."""
    backups = backups_for(find_exercise(prefix))
    return backups[-1][1] if backups else None


def _norm_eol(b):
    return b.replace(b"\r\n", b"\n")


def list_exercises_with_status():
    """One dict per exercise directory: name, has_kernel, has_spec, metric,
    unit, solved_guess (kernel.py differs from git HEAD?), backups."""
    out = []
    for d in sorted(p for p in EXDIR.iterdir() if p.is_dir()):
        kernel = d / "kernel.py"
        spec = d / "spec.py"
        metric = None
        if spec.exists():
            try:
                metric = read_metric(d)
            except DrillError:
                metric = None
        solved = None
        if kernel.exists():
            head = _git_show_bytes(f"exercises/{d.name}/kernel.py")
            if head is not None:
                solved = _norm_eol(kernel.read_bytes()) != _norm_eol(head)
        out.append({
            "name": d.name,
            "has_kernel": kernel.exists(),
            "has_spec": spec.exists(),
            "metric": metric,
            "unit": metric_unit(metric) if metric and metric != "none" else None,
            "solved_guess": solved,
            "backups": [p.name for _, p in backups_for(d)],
        })
    return out


# --------------------------------------------------------------------------
# core: predict
# --------------------------------------------------------------------------

PERF_RE = re.compile(r"\[PERF\]\s+[\d.]+\s+ms\s+([\d.]+)\s+(GB/s|TFLOP/s)")


def read_metric(folder):
    spec = folder / "spec.py"
    if not spec.exists():
        raise DrillError(f"{folder.name} has no spec.py (CUDA exercise?) -- "
                         "predict only supports the Triton exercises.")
    text = spec.read_text(encoding="utf-8")
    m = re.search(r'^METRIC\s*=\s*["\'](\w+)["\']', text, re.MULTILINE)
    if not m:
        raise DrillError(f"Could not find METRIC in {spec}")
    return m.group(1)


def metric_unit(metric):
    return "GB/s" if metric == "bandwidth" else "TFLOP/s"


def parse_perf(output):
    """Return (value, unit) from runner output, or None."""
    m = PERF_RE.search(output)
    if not m:
        return None
    return float(m.group(1)), m.group(2)


def run_benchmark(prefix, line_callback=None):
    """Run `python -m harness.runner <eNN>` and parse the [PERF] line.

    Returns {exercise, status, output, measured, unit} where status is
    'ok', 'todo' (kernel unimplemented), or 'no_perf' (failed run).
    """
    folder = find_exercise(prefix)
    ex = folder.name.split("_")[0]
    cmd = [sys.executable, "-m", "harness.runner", ex]
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        if line_callback:
            line_callback(line)
        lines.append(line)
    proc.wait()
    output = "".join(lines)

    result = {"exercise": folder.name, "status": "ok", "output": output,
              "measured": None, "unit": None}
    if "[TODO]" in output:
        result["status"] = "todo"
        return result
    perf = parse_perf(output)
    if perf is None:
        result["status"] = "no_perf"
        return result
    result["measured"], result["unit"] = perf
    return result


def log_prediction(exercise, unit, predicted, measured, pct_error, reasoning=""):
    """Append one row to drill/.state/predictions.csv (creates it + header)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not PRED_FILE.exists()
    with PRED_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "exercise", "unit", "predicted", "measured",
                        "pct_error", "reasoning"])
        w.writerow([date.today().isoformat(), exercise, unit,
                    f"{predicted:g}", f"{measured:g}", f"{pct_error:.1f}",
                    reasoning])


def load_predictions():
    if not PRED_FILE.exists():
        return []
    with PRED_FILE.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prediction_summary(rows):
    """Overall + per-exercise mean absolute % error."""
    per_ex = {}
    for r in rows:
        per_ex.setdefault(r["exercise"], []).append(float(r["pct_error"]))
    all_errs = [e for errs in per_ex.values() for e in errs]
    return {
        "overall_mape": (sum(all_errs) / len(all_errs)) if all_errs else None,
        "per_exercise": {ex: {"mape": sum(errs) / len(errs), "n": len(errs)}
                         for ex, errs in sorted(per_ex.items())},
    }


def verdict(pct_error):
    if pct_error <= 10:
        return "calibrated."
    if pct_error <= 30:
        return "in the neighborhood -- refine the mental model."
    return "way off -- work out WHY before moving on."


# --------------------------------------------------------------------------
# CLI wrappers
# --------------------------------------------------------------------------

def _warn_sample_bank():
    if using_sample_bank():
        print("[warn] drill/bank.json not found -- using bank.sample.json")


def cmd_quiz(args):
    _warn_sample_bank()
    bank = load_bank()
    state = load_state()
    session = select_due_questions(bank, state, args.n, args.family, args.new)
    if not session:
        print("Nothing due." + (" (no unseen questions)" if args.new else
                                " Come back tomorrow, or use --new."))
        _print_due_summary(bank, state)
        return

    print(f"{len(session)} question(s) this session. Ctrl+C anytime -- "
          "progress saves after every answer.\n")
    results = []  # (question, correct)
    try:
        for i, q in enumerate(session, 1):
            print(f"--- {i}/{len(session)}  [{q['family']}]  ({q['source']})")
            print(q["q"])
            input("[Enter] to reveal ")
            print("A: " + q["a"])
            print("Why: " + q["why"])
            while True:
                ans = input("Got it right? [y/n] ").strip().lower()
                if ans in ("y", "n"):
                    break
            correct = ans == "y"
            grade_question(state, q["id"], correct)  # saves immediately
            results.append((q, correct))
            print()
    except (KeyboardInterrupt, EOFError):
        print("\n[interrupted -- progress saved]")

    if results:
        right = sum(c for _, c in results)
        print(f"Session: {right}/{len(results)} correct")
        fams = {}
        for q, c in results:
            got, tot = fams.get(q["family"], (0, 0))
            fams[q["family"]] = (got + c, tot + 1)
        for fam in sorted(fams):
            got, tot = fams[fam]
            print(f"  {fam:24s} {got}/{tot}")
    _print_due_summary(bank, state)


def _print_due_summary(bank, state):
    d = due_counts(bank, state)
    print(f"Due by tomorrow: {d['due_tomorrow']}   "
          f"due within 3 days: {d['due_3d']}")


def cmd_redo(args):
    if args.restore:
        folder = find_exercise(args.exercise)
        backups = backups_for(folder)
        if not backups:
            raise DrillError(f"No backups for {folder.name} in {BACKUP_DIR}")
        _, latest = backups[-1]
        print(f"Restore  {latest}")
        print(f"    over {folder / 'kernel.py'}")
        ans = input("Proceed? [y/n] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        redo_restore(args.exercise)
        print("Restored.")
        return

    info = redo_backup(args.exercise)
    print(f"Backed up your solution to:")
    print(f"  {info['backup']}")
    print(f"Reset {info['kernel']} to the pristine stub from git HEAD.")
    print("Rebuild it from the blank page, then validate with:")
    print(f"  {info['runner_cmd']}")
    print(f"To get your old solution back: python drill/drill.py redo "
          f"{args.exercise} --restore")


def cmd_predict(args):
    folder = find_exercise(args.exercise)
    metric = read_metric(folder)
    if metric == "none":
        raise DrillError(f"{folder.name} has METRIC = 'none' -- nothing to predict.")
    unit = metric_unit(metric)

    while True:
        raw = input(f"Predicted {unit} for {folder.name}? ").strip()
        try:
            predicted = float(raw)
            break
        except ValueError:
            print("Enter a number.")
    reasoning = input("reasoning (one line, optional): ").strip()

    prefix = folder.name.split("_")[0]
    print(f"\nRunning: python -m harness.runner {prefix}\n")
    result = run_benchmark(args.exercise,
                           line_callback=lambda line: print(line, end=""))

    if result["status"] == "todo":
        print("\nExercise is unimplemented -- nothing logged.")
        return
    if result["status"] == "no_perf":
        print("\nNo [PERF] line in the output (failed run?) -- nothing logged.")
        return
    measured, meas_unit = result["measured"], result["unit"]
    if meas_unit != unit:
        print(f"[warn] spec says {unit} but runner printed {meas_unit}")
        unit = meas_unit

    pct_error = abs(predicted - measured) / measured * 100
    log_prediction(folder.name, unit, predicted, measured, pct_error, reasoning)

    print(f"\nPredicted {predicted:g} {unit}, measured {measured:g} {unit} "
          f"-> {pct_error:.1f}% error")
    print("Verdict: " + verdict(pct_error))


def cmd_stats(args):
    _warn_sample_bank()
    bank = load_bank()
    state = load_state()

    print("== Quiz ==")
    qs = quiz_stats(bank, state)
    print(f"{'family':24s} {'boxes 0..4':15s} accuracy")
    for fam in sorted(qs["families"]):
        f = qs["families"][fam]
        dist = "/".join(str(b) for b in f["boxes"])
        acc = f"{f['accuracy']:.0f}%" if f["accuracy"] is not None else "--"
        print(f"{fam:24s} {dist:15s} {acc}")
    print(f"Due today: {qs['due_today']}/{qs['total']}")

    print("\n== Predictions ==")
    rows = load_predictions()
    if not rows:
        print("No predictions logged yet.")
        return
    print(f"{'date':11s} {'exercise':26s} {'pred':>9s} {'meas':>9s} "
          f"{'err%':>6s}  unit")
    for r in rows:
        print(f"{r['date']:11s} {r['exercise']:26s} {r['predicted']:>9s} "
              f"{r['measured']:>9s} {r['pct_error']:>6s}  {r['unit']}")
    summary = prediction_summary(rows)
    print(f"\nMean abs % error overall: {summary['overall_mape']:.1f}%")
    for ex, s in summary["per_exercise"].items():
        print(f"  {ex:26s} {s['mape']:.1f}%  (n={s['n']})")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Self-study drills: quiz / redo / predict / stats")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("quiz", help="spaced-repetition flashcards")
    p.add_argument("--family", help="only serve this family")
    p.add_argument("--n", type=int, default=12, help="questions per session (default 12)")
    p.add_argument("--new", action="store_true", help="only never-seen questions")
    p.set_defaults(fn=cmd_quiz)

    p = sub.add_parser("redo", help="reset an exercise to the pristine stub")
    p.add_argument("exercise", help="exercise prefix, e.g. e07")
    p.add_argument("--restore", action="store_true",
                   help="restore the latest backup instead")
    p.set_defaults(fn=cmd_redo)

    p = sub.add_parser("predict", help="predict perf, then measure")
    p.add_argument("exercise", help="exercise prefix, e.g. e05")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("stats", help="quiz + prediction history")
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    try:
        args.fn(args)
    except DrillError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        # quiz saves per-answer; anything else has nothing to lose.
        print("\n[interrupted]")
        sys.exit(1)


if __name__ == "__main__":
    main()
