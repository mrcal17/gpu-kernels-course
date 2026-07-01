"""Exercise runner: checks correctness, then measures performance.

Run from the repo root:

    python -m harness.runner e01            # run one exercise once
    python -m harness.runner e01 --watch    # re-run every time you save kernel.py
    python -m harness.runner --all          # run every exercise

Each exercise folder under exercises/ provides:
  kernel.py  - YOU write this (the kernel + its launch wrapper)
  spec.py    - the harness contract (reference impl, inputs, metric). Don't edit.

A spec.py must define:
  TITLE        : str
  ENTRYPOINT   : str          # name of the function in kernel.py to call
  make_inputs(): tuple        # tensors (already on cuda)
  reference(*inputs)          # the correct answer (torch)
  METRIC       : "bandwidth" | "flops" | "none"
  bytes_moved(*inputs) -> int # required if METRIC == "bandwidth"
  flops(*inputs) -> int       # required if METRIC == "flops"
  TOL          : dict         # optional, e.g. {"atol": 1e-2, "rtol": 1e-2}
"""
import argparse
import importlib.util
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# torch is only needed for the Triton path. Import it lazily so that a
# torch-less environment can still build and run pure-CUDA (kernel.cu)
# exercises. The Triton branch checks `torch is not None` before using it.
try:
    import torch
except Exception:  # pragma: no cover - depends on the host env
    torch = None

REPO = Path(__file__).resolve().parent.parent
EXDIR = REPO / "exercises"

_counter = 0


def _load(path: Path):
    """Load a .py file as a fresh module every call (so --watch sees edits)."""
    global _counter
    _counter += 1
    name = f"{path.parent.name}_{path.stem}_{_counter}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _peak_bw_gbps() -> float | None:
    try:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        # memory_clock_rate is kHz; x2 for double-data-rate; bus_width in bits.
        return 2 * (p.memory_clock_rate * 1e3) * (p.memory_bus_width / 8) / 1e9
    except Exception:
        return None


def _bench(fn) -> float:
    """Return median milliseconds per call."""
    try:
        from triton.testing import do_bench
        try:
            return float(do_bench(fn, warmup=25, rep=100, return_mode="median"))
        except TypeError:
            # Older/newer triton without return_mode: ask for the 0.5 quantile.
            return float(do_bench(fn, warmup=25, rep=100, quantiles=[0.5]))
    except Exception:
        # Fallback: CUDA events, collecting per-iteration times and taking the median.
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        n = 50
        times = []
        for _ in range(n):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return statistics.median(times)


def _find_exercise(prefix: str) -> Path:
    matches = sorted(d for d in EXDIR.iterdir() if d.is_dir() and d.name.startswith(prefix))
    if not matches:
        raise SystemExit(f"No exercise matches '{prefix}' in {EXDIR}")
    return matches[0]


def _as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


# --------------------------------------------------------------------------
# CUDA C++ path (kernel.cu + harness.cu, built with standalone nvcc)
# --------------------------------------------------------------------------

_VCVARS_FALLBACK = (
    "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/"
    "VC/Auxiliary/Build/vcvars64.bat"
)


def _find_vcvars() -> str:
    """Locate vcvars64.bat so nvcc can find the MSVC host compiler.

    Resolution order:
      1. VCVARS env override (if set and the path exists).
      2. vswhere -latest -products * -find VC\\Auxiliary\\Build\\vcvars64.bat
      3. Hardcoded BuildTools 2022 fallback path.
    Raises SystemExit if none exist.
    """
    # 1. Env override.
    override = os.environ.get("VCVARS")
    if override and Path(override).exists():
        return override

    # 2. vswhere.
    pf86 = os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.exists():
        try:
            proc = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-find",
                    r"VC\Auxiliary\Build\vcvars64.bat",
                ],
                capture_output=True,
                text=True,
            )
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line and Path(line).exists():
                    return line
        except OSError:
            pass

    # 3. Hardcoded fallback.
    if Path(_VCVARS_FALLBACK).exists():
        return _VCVARS_FALLBACK

    raise SystemExit("could not locate vcvars64.bat; set VCVARS")


def _cuda_title(folder: Path) -> str:
    """Read the title from the first `// TITLE: ...` line of harness.cu."""
    harness = folder / "harness.cu"
    try:
        with harness.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("// TITLE:"):
                    return s[len("// TITLE:"):].strip()
    except OSError:
        pass
    return folder.name


def _run_cuda(folder: Path) -> bool:
    """Build kernel.cu + harness.cu with nvcc and run the binary, relaying
    its stdout (which already prints [TODO]/[PASS]/[FAIL]/[PERF])."""
    title = _cuda_title(folder)
    print(f"\n=== {folder.name}: {title} ===")

    build_dir = folder / "build"
    build_dir.mkdir(exist_ok=True)
    exe = build_dir / "run.exe"

    vcvars = _find_vcvars()
    harness_cu = folder / "harness.cu"
    kernel_cu = folder / "kernel.cu"

    # Single `cmd /c` string. The inner command contains quoted paths, so the
    # whole thing is wrapped in an extra pair of quotes (cmd /c "...") and the
    # inner quotes are escaped for the Python string.
    inner = (
        f'"{vcvars}" >nul && '
        f'nvcc -O3 -std=c++17 -arch=sm_120 '
        f'"{harness_cu}" "{kernel_cu}" -o "{exe}"'
    )
    cmd = f'cmd /c "{inner}"'

    # Always recompile: nvcc is fast for single-file programs and this avoids
    # running a stale binary.
    build = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if build.returncode != 0:
        # nvcc errors land in BOTH stdout and stderr; show the tail of both.
        combined = (build.stdout or "") + (build.stderr or "")
        tail = "\n".join(combined.splitlines()[-25:])
        print("[FAIL] build error:")
        if tail.strip():
            print(tail)
        else:
            print("(no compiler output) vcvars64.bat likely failed -- check your "
                  "VS BuildTools install or set VCVARS to the correct path")
        return False

    # Run the freshly built binary, relaying its stdout verbatim. Use Popen so
    # a hung binary (e.g. a deadlocked kernel) can be killed explicitly --
    # otherwise the exe stays alive, keeps build\run.exe locked, and the next
    # rebuild fails confusingly.
    proc = subprocess.Popen(
        [str(exe)], cwd=str(folder),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        out, err = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        # Kill the whole process tree so the exe is really gone and the file
        # handle on run.exe is released before the next build.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
        proc.wait()
        print("[FAIL] timed out after 120s -- deadlocked kernel? "
              "(classic cause: __syncthreads() behind a divergent branch)")
        return False
    print(out, end="")
    if proc.returncode != 0:
        # A nonzero exit means a CUDA_CHECK abort (or similar). Surface stderr.
        if err:
            print(err, end="")
        return False
    return "[PASS]" in out


def run_one(folder: Path) -> bool:
    # Dispatch BEFORE loading spec.py: CUDA folders have no spec.py.
    if (folder / "kernel.cu").exists():
        return _run_cuda(folder)

    if torch is None:
        print(f"\n=== {folder.name} ===")
        print("[ERROR] torch is required for Triton exercises but is not installed.")
        return False
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available to PyTorch.")

    spec = _load(folder / "spec.py")
    title = getattr(spec, "TITLE", folder.name)
    print(f"\n=== {folder.name}: {title} ===")

    try:
        kernel = _load(folder / "kernel.py")
    except Exception as e:
        print(f"[ERROR] kernel.py failed to import: {e!r}")
        return False

    entry = getattr(kernel, spec.ENTRYPOINT, None)
    if entry is None:
        print(f"[ERROR] kernel.py has no function '{spec.ENTRYPOINT}'")
        return False

    inputs = spec.make_inputs()

    # Compute the reference BEFORE running the kernel: a buggy kernel that
    # mutates its inputs in place must not be able to corrupt the reference
    # answer and spuriously "pass".
    ref = spec.reference(*inputs)

    # --- correctness ---
    try:
        out = entry(*inputs)
    except NotImplementedError as e:
        print(f"[TODO] Not implemented yet -- go write the kernel. ({e})")
        return False
    except Exception as e:
        print(f"[FAIL] kernel raised: {e!r}")
        return False

    tol = getattr(spec, "TOL", {"atol": 1e-2, "rtol": 1e-2})
    try:
        for o, r in zip(_as_tuple(out), _as_tuple(ref)):
            torch.testing.assert_close(o, r, **tol)
    except AssertionError as e:
        print("[FAIL] wrong answer:")
        print("       " + str(e).strip().splitlines()[0])
        return False
    print("[PASS] correct")

    # --- performance ---
    metric = getattr(spec, "METRIC", "none")
    if metric == "none":
        return True

    ms = _bench(lambda: entry(*inputs))
    sec = ms * 1e-3
    if metric == "bandwidth":
        gbps = spec.bytes_moved(*inputs) / sec / 1e9
        peak = _peak_bw_gbps()
        pct = f"  ({100 * gbps / peak:.0f}% of ~{peak:.0f} GB/s peak)" if peak else ""
        print(f"[PERF] {ms:.3f} ms   {gbps:.0f} GB/s{pct}")
    elif metric == "flops":
        tflops = spec.flops(*inputs) / sec / 1e12
        print(f"[PERF] {ms:.3f} ms   {tflops:.1f} TFLOP/s")

    # Compare against the torch reference for a sense of how close you are.
    try:
        ref_ms = _bench(lambda: spec.reference(*inputs))
        print(f"[REF]  torch does it in {ref_ms:.3f} ms  ({ref_ms / ms:.2f}x your time)")
    except Exception:
        pass
    return True


def watch(folder: Path) -> None:
    target = folder / "kernel.cu" if (folder / "kernel.cu").exists() else folder / "kernel.py"
    print(f"Watching {target} -- save to re-run, Ctrl+C to stop.")
    last = 0.0
    while True:
        try:
            mtime = target.stat().st_mtime
            if mtime != last:
                last = mtime
                run_one(folder)
                print("\n... waiting for next save ...")
            time.sleep(0.4)
        except KeyboardInterrupt:
            print("\nbye")
            return
        except (FileNotFoundError, PermissionError, OSError):
            # On Windows an editor saving kernel.py can briefly make stat()
            # raise (file locked/replaced). Skip this poll and retry next one.
            time.sleep(0.4)
            continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exercise", nargs="?", help="exercise prefix, e.g. e01")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    # NOTE: the torch / CUDA-availability check now lives inside run_one's
    # Triton branch, so pure-CUDA (kernel.cu) exercises run with only
    # nvcc + MSVC -- no torch required.

    if args.all:
        folders = sorted(d for d in EXDIR.iterdir() if d.is_dir())
        ok = sum(run_one(f) for f in folders)
        print(f"\n{ok}/{len(folders)} exercises passing.")
        return

    if not args.exercise:
        raise SystemExit("Give an exercise prefix (e.g. e01) or --all.")

    folder = _find_exercise(args.exercise)
    if args.watch:
        watch(folder)
    else:
        run_one(folder)


if __name__ == "__main__":
    main()
