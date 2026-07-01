"""Export the lecture notebooks to a static WASM site in docs/ (GitHub Pages).

    python build_site.py

Mirrors the info-theory-course build: each notebook -> docs/<module>/index.html
via `marimo export html-wasm --mode run --show-code`. Only LECTURE notebooks are
exported; the exercise harness is local-only (it needs a real GPU).

Note: notebooks are pyodide-safe (numpy/scipy/matplotlib). They *show* kernel code
in markdown but never import torch/triton, so the WASM build runs fine in-browser.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks"
DOCS = ROOT / "docs"


def module_name(path: Path) -> str:
    # 0b_execution_model.py -> 0b_execution_model ; home.py -> the site root
    return path.stem


def dedupe_assets() -> None:
    """Collapse every per-notebook docs/<module>/assets/ into one shared docs/assets/.

    marimo's html-wasm export copies the full (content-hashed) asset bundle into
    each notebook's folder -- ~25 MB x 27 exported files (26 lectures + home).
    The filenames are content
    hashes, so identical names are identical files; we keep one copy in
    docs/assets/ and rewrite each module's index.html (./assets/ -> ../assets/).
    home (docs/index.html) already sits next to docs/assets/, so it is left alone.
    """
    shared = DOCS / "assets"
    shared.mkdir(exist_ok=True)
    for sub in sorted(DOCS.iterdir()):
        if not sub.is_dir() or sub.name == "assets":
            continue
        # marimo bundles the repo's CLAUDE.md into each export dir -- strip it.
        stray = sub / "CLAUDE.md"
        if stray.exists():
            stray.unlink()
        mod_assets = sub / "assets"
        if mod_assets.is_dir():
            for f in mod_assets.rglob("*"):
                if f.is_file():
                    dest = shared / f.relative_to(mod_assets)
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dest)
            shutil.rmtree(mod_assets)
        index = sub / "index.html"
        if index.exists():
            index.write_text(
                index.read_text(encoding="utf-8").replace("./assets/", "../assets/"),
                encoding="utf-8",
            )
    print(f"Deduped per-notebook assets into {shared}")


def main() -> None:
    if not NB.exists():
        sys.exit(f"no notebooks/ dir at {NB}")
    DOCS.mkdir(exist_ok=True)

    notebooks = sorted(NB.glob("*.py"))
    if not notebooks:
        sys.exit("no notebooks to build")

    for nb in notebooks:
        name = module_name(nb)
        out_dir = DOCS if name == "home" else DOCS / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "index.html"
        print(f"  {nb.name} -> {out.relative_to(ROOT)}")
        subprocess.run(
            [sys.executable, "-m", "marimo", "export", "html-wasm",
             str(nb), "-o", str(out), "--mode", "run", "--show-code"],
            check=True,
        )

    dedupe_assets()

    # GitHub Pages: don't run Jekyll over the export.
    (DOCS / ".nojekyll").write_text("")
    print(f"\nBuilt {len(notebooks)} notebooks into {DOCS}")
    print("Commit docs/ and enable GitHub Pages on the docs/ folder to publish.")


if __name__ == "__main__":
    main()
