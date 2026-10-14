#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Re-render all Appendix E caption figures (MHCap Figures 3-12, WESADCap Figures 13-27) as vector PDFs
without dates in the figure titles.

The paper's PNGs were produced by scripts/render_caption_visualization.py on branch KarlDeck/Visualization:
  - MHC   figures: commit 340773f ("visualizer", figures/MHC/*.png, shipped as the paper's "Masked Set")
  - WESAD figures: commit a4f5c5b ("WESAD added", figures/WESAD/wesad_auto_row_*.png)
This script exports those commits with `git archive` (the repo itself is not modified), runs the same
render commands as figures/generate_images.sh, and restores the MHC paper layout before exporting:
  - vector PDF with embedded TrueType fonts instead of a 220-dpi PNG,
  - the date is dropped from the title (OpenMHC / WESAD dates are shifted and may change),
  - MHC titles drop the "MHC captioning pipeline " prefix exactly like the Masked Set did.

Usage, from the repo root (Python >= 3.10 with requirements.txt installed):
    python scripts/render_appendix_figures.py --out appendix_pdfs \\
        --mhc-daily-dir /path/to/daily-mhc \\
        --wesad-arrow-dir /path/to/wesad-export
The commits above are fetched from origin/KarlDeck/Visualization automatically if they are missing.
Either dataset flag can be omitted to render only the other half.

Output file names match the paper (sections/figures/Appendix/Masked Set/masked_*.pdf and
sections/figures/Appendix/WESAD/wesad_auto_row_*.pdf), so only the extension in 06_appendix.tex changes.
"""
from __future__ import annotations

import argparse
import os
import re
import runpy
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

MHC_COMMIT = "340773f"
WESAD_COMMIT = "a4f5c5b"
CARDS = ["--include-statistical", "--caption-types", "statistical", "structural", "semantic", "cross_channel",
         "--max-cards", "8"]

# (output name, render_caption_visualization.py args) -- from figures/generate_images.sh at 340773f.
MHC_FIGURES = [
    ("masked_sleep-figure1", ["--row-index", "3402", "--example", "auto", *CARDS]),
    ("masked_sleep-figure2", ["--row-index", "3662", "--example", "auto", *CARDS]),
    ("masked_sleep-figure3", ["--row-index", "3724", "--example", "auto", *CARDS]),
    ("masked_cycling", ["--row-index", "3402", "--example", "auto", "--channels", "WHR", "WSteps", "WDist", "WEnergy", "Cycle", *CARDS]),
    ("masked_elliptical", ["--row-index", "2927", "--example", "auto", "--channels", "WHR", "WSteps", "WDist", "WEnergy", "Elliptical", *CARDS]),
    ("masked_running", ["--row-index", "8229", "--example", "auto", "--channels", "WHR", "WSteps", "WDist", "WEnergy", "Run", *CARDS]),
    ("masked_hiit", ["--row-index", "8574", "--example", "auto", "--channels", "WHR", "WSteps", "WDist", "WEnergy", "HIIT", *CARDS]),
    ("masked_walking1", ["--row-index", "3463", "--example", "walking", *CARDS]),
    ("masked_walking2", ["--row-index", "3402", "--example", "walking", *CARDS]),
    ("masked_yoga", ["--row-index", "8844", "--example", "auto", "--channels", "WHR", "WSteps", "WDist", "WEnergy", "Yoga", *CARDS]),
]
# From figures/generate_images.sh at a4f5c5b.
WESAD_FIGURES = [(f"wesad_auto_row_{i}", ["--dataset", "wesad", "--row-index", str(i), "--example", "auto"])
                 for i in range(15)]

MHC_TITLE_PREFIX = "MHC captioning pipeline "
# The committed figures/MHC/*.png have both panels' top edge at 0.897 of the figure height; the code at
# 340773f says 0.932, which makes the panel headers collide with the subtitle.
MHC_PANEL_TOP = 0.897
# The committed PNGs use these per-stage line limits. The source saved in the
# same commit removed them, producing much taller figures and different text
# truncation. Restore the limits for both height estimation and card drawing.
MHC_CARD_LINES = {"statistical": 6, "structural": 7, "semantic": 6, "cross_channel": 7}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def strip_date(title: str) -> str:
    return " | ".join(part for part in title.split(" | ") if not DATE_RE.match(part.strip()))


def mask_title_prefix(fig, prefix: str) -> None:
    """Drop the title prefix but keep the rest at its original x position (as in the Masked Set)."""
    title = next(t for t in fig.texts if t.get_text().startswith(prefix))
    probe = fig.text(*title.get_position(), prefix, fontsize=title.get_fontsize(),
                     weight=title.get_weight(), ha="left", va="top")
    width = probe.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
    probe.remove()
    x, y = title.get_position()
    title.set_text(title.get_text()[len(prefix):])
    title.set_position((x + width, y))


def run_one(checkout: str, kind: str, out_pdf: str, script_args: list[str]) -> None:
    """Runs inside a subprocess with cwd=checkout: patch the plot function, then run the repo script."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType -> selectable, searchable text
    sys.path.insert(0, checkout)
    import visualizer

    if kind == "mhc":
        original_wrap = visualizer._wrapped_stage_card_lines

        def paper_card_lines(caption_type, annotations, *, max_lines=None):
            return original_wrap(caption_type, annotations, max_lines=MHC_CARD_LINES[caption_type])

        visualizer._wrapped_stage_card_lines = paper_card_lines

    original = visualizer.plot_caption_pipeline_row

    def plot_to_pdf(row, save_path=None, *, title=None, **kwargs):
        fig = original(row, None, title=strip_date(title) if title else title, **kwargs)
        if kind == "mhc":
            mask_title_prefix(fig, MHC_TITLE_PREFIX)
            for ax in fig.axes:
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0, pos.width, MHC_PANEL_TOP - pos.y0])
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        return None

    visualizer.plot_caption_pipeline_row = plot_to_pdf
    sys.argv = ["render_caption_visualization.py", *script_args, "--out", out_pdf]
    runpy.run_path(str(Path(checkout) / "scripts" / "render_caption_visualization.py"), run_name="__main__")


def ensure_commit(repo: Path, commit: str) -> None:
    present = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
                             capture_output=True).returncode == 0
    if not present:
        print(f"Fetching KarlDeck/Visualization for commit {commit} ...")
        subprocess.run(["git", "-C", str(repo), "fetch", "origin", "KarlDeck/Visualization"], check=True)


def export_commit(repo: Path, commit: str, dest: Path) -> Path:
    target = dest / commit
    if not target.exists():
        ensure_commit(repo, commit)
        target.mkdir(parents=True)
        archive = subprocess.run(["git", "-C", str(repo), "archive", commit], check=True, capture_output=True).stdout
        tar_path = dest / f"{commit}.tar"
        tar_path.write_bytes(archive)
        with tarfile.open(tar_path) as tar:
            tar.extractall(target)
    return target


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_run":
        run_one(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="TimeCAP git checkout (default: this repo).")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the PDFs.")
    parser.add_argument("--mhc-daily-dir", type=Path, help="Daily MHC HuggingFace dataset dir (MHC_DATASET_DIR).")
    parser.add_argument("--wesad-arrow-dir", type=Path, help="Directory with WESAD recordings_*.arrow shards.")
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "timecap_appendix_figures")
    args = parser.parse_args()
    if not (args.mhc_daily_dir or args.wesad_arrow_dir):
        parser.error("pass --mhc-daily-dir and/or --wesad-arrow-dir")

    (args.out / "Masked Set").mkdir(parents=True, exist_ok=True)
    (args.out / "WESAD").mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.mhc_daily_dir:
        checkout = export_commit(args.repo.expanduser(), MHC_COMMIT, args.work_dir)
        env = {"MHC_DATASET_DIR": str(args.mhc_daily_dir.expanduser().resolve())}
        jobs += [(checkout, "mhc", args.out / "Masked Set" / f"{name}.pdf", a, env) for name, a in MHC_FIGURES]
    if args.wesad_arrow_dir:
        checkout = export_commit(args.repo.expanduser(), WESAD_COMMIT, args.work_dir)
        arrow = ["--wesad-arrow-dir", str(args.wesad_arrow_dir.expanduser().resolve())]
        jobs += [(checkout, "wesad", args.out / "WESAD" / f"{name}.pdf", a + arrow, {}) for name, a in WESAD_FIGURES]

    failed = []
    for checkout, kind, out_pdf, script_args, env in jobs:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_run", str(checkout), kind,
               str(out_pdf.resolve()), *script_args]
        result = subprocess.run(cmd, cwd=checkout, env={**os.environ, **env, "MPLBACKEND": "Agg"},
                                capture_output=True, text=True)
        status = "ok" if result.returncode == 0 and out_pdf.exists() else "FAILED"
        print(f"[{status}] {out_pdf.relative_to(args.out)}")
        if status != "ok":
            failed.append(out_pdf.name)
            print(result.stderr[-2000:])
    print(f"{len(jobs) - len(failed)}/{len(jobs)} figures written to {args.out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
