"""
One command: build, execute, and export the ARGUS simulation report.

    python sim/report/run_report.py                 # full run, ~25 min (5 real FPS runs)
    python sim/report/run_report.py --reuse-fps     # reuse cached FPS runs, ~15 min
    python sim/report/run_report.py --copy-to "C:/Users/colab/Dropbox/gio/argus/argus"

Writes, beside this file:
    ARGUS_Simulation_Results.ipynb   executed notebook (code collapsed)
    ARGUS_Simulation_Results.html    no code, for sharing
    ARGUS_Simulation_Results.pdf     printed from the HTML by headless Chrome

Close other heavy apps first: the frame-rate section measures this PC, and
background load shows up as a training cost.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "ARGUS_Simulation_Results"
CHROME_CANDIDATES = (
    shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("msedge"),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def step(label: str, cmd: list, env: dict = None) -> None:
    print(f"\n==> {label}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=HERE, env=env)
    if result.returncode != 0:
        sys.exit(f"FAILED: {label} (exit {result.returncode})")
    print(f"    done in {time.perf_counter() - t0:.0f}s")


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    sys.exit("No Chrome or Edge found for the PDF step. The .ipynb and .html were still written.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse-fps", action="store_true", help="reuse cached frame-rate runs from sim/_run")
    ap.add_argument("--copy-to", help="also copy the .ipynb/.html/.pdf here, dated")
    args = ap.parse_args()

    py = sys.executable
    env = {**os.environ, "ARGUS_REFRESH_FPS": "0" if args.reuse_fps else "1"}
    nb = f"{NAME}.ipynb"

    step("build notebook", [py, "build_notebook.py"])
    step("execute (long)", [py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                            "--inplace", "--ExecutePreprocessor.timeout=2400", nb], env)
    step("export html", [py, "-m", "jupyter", "nbconvert", "--to", "html", "--no-input", nb])
    step("print pdf", [find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                       f"--print-to-pdf={HERE / (NAME + '.pdf')}", (HERE / f"{NAME}.html").as_uri()])

    if args.copy_to:
        dest = Path(args.copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        for ext in ("ipynb", "html", "pdf"):
            shutil.copy2(HERE / f"{NAME}.{ext}", dest / f"{NAME}_{date.today()}.{ext}")
        print(f"\ncopied to {dest}")
    print(f"\nreport: {HERE / (NAME + '.pdf')}")


if __name__ == "__main__":
    main()
