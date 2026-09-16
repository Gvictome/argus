#!/usr/bin/env python3
"""
Render ARGUS documentation to PDF.

Markdown is the source of truth and stays in the repo; this produces
handout copies for a poster session, an advisor, or a submission.

Uses headless Chrome, which is already on every machine that runs the
demo, rather than adding a print stack (weasyprint needs GTK on Windows;
wkhtmltopdf is unmaintained). The intermediate HTML is written next to
the PDF so a failed render can be inspected.

Usage:
    python scripts/build_docs_pdf.py
    python scripts/build_docs_pdf.py --docs PROPOSAL SUBSYSTEMS
    python scripts/build_docs_pdf.py --out-dir handouts

Exit codes:
    0  all requested PDFs built
    1  one or more failed
    2  no usable Chrome found
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR  # noqa: E402

DOCS_DIR = BASE_DIR / "docs"

DEFAULT_DOCS = [
    ("QUICKSTART", "ARGUS Quick Start", "Every command, labelled PI or LAPTOP"),
    ("OVERVIEW", "ARGUS in One Pass", "What each part does, and what runs it"),
    ("DEMO_RUNBOOK", "ARGUS Demo Runbook", "Running the demo end to end"),
    ("SUBSYSTEMS", "ARGUS Subsystems", "How the system fits together"),
    ("PROPOSAL", "ARGUS Project Proposal", "Revision 2 — Senior Design 2026"),
    ("DEMO_SCRIPT", "ARGUS Demo Script", "Walkthrough, queries, codebase"),
]

# Plain-language handouts, one topic per PDF so a reader can be given the
# one part they need. Rendered with --guides, usually into a shared folder
# rather than the repo.
GUIDES = [
    ("guides/01-what-is-argus", "1 · What ARGUS Is", "The system in plain language"),
    ("guides/02-set-up-the-pi", "2 · Setting Up the Pi", "From a bare Pi to a running camera"),
    ("guides/03-run-the-demo", "3 · Running the Demo", "What to run, and what to say"),
    ("guides/04-the-dashboards", "4 · The Dashboards", "Both screens, and what they show"),
    ("guides/05-what-it-detects", "5 · What It Detects", "The four classes, and how to test them"),
    ("guides/06-federated-learning", "6 · How the Learning Works", "Sharing what is learned, not what is seen"),
    ("guides/07-when-things-break", "7 · When Things Break", "Every failure we hit, and the fix"),
    ("guides/08-for-developers", "8 · For Developers", "Layout, decisions, and open work"),
]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
]

# Print styling. Deliberately conservative: this is read on paper and in
# a PDF viewer, so it targets legibility and page breaks, not screen
# aesthetics. No web fonts -- a font fetch that fails silently on an
# offline machine would reflow every table.
CSS = """
@page { size: Letter; margin: 20mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #16191d; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.cover { border-bottom: 2.5pt solid #16191d; padding-bottom: 12pt; margin-bottom: 20pt; }
.cover .kicker {
  font-size: 8pt; letter-spacing: .16em; text-transform: uppercase;
  color: #6b7480; margin-bottom: 7pt; font-weight: 600;
}
.cover h1 { font-size: 23pt; margin: 0 0 5pt; letter-spacing: -.01em; }
.cover .sub { font-size: 10.5pt; color: #5b636e; margin: 0; }

h1, h2, h3, h4 { page-break-after: avoid; break-after: avoid; line-height: 1.25; }
h1 { font-size: 17pt; margin: 20pt 0 7pt; }
h2 {
  font-size: 13.5pt; margin: 17pt 0 6pt;
  padding-bottom: 3pt; border-bottom: .75pt solid #d3d8de;
}
h3 { font-size: 11.5pt; margin: 13pt 0 4pt; }
p { margin: 0 0 8pt; orphans: 3; widows: 3; }

table {
  border-collapse: collapse; width: 100%; margin: 9pt 0;
  font-size: 9pt; page-break-inside: avoid; break-inside: avoid;
}
th {
  background: #eef1f4; text-align: left; font-weight: 600;
  padding: 5pt 7pt; border: .75pt solid #ccd2d9; font-size: 8.5pt;
  letter-spacing: .03em;
}
td { padding: 5pt 7pt; border: .75pt solid #d8dde3; vertical-align: top; }

pre {
  background: #f4f6f8; border: .75pt solid #d8dde3; border-radius: 2pt;
  padding: 8pt 10pt; font-size: 8.5pt; line-height: 1.42;
  overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;
  page-break-inside: avoid; break-inside: avoid;
}
code {
  font-family: "Cascadia Mono", Consolas, "SF Mono", monospace;
  font-size: 9pt; background: #eef1f4; padding: .5pt 3pt; border-radius: 2pt;
}
pre code { background: none; padding: 0; font-size: inherit; }

blockquote {
  margin: 9pt 0; padding: 2pt 0 2pt 12pt;
  border-left: 2.5pt solid #b8c0c9; color: #444b54;
}
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
hr { border: none; border-top: .75pt solid #d3d8de; margin: 16pt 0; }
a { color: #1f4e79; text-decoration: none; }
strong { font-weight: 600; }
"""

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head><body>
<div class="cover">
  <div class="kicker">ARGUS &middot; Senior Design 2026</div>
  <h1>{title}</h1>
  <p class="sub">{subtitle}</p>
</div>
{body}
</body></html>"""


def find_chrome() -> str:
    for name in ("chrome", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    return ""


def render(md_path: Path, title: str, subtitle: str, out_dir: Path, chrome: str) -> bool:
    import markdown

    if not md_path.exists():
        print(f"  [FAIL] {md_path.name} not found")
        return False

    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    html_path = out_dir / f"{md_path.stem}.html"
    pdf_path = out_dir / f"{md_path.stem}.pdf"
    html_path.write_text(
        HTML.format(title=title, subtitle=subtitle, css=CSS, body=body),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            chrome, "--headless", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        capture_output=True, timeout=180,
    )

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        print(f"  [FAIL] {md_path.name}: Chrome produced no PDF")
        err = result.stderr.decode("utf-8", "ignore").strip()
        if err:
            print(f"         {err.splitlines()[-1][:120]}")
        return False

    kb = pdf_path.stat().st_size / 1024
    print(f"  [ ok ] {pdf_path.name}  ({kb:.0f} KB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Render ARGUS docs to PDF.")
    ap.add_argument("--docs", nargs="*", help="Doc stems, e.g. PROPOSAL.")
    ap.add_argument("--guides", action="store_true",
                    help="Render the plain-language guide set instead.")
    ap.add_argument("--out-dir", default="docs/pdf",
                    help="Output folder; an absolute path is used as given.")
    args = ap.parse_args()

    chrome = find_chrome()
    if not chrome:
        print("[FAIL] No Chrome or Chromium found.")
        print("       Checked PATH and the usual install locations.")
        return 2
    print(f"Using: {chrome}\n")

    out_dir = BASE_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source = GUIDES if args.guides else DEFAULT_DOCS
    wanted = source
    if args.docs:
        chosen = {d.upper().removesuffix(".MD") for d in args.docs}
        wanted = [d for d in source if d[0].upper() in chosen]
        for name in sorted(chosen - {d[0].upper() for d in source}):
            wanted.append((name, name.replace("_", " ").title(), ""))

    results = [
        render(DOCS_DIR / f"{stem}.md", title, subtitle, out_dir, chrome)
        for stem, title, subtitle in wanted
    ]

    print()
    if all(results):
        print(f"All PDFs written to {out_dir}")
        return 0
    print(f"{results.count(False)} of {len(results)} failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
