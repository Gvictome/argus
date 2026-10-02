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
    python scripts/build_docs_pdf.py --guides
    python scripts/build_docs_pdf.py --master          # everything, one PDF
    python scripts/build_docs_pdf.py --out-dir handouts

Exit codes:
    0  all requested PDFs built
    1  one or more failed
    2  no usable Chrome found
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
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

# The master reference: everything a new team member needs, in reading
# order, as one PDF. Paths are relative to the repo root because the
# simulator's notes live outside docs/. Each entry is (path, part).
MASTER = [
    ("docs/reference/01-system-overview.md", "Part I · Technical reference"),
    ("docs/reference/02-how-to-run.md", "Part I · Technical reference"),
    ("docs/reference/03-computer-vision.md", "Part I · Technical reference"),
    ("docs/reference/04-threat-classification.md", "Part I · Technical reference"),
    ("docs/reference/05-federated-learning.md", "Part I · Technical reference"),
    ("docs/reference/06-api-data-security.md", "Part I · Technical reference"),
    ("docs/reference/07-scripts-and-tools.md", "Part I · Technical reference"),
    ("docs/reference/08-configuration.md", "Part I · Technical reference"),
    ("docs/reference/09-tests.md", "Part I · Technical reference"),
    ("docs/reference/10-status-and-gaps.md", "Part I · Technical reference"),
] + [
    (f"docs/{stem}.md", "Part II · The guides") for stem, _, _ in GUIDES
] + [
    ("docs/QUICKSTART.md", "Part III · Quick start"),
    ("sim/README.md", "Appendix · The simulator"),
]

# Additions for the master build only. Chrome draws the page number from
# the @page margin box; older Chrome ignores the rule and prints no number,
# which is harmless.
MASTER_CSS = """
@page { @bottom-right { content: counter(page); font-size: 8pt; color: #6b7480; } }
.doc { break-before: page; }
.doc .part {
  font-size: 7.5pt; letter-spacing: .16em; text-transform: uppercase;
  color: #6b7480; font-weight: 600; margin-bottom: 4pt;
}
.toc { list-style: none; padding-left: 0; }
.toc > li { margin: 0 0 3pt; }
.toc .part-head {
  margin-top: 12pt; font-size: 8pt; letter-spacing: .14em;
  text-transform: uppercase; color: #6b7480; font-weight: 600;
}
.toc ul { list-style: none; padding-left: 14pt; margin: 2pt 0 4pt; font-size: 9pt; }
.toc ul li { margin: 0 0 1pt; color: #444b54; }
"""

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


MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "attr_list", "toc"]


def print_pdf(chrome: str, html_path: Path, pdf_path: Path) -> bool:
    """Print `html_path` to `pdf_path` with headless Chrome.

    Chrome exits 0 even when it cannot write its output, so printing
    straight over a PDF that a viewer holds open leaves the old file in
    place and looks like success. Print to a fresh temporary file instead,
    then copy it over the target, where a lock fails loudly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / pdf_path.name
        result = subprocess.run(
            [
                chrome, "--headless", "--disable-gpu", "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={fresh}",
                html_path.as_uri(),
            ],
            capture_output=True, timeout=300,
        )
        if not fresh.exists() or fresh.stat().st_size == 0:
            print(f"  [FAIL] {pdf_path.name}: Chrome produced no PDF")
            err = result.stderr.decode("utf-8", "ignore").strip()
            if err:
                print(f"         {err.splitlines()[-1][:120]}")
            return False
        try:
            shutil.copyfile(fresh, pdf_path)
        except PermissionError:
            print(f"  [FAIL] {pdf_path} is open in another program. Close it and run again.")
            return False

    kb = pdf_path.stat().st_size / 1024
    print(f"  [ ok ] {pdf_path.name}  ({kb:.0f} KB)")
    return True


def render(md_path: Path, title: str, subtitle: str, out_dir: Path, chrome: str) -> bool:
    import markdown

    if not md_path.exists():
        print(f"  [FAIL] {md_path.name} not found")
        return False

    body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=MD_EXTENSIONS)
    html_path = out_dir / f"{md_path.stem}.html"
    html_path.write_text(
        HTML.format(title=title, subtitle=subtitle, css=CSS, body=body),
        encoding="utf-8",
    )
    return print_pdf(chrome, html_path, out_dir / f"{md_path.stem}.pdf")


def _doc_anchor(path: str) -> str:
    return "doc-" + Path(path).stem.lower()


def _master_section(rel_path: str, part: str) -> tuple:
    """One document as HTML, plus its title and H2s for the contents page.

    Links between reference files (`05-federated-learning.md`) point at
    files that do not exist inside a single PDF, so they are rewritten to
    in-document anchors. Heading ids are prefixed per document, because
    every guide has its own "Step 1" and duplicate ids send every link to
    the first one.
    """
    import markdown

    md = markdown.Markdown(extensions=MD_EXTENSIONS)
    text = (BASE_DIR / rel_path).read_text(encoding="utf-8")
    # House style for this handout: no em dashes. Source files keep theirs.
    text = text.replace(" — ", " - ").replace("—", " - ")
    anchor = _doc_anchor(rel_path)
    body = md.convert(text)
    body = re.sub(r'id="([^"]+)"', rf'id="{anchor}-\1"', body)
    body = re.sub(r'href="#([^"]+)"', rf'href="#{anchor}-\1"', body)
    body = re.sub(
        r'href="(?:\.\./)*(?:[\w./-]*/)?([\w.-]+)\.md(?:#[^"]*)?"',
        lambda m: f'href="#doc-{m.group(1).lower()}"', body,
    )

    tokens = getattr(md, "toc_tokens", [])
    title = tokens[0]["name"] if tokens else Path(rel_path).stem
    subs = [c["name"] for c in (tokens[0]["children"] if tokens else [])]
    html = (f'<section class="doc" id="{anchor}">'
            f'<div class="part">{part}</div>{body}</section>')
    return title, subs, html


def build_master(out_dir: Path, chrome: str) -> bool:
    """Every reference section, guide and quick start as one PDF."""
    missing = [p for p, _ in MASTER if not (BASE_DIR / p).exists()]
    if missing:
        for p in missing:
            print(f"  [FAIL] {p} not found")
        return False

    toc, sections, last_part = [], [], None
    for rel_path, part in MASTER:
        title, subs, html = _master_section(rel_path, part)
        sections.append(html)
        if part != last_part:
            toc.append(f'<li class="part-head">{part}</li>')
            last_part = part
        items = "".join(f"<li>{s}</li>" for s in subs)
        toc.append(f'<li><a href="#{_doc_anchor(rel_path)}">{title}</a>'
                   + (f"<ul>{items}</ul>" if items else "") + "</li>")

    stamp = date.today().isoformat()
    body = (f'<h2>Contents</h2><ul class="toc">{"".join(toc)}</ul>'
            + "".join(sections))
    html_path = out_dir / "ARGUS_Master_Reference.html"
    html_path.write_text(
        HTML.format(
            title="ARGUS Master Reference",
            subtitle=(f"How it works, how to run it, every important script, "
                      f"and the tests. Built {stamp} from docs/ on the "
                      f"feat/threat-detection branch."),
            css=CSS + MASTER_CSS, body=body,
        ),
        encoding="utf-8",
    )
    return print_pdf(chrome, html_path, out_dir / "ARGUS_Master_Reference.pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description="Render ARGUS docs to PDF.")
    ap.add_argument("--docs", nargs="*", help="Doc stems, e.g. PROPOSAL.")
    ap.add_argument("--guides", action="store_true",
                    help="Render the plain-language guide set instead.")
    ap.add_argument("--master", action="store_true",
                    help="Render the reference, guides and quick start as one PDF.")
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

    if args.master:
        ok = build_master(out_dir, chrome)
        print()
        print(f"Master PDF written to {out_dir}" if ok else "Master PDF failed")
        return 0 if ok else 1

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
