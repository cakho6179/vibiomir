"""Render a Markdown document (tables, code, mermaid diagrams) to PDF with headless Chrome / Edge.

    python scripts/release/md_to_pdf.py docs/THUYET_MINH_SAN_PHAM.md release/THUYET_MINH_SAN_PHAM.pdf

Needs the `markdown` package, a Chromium browser, and internet once (mermaid.js from jsdelivr) to draw the diagrams.
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser", "msedge"]
CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; margin: 0 6mm; }
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 14pt; margin-top: 18px; border-bottom: 1px solid #aaa; page-break-after: avoid; }
h3 { font-size: 12pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9pt; page-break-inside: auto; }
th, td { border: 1px solid #999; padding: 3px 6px; vertical-align: top; }
th { background: #eee; }
tr { page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.5pt; background: #f3f3f3; padding: 0 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6px; font-size: 8pt; white-space: pre-wrap; }
pre code { background: none; }
.mermaid { text-align: center; background: none; border: none; page-break-inside: avoid; }
"""


LIST = re.compile(r"^\s*([-*]|\d+\.)\s")


def loosen_lists(text: str) -> str:
    """Python-Markdown needs a blank line before a list that follows a paragraph (GitHub does not)."""
    out, fenced = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        prev = out[-1] if out else ""
        if not fenced and LIST.match(line) and prev.strip() and not LIST.match(prev) and not prev.startswith("  "):
            out.append("")
        out.append(line)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    text = args.src.read_text(encoding="utf-8")
    diagrams: list[str] = []

    def stash(m: re.Match) -> str:
        diagrams.append(m.group(1))
        return f"\n\nMERMAIDBLOCK{len(diagrams) - 1}\n\n"

    text = re.sub(r"```mermaid\n(.*?)```", stash, text, flags=re.S)
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    for i, d in enumerate(diagrams):
        body = body.replace(f"<p>MERMAIDBLOCK{i}</p>", f'<pre class="mermaid">{html.escape(d)}</pre>')
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>{args.src.stem}</title>
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script></head><body>{body}
<script>mermaid.initialize({{startOnLoad: true, theme: 'neutral', flowchart: {{useMaxWidth: true}}}});</script>
</body></html>"""
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if browser is None:
        raise SystemExit("no Chrome / Edge found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page_file = Path(tmp) / "page.html"
        page_file.write_text(page, encoding="utf-8")
        (args.out.with_suffix(".html")).write_text(page, encoding="utf-8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=20000", f"--print-to-pdf={args.out.resolve()}",
                        page_file.resolve().as_uri()], check=True, capture_output=True, timeout=180)
    print(f"-> {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
