"""Gom mọi file zip bài nộp về một chỗ: vibiomir/submissions/ + manifest.

    python scripts/submit/collect_submissions.py            # di chuyển zip mới, cập nhật manifest
    python scripts/submit/collect_submissions.py --dry-run  # chỉ in kế hoạch

Nguồn quét: runs/submissions/*.zip và kaggle/outputs/*/*.zip. File được DI CHUYỂN (không xoá), bỏ tiền tố
"submission_"; trùng tên khác nội dung thì thêm tên kernel. Điểm LB lấy từ bảng A của EXPERIMENTS.md (nguồn sự
thật); trạng thái/ghi chú sửa tay trong submissions/manifest.json được giữ nguyên qua các lần chạy.
Bài đã có điểm chuyển sang temp_submission/, bài "không nộp" sang temp_submission/khong_nop/ để submissions/
chỉ còn bài chờ nộp. Ghi submissions/manifest.json và submissions/MANIFEST.md (bảng để chọn bài nộp).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
DEST = ROOT / "submissions"            # chỉ bài CHƯA nộp, còn đáng nộp
DONE = ROOT / "temp_submission"        # bài đã có điểm (hoặc lỗi); khong_nop/ = bài quyết định không nộp
SOURCES = [ROOT / "runs/submissions", *sorted((ROOT / "kaggle/outputs").glob("*"))]
FIELDS = ("status", "note", "final", "docs_pr", "chunks_pr")


def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def clean_name(name: str) -> str:
    return name[len("submission_"):] if name.startswith("submission_") else name


def lb_scores() -> dict[str, dict]:
    """Bảng A của EXPERIMENTS.md: | Ngày | File | Hệ thống | N | FINAL | Docs P / R | Chunks P / R |."""
    out = {}
    text = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    section = text.split("## A.", 1)[1].split("\n## ", 1)[0]
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7 or not re.match(r"\d{1,2}/\d{1,2}", cells[0]):
            continue
        name = clean_name(Path(cells[1]).name)
        final = cells[4].replace("*", "")
        out[name] = {"status": "lỗi" if "lỗi" in final else "đã nộp", "final": final,
                     "docs_pr": cells[5], "chunks_pr": cells[6], "system": cells[2]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    DEST.mkdir(exist_ok=True)
    man_path = DEST / "manifest.json"
    manifest = {e["file"]: e for e in json.loads(man_path.read_text(encoding="utf-8"))} if man_path.exists() else {}

    for src_dir in SOURCES:
        for src in sorted(src_dir.glob("*.zip")) if src_dir.is_dir() else []:
            name = clean_name(src.name)
            target = DEST / name
            digest = sha12(src)
            if target.exists() and sha12(target) != digest:
                name = f"{src_dir.name.removeprefix('vibiomir-')}__{name}"
                target = DEST / name
            print(f"{src.relative_to(ROOT)} -> submissions/{name}")
            if args.dry_run:
                continue
            try:
                if target.exists():  # đã có cùng nội dung: bỏ bản sao nguồn vào _temp thay vì xoá
                    trash = ROOT / "_temp" / time.strftime("%Y-%m-%d") / "dup_zips" / src.relative_to(ROOT)
                    trash.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), trash)
                else:
                    shutil.copy2(src, target)
                    src.unlink()
            except PermissionError:  # file đang mở (vd. đang upload): để nguyên, lần chạy sau gom tiếp
                print(f"  bị khoá, bỏ qua lần này: {src.relative_to(ROOT)}")
                if not target.exists():
                    continue
            entry = manifest.setdefault(name, {"file": name, "status": "chưa nộp", "note": ""})
            entry.update({"from": str(src.relative_to(ROOT)).replace("\\", "/"), "sha12": digest,
                          "mb": round(target.stat().st_size / 1e6, 1),
                          "added": entry.get("added") or time.strftime("%Y-%m-%d %H:%M")})
            stats = src_dir / "build_stats.json"
            if stats.exists():
                s = json.loads(stats.read_text()).get(src.name, {})
                entry.update({k: s[k] for k in ("docs_per_query", "chunks_per_query") if k in s})
    if args.dry_run:
        return
    for f in DEST.glob("*.zip"):  # file có trong thư mục nhưng chưa có dòng manifest
        e = manifest.setdefault(f.name, {"file": f.name, "status": "chưa nộp", "note": ""})
        e.setdefault("sha12", sha12(f))
        e.setdefault("mb", round(f.stat().st_size / 1e6, 1))
        e.setdefault("added", time.strftime("%Y-%m-%d %H:%M"))

    for name, sc in lb_scores().items():
        if name in manifest:
            e = manifest[name]
            e.update({k: sc[k] for k in ("status", "final", "docs_pr", "chunks_pr")})
            e["note"] = e.get("note") or sc["system"]
    for name, e in manifest.items():
        sub = {"đã nộp": DONE, "lỗi": DONE, "không nộp": DONE / "khong_nop"}.get(e["status"])
        if sub and (DEST / name).exists():
            sub.mkdir(parents=True, exist_ok=True)
            shutil.move(str(DEST / name), sub / name)
        archived = sorted((ROOT / "_temp").glob(f"*/old_submissions/{name}"))
        if archived and not (DEST / name).exists():
            e["status"] = "lưu trữ"
        e["where"] = next((str(d.relative_to(ROOT)).replace("\\", "/") for d in
                           (DEST, DONE, DONE / "khong_nop", *(a.parent for a in archived)) if (d / name).exists()), "")
        e["exists"] = bool(e["where"])

    entries = sorted(manifest.values(), key=lambda e: (e["status"] != "chưa nộp", e.get("added", ""), e["file"]))
    man_path.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Bài nộp ViBioMIR — một chỗ duy nhất", "",
             "Sinh bởi `scripts/submit/collect_submissions.py`; sửa `status`/`note` trong `manifest.json`, điểm LB ghi ở "
             "`EXPERIMENTS.md` bảng A rồi chạy lại script.", "",
             "| File | Trạng thái | Ghi chú / hệ thống | MB | doc/q | chunk/q | FINAL | Docs P / R | Chunks P / R | Nằm ở | Nguồn |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    shown = [e for e in entries if e["status"] != "lưu trữ"]
    for e in shown:
        lines.append(f"| `{e['file']}`{'' if e.get('exists', True) else ' (mất)'} | {e['status']} | {e.get('note', '')} | "
                     f"{e.get('mb', '')} | {e.get('docs_per_query', '')} | {e.get('chunks_per_query', '')} | "
                     f"{e.get('final', '')} | {e.get('docs_pr', '')} | {e.get('chunks_pr', '')} | {e.get('where', '')} | {e.get('from', '')} |")
    n_arch = len(entries) - len(shown)
    if n_arch:
        lines += ["", f"{n_arch} bài cũ đã lưu trữ ở `_temp/*/old_submissions/` (README.md trong đó; vẫn có trong manifest.json)."]
    (DEST / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(entries)} entries -> submissions/MANIFEST.md")


if __name__ == "__main__":
    main()
