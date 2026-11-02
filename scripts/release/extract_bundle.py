"""Write out the stage code embedded in a Kaggle kernel bundle, so it can be read or run outside Kaggle.

    python scripts/release/extract_bundle.py kaggle/kernels_generic/vibiomir-mat-st3-acc2 --out tools/mat-st3

Every production bundle (kaggle/kernels_generic/<name>/run.py) carries the exact source it ran: SRC (the stage
script), an optional dict of helper modules, the pip packages and the argument list. This writes
<out>/<script>.py, the helper modules next to it, and <out>/ARGS.txt with the arguments as the kernel passed them
({input} = /kaggle/input, {working} = /kaggle/working on Kaggle).
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tree = ast.parse((args.bundle / "run.py").read_text(encoding="utf-8"))
    src = modules = script = argv = pip = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "SRC":
                src = ast.literal_eval(node.value)
            elif name == "MODULES":
                modules = ast.literal_eval(node.value)
            elif name == "pip":
                pip = ast.literal_eval(node.value)
            elif name == "script":
                script = Path(ast.literal_eval(node.value.args[0])).name
            elif name == "args" and isinstance(node.value, ast.ListComp):
                argv = ast.literal_eval(node.value.generators[0].iter)
        elif (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)  # for m, s in {...}.items(): inline modules
              and isinstance(node.iter.func, ast.Attribute) and isinstance(node.iter.func.value, ast.Dict)):
            modules = ast.literal_eval(node.iter.func.value)
    if src is None or script is None:
        raise SystemExit(f"{args.bundle}: no embedded stage script; run.py is itself the code that ran")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / script).write_text(src, encoding="utf-8", newline="")
    for name, text in (modules or {}).items():
        (args.out / name).write_text(text, encoding="utf-8", newline="")
    (args.out / "ARGS.txt").write_text(f"pip: {pip}\nargs: {argv}\n", encoding="utf-8")
    print(f"{args.bundle.name}: {script} + {sorted(modules or {})} -> {args.out}; args {argv}")


if __name__ == "__main__":
    main()
