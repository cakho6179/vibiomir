# vendor/

Pure-Python wheels the system imports straight from the zip (no install step, works offline).

| File | Source | License | sha256 |
|---|---|---|---|
| `anyjev-0.3.0-py3-none-any.whl` | PyPI `anyjev` 0.3.0 (github.com/nokia-applied-research/AnyJev) | Apache-2.0, `LICENSE-anyjev` | `24217f23eecb3859353d625b89e0047d02c70c8ce993f956e118d7e39b9dcf3f` |

`r2ai2026/stages/decide.py` puts the wheel on `sys.path` when `anyjev` is not installed. It needs only numpy,
torch and transformers. Only the training-free `Decider` is used, with allowed base models (Qwen3 family);
the Tacit checkpoints the package also offers are released after the 2026-08-01 cut-off and are not used.
