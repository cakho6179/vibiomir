Pinned copy of the `r2ai2026` code that the materialiser kernels (vibiomir-mat-st3/st10/st11/st12) import from the
Kaggle dataset `tthiutoko/vibiomir-chunk-v21-assets` (`vibiomir-code/`, CODE_VERSION 06f71bf-dirty, uploaded 25/10).
Downloaded 02/11 from that dataset; the working tree has since changed these files.

- build_submission.py = r2ai2026/stages/build_submission.py (lexical_best, rr_spans, window_chunk)
- submission.py       = r2ai2026/core/submission.py (validate_submission, write_submission_zip, submission_stats)

stair_chunk ("seed" chunks) is embedded in each kernel's run.py, so it is pinned by the kernel bundle itself.
