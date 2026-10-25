"""Resumable storage for crawled documents.

A 1.2M-URL crawl cannot be restarted from scratch, so progress lives in SQLite
(``done ids`` for filtering) while the payload lives in an append-only JSONL
file (easy to stream, compress and index later).
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import orjson

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    url TEXT,
    status INTEGER,
    ok INTEGER,
    n_chars INTEGER,
    error TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_docs_ok ON docs(ok);
"""

# marks a flush() request inside the writer queue (see CrawlSink.flush)
_SYNC = object()


class _AppendLock:
    """Cross-process mutex for appending to one shared JSONL file.

    A single ``write()`` on an O_APPEND handle is atomic, but a 50 KB article is
    split by the buffered/text layers into several syscalls -- two crawl
    processes then interleave halves of a line and ``json.loads`` drops the
    record silently. Holding an OS file lock around one raw write() removes the
    failure mode entirely.
    """

    def __init__(self, lock_path: Path) -> None:
        self._handle = open(lock_path, "a+b")

    def __enter__(self) -> _AppendLock:
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            for _ in range(300):  # msvcrt gives up after ~10 s; keep waiting
                try:
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                    return self
                except OSError:
                    time.sleep(0.05)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
            return self
        import fcntl

        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            with contextlib.suppress(OSError):
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._handle.close()


class CrawlSink:
    """Append crawl records to JSONL, tracking progress in SQLite."""

    def __init__(
        self,
        jsonl_path: str | Path,
        sqlite_path: str | Path,
        *,
        flush_every: int = 500,
        writer_thread: bool = True,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.sqlite_path = Path(sqlite_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.sqlite_path, timeout=60.0, check_same_thread=False)
        self._setup_schema()
        # Binary, unbuffered append: one write() syscall per record, guarded by an
        # OS file lock so a second crawl process cannot interleave half a line.
        self._handle = open(self.jsonl_path, "ab", buffering=0)
        self._append_lock = _AppendLock(self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".lock"))
        self._flush_every = flush_every
        self._lock = threading.Lock()
        self._rows: list[tuple] = []
        self._queue: queue.Queue = queue.Queue(maxsize=200_000)
        self._writer: threading.Thread | None = None
        if writer_thread:
            self._writer = threading.Thread(target=self._writer_loop, name="crawl-sink", daemon=True)
            self._writer.start()
        self.counters = {"written": 0, "ok": 0, "failed": 0}

    def _with_lock_retry(self, action: Callable[[], None], attempts: int = 6) -> None:
        """Run a schema/pragma step that a second crawl process may be holding a lock on.

        ``busy_timeout`` does not cover ``PRAGMA journal_mode=WAL`` (it needs an
        exclusive lock), so a second process used to die with "database is locked"
        the moment a parallel harvest was running.
        """
        for attempt in range(attempts):
            try:
                action()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == attempts - 1:
                    raise
                time.sleep(0.25 * (attempt + 1))

    def _setup_schema(self) -> None:
        # WAL is an optimisation; if another process is mid-write we keep the
        # current mode rather than refusing to start.
        try:
            self._with_lock_retry(lambda: self._connection.execute("PRAGMA journal_mode=WAL"))
        except sqlite3.OperationalError:
            pass
        self._connection.execute("PRAGMA busy_timeout=60000")
        self._connection.execute("PRAGMA synchronous=NORMAL")

        def create_schema() -> None:
            self._connection.executescript(_SCHEMA)
            self._connection.commit()

        self._with_lock_retry(create_schema)

    def done_ids(self) -> set[int]:
        cursor = self._connection.execute("SELECT id FROM docs")
        return {int(row[0]) for row in cursor.fetchall()}

    def succeeded_ids(self) -> set[int]:
        """Ids with usable text — what a resume should skip (failures must be retried)."""
        cursor = self._connection.execute("SELECT id FROM docs WHERE ok = 1")
        return {int(row[0]) for row in cursor.fetchall()}

    def ids_with_error_prefix(self, prefix: str) -> set[int]:
        """Ids stored as failures with a given error family (permanent vs retryable)."""
        cursor = self._connection.execute(
            "SELECT id FROM docs WHERE ok = 0 AND error LIKE ?", (prefix + "%",)
        )
        return {int(row[0]) for row in cursor.fetchall()}

    def failed_rows(
        self,
        *,
        errors: Iterable[str] | None = None,
        hosts: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[tuple[int, str]]:
        """``(doc_id, url)`` for documents that failed, optionally by error and host."""
        query = "SELECT id, url FROM docs WHERE ok = 0"
        params: list[object] = []
        error_list = [value for value in (errors or []) if value]
        if error_list:
            query += f" AND error IN ({','.join('?' for _ in error_list)})"
            params.extend(error_list)
        host_list = [value for value in (hosts or []) if value]
        if host_list:
            # match both "//host" (exact) and ".host" (subdomains of the host)
            clauses = " OR ".join("(url LIKE ? OR url LIKE ?)" for _ in host_list)
            query += f" AND ({clauses})"
            for host in host_list:
                params.extend([f"%//{host}%", f"%.{host}%"])
        query += " ORDER BY id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        cursor = self._connection.execute(query, params)
        return [(int(row[0]), str(row[1])) for row in cursor.fetchall()]

    def append(self, record: dict) -> None:
        """Write one record. Cheap on purpose: the event loop must never wait on SQLite.

        Two real failure modes are handled here, both measured on the live
        corpus: (1) blocking the asyncio loop in a commit retry collapsed the
        crawl to 0 doc/s while a parallel harvest held the SQLite write lock --
        so the DB bookkeeping goes to a writer thread; (2) a 50 KB article is
        split into several syscalls and two processes interleaved halves of a
        line -- so the raw append runs under an OS file lock.
        """
        line = orjson.dumps(record) + b"\n"
        with self._lock:
            with self._append_lock:
                self._handle.write(line)
            self.counters["written"] += 1
            self.counters["ok" if record.get("ok") else "failed"] += 1
        row = (
            int(record["id"]),
            str(record.get("url", "")),
            int(record.get("status", 0) or 0),
            1 if record.get("ok") else 0,
            int(record.get("n_chars", 0) or 0),
            str(record.get("error", ""))[:200],
        )
        if self._writer is not None:
            self._queue.put(row)
        else:
            with self._lock:
                self._rows.append(row)
                if len(self._rows) >= self._flush_every:
                    self._flush_locked()

    # ------------------------------------------------------------ writer thread
    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SYNC:
                    # a flush() request: FIFO order guarantees every record queued
                    # before it is already in self._rows
                    self._flush_rows()
                    continue
                if item is None:
                    break
                self._rows.append(item)
                if len(self._rows) >= self._flush_every:
                    self._flush_rows()
            finally:
                self._queue.task_done()
        self._flush_rows()

    def _flush_rows(self) -> None:
        """Commit the buffered rows in one short transaction.

        ``BEGIN IMMEDIATE`` takes the write lock up front and ``busy_timeout``
        then waits inside SQLite, instead of failing later on a lock upgrade
        (SQLite does not run the busy handler for those).
        """
        if not self._rows:
            return
        rows = self._rows
        self._rows = []
        for attempt in range(5):
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.executemany(
                    "INSERT OR REPLACE INTO docs (id, url, status, ok, n_chars, error) VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._connection.commit()
                return
            except sqlite3.OperationalError as exc:
                with contextlib.suppress(sqlite3.OperationalError):
                    self._connection.rollback()
                if "locked" not in str(exc).lower() or attempt == 4:
                    raise
                time.sleep(0.4 * (attempt + 1))
            except Exception:
                with contextlib.suppress(sqlite3.OperationalError):
                    self._connection.rollback()
                raise

    def _flush_locked(self) -> None:
        self._handle.flush()
        self._flush_rows()

    def flush(self) -> None:
        """Block until everything appended so far is on disk (used by progress/reporting).

        The sync marker travels through the same FIFO queue as the records: a
        flag the writer checks after ``get()`` loses the race when the queue is
        already drained, and the row then sits in the buffer until the next
        append.
        """
        if self._writer is None:
            with self._lock:
                self._handle.flush()
                self._flush_rows()
            return
        self._queue.put(_SYNC)
        self._queue.join()

    def close(self) -> None:
        if self._writer is not None:
            self._queue.put(None)
            self._queue.join()
            self._writer.join(timeout=60.0)
            self._writer = None
        with self._lock:
            try:
                self._handle.flush()
                self._flush_rows()
            finally:
                self._handle.close()
                self._append_lock.close()
                self._connection.close()

    def __enter__(self) -> CrawlSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def mark_ids_failed(sqlite_path: str | Path, ids: Iterable[int], error: str = "corrupt_line") -> int:
    """Push ids back into the retry pool (``ok = 0``) in one short transaction.

    Used when a payload line in ``docs.jsonl`` is corrupt: the SQLite row still
    exists, so a resume would skip the id forever while the text is unreadable.
    The crawler then re-fetches it on the next pass.
    """
    rows = [(error, int(doc_id)) for doc_id in ids]
    if not rows:
        return 0
    connection = sqlite3.connect(str(sqlite_path), timeout=60.0)
    try:
        connection.execute("PRAGMA busy_timeout=60000")
        for attempt in range(5):
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany("UPDATE docs SET ok = 0, error = ? WHERE id = ?", rows)
                connection.commit()
                return len(rows)
            except sqlite3.OperationalError as exc:
                with contextlib.suppress(sqlite3.OperationalError):
                    connection.rollback()
                if "locked" not in str(exc).lower() or attempt == 4:
                    raise
                time.sleep(0.4 * (attempt + 1))
    finally:
        connection.close()
    return 0


def count_jsonl(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    counter = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                counter += 1
    return counter


def iter_jsonl(path: str | Path):
    """Yield records; a truncated/interleaved line is skipped, not fatal."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
