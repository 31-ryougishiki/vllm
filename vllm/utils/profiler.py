# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lightweight wall-clock profiler for measuring latency between arbitrary
code points.  Works on CUDA, NPU, and CPU-only environments.

Usage::

    from vllm.utils.profiler import SimpleProfiler

    profiler = SimpleProfiler(enabled=True)

    profiler.start("q_proj")
    # ... Q projection code ...
    profiler.end("q_proj")

    profiler.start("kv_proj")
    # ... KV projection code ...
    elapsed_ms = profiler.end("kv_proj")

    # Or use as a context manager:
    with profiler.measure("attention_kernel"):
        # ... attention kernel ...

    # Print accumulated statistics:
    profiler.summary()
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


def _sync_device() -> None:
    """Synchronize the current device stream.

    Tries NPU first (Ascend), then CUDA, then falls back to CPU.
    """
    try:
        import torch_npu  # type: ignore[import-untyped]

        torch_npu.npu.synchronize()
        return
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return
    except (ImportError, AttributeError, RuntimeError):
        pass


class SimpleProfiler:
    """A lightweight wall-clock profiler for ad-hoc latency measurements.

    Each named measurement records the elapsed wall-clock time between
    :meth:`start` and :meth:`end` calls.  Multiple calls with the same
    name accumulate statistics (count, total, min, max).

    The ``enabled`` flag gates all operations; when ``False``, every
    method is a no-op (zero overhead path).

    Parameters
    ----------
    enabled:
        When ``False``, all methods return immediately without measuring.
    sync_device:
        When ``True`` (default), call ``_sync_device()`` before each
        timestamp to ensure device-side work is complete.
    """

    def __init__(
        self,
        enabled: bool = True,
        *,
        sync_device: bool = True,
    ) -> None:
        self._enabled = enabled
        self._sync_device = sync_device
        self._starts: dict[str, float] = {}
        self._records: dict[str, _Measure] = {}
        self._marks: dict[str, float] = {}  # last timestamp for mark()

    # -- enable / disable -------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # -- core API ---------------------------------------------------------

    def start(self, name: str) -> None:
        """Record the start of a named measurement segment.

        Must be paired with a later :meth:`end` call using the same name.
        Calling ``start`` again with the same name before ``end``
        overwrites the previous start time.
        """
        if not self._enabled:
            return
        if self._sync_device:
            _sync_device()
        self._starts[name] = time.perf_counter()

    def end(self, name: str) -> float:
        """End the measurement for *name* and return elapsed milliseconds.

        Raises ``KeyError`` if ``start(name)`` was not called first.
        """
        if not self._enabled:
            return 0.0
        if self._sync_device:
            _sync_device()
        now = time.perf_counter()
        start = self._starts.pop(name)
        elapsed_ms = (now - start) * 1000.0
        rec = self._records.get(name)
        if rec is None:
            self._records[name] = _Measure(
                count=1, total_ms=elapsed_ms, min_ms=elapsed_ms, max_ms=elapsed_ms
            )
        else:
            rec.count += 1
            rec.total_ms += elapsed_ms
            if elapsed_ms < rec.min_ms:
                rec.min_ms = elapsed_ms
            if elapsed_ms > rec.max_ms:
                rec.max_ms = elapsed_ms
        return elapsed_ms

    def measure(self, name: str) -> Generator[None, None, None]:
        """Context manager that times the enclosed block.

        Usage::

            with profiler.measure("attention_kernel"):
                # ... attention computation ...
        """
        self.start(name)
        try:
            yield
        finally:
            self.end(name)

    # -- mark-based API (for sequential timing between points) -------------

    def mark(self, name: str) -> float:
        """Record a timestamp and return elapsed ms since the previous
        ``mark`` or ``start`` call with the same name.

        On the first call with a new *name*, records the initial timestamp
        and returns 0.0.
        """
        if not self._enabled:
            return 0.0
        if self._sync_device:
            _sync_device()
        now = time.perf_counter()
        elapsed_ms = 0.0
        if name in self._marks:
            elapsed_ms = (now - self._marks[name]) * 1000.0
        self._marks[name] = now
        return elapsed_ms

    # -- query ------------------------------------------------------------

    def elapsed(self, name: str) -> float:
        """Return total accumulated milliseconds for *name* so far."""
        rec = self._records.get(name)
        return rec.total_ms if rec else 0.0

    def count(self, name: str) -> int:
        """Return number of measurements recorded for *name*."""
        rec = self._records.get(name)
        return rec.count if rec else 0

    def avg(self, name: str) -> float:
        """Return average milliseconds per measurement for *name*."""
        rec = self._records.get(name)
        if rec is None or rec.count == 0:
            return 0.0
        return rec.total_ms / rec.count

    def stats(self, name: str) -> dict[str, float]:
        """Return ``{count, total_ms, min_ms, max_ms, avg_ms}`` for *name*."""
        rec = self._records.get(name)
        if rec is None:
            return {"count": 0, "total_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0}
        return {
            "count": rec.count,
            "total_ms": rec.total_ms,
            "min_ms": rec.min_ms,
            "max_ms": rec.max_ms,
            "avg_ms": rec.total_ms / rec.count if rec.count else 0.0,
        }

    def names(self) -> list[str]:
        """Return all measurement names in insertion order."""
        return list(self._records.keys())

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Clear all measurements and start timestamps."""
        self._starts.clear()
        self._records.clear()
        self._marks.clear()

    def summary(self, *, precision: int = 3) -> str:
        """Return a formatted multi-line summary of all measurements.

        Parameters
        ----------
        precision:
            Decimal places for millisecond values.
        """
        if not self._records:
            return "SimpleProfiler: no measurements recorded."

        lines = ["SimpleProfiler summary:"]
        fmt = f"  {{:<40s} count={{:>6d}}  total={{:>10.{precision}f}}ms  "
        fmt += f"avg={{:>10.{precision}f}}ms  "
        fmt += f"min={{:>10.{precision}f}}ms  "
        fmt += f"max={{:>10.{precision}f}}ms"
        for name in self._records:
            rec = self._records[name]
            avg = rec.total_ms / rec.count if rec.count else 0.0
            lines.append(
                fmt.format(
                    name,
                    count=rec.count,
                    total=rec.total_ms,
                    avg=avg,
                    min=rec.min_ms,
                    max=rec.max_ms,
                )
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "enabled" if self._enabled else "disabled"
        n = len(self._records)
        return f"SimpleProfiler({status}, {n} measurements)"


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

class _Measure:
    __slots__ = ("count", "total_ms", "min_ms", "max_ms")

    def __init__(self, count: int, total_ms: float, min_ms: float, max_ms: float) -> None:
        self.count = count
        self.total_ms = total_ms
        self.min_ms = min_ms
        self.max_ms = max_ms


# module-level singleton (disabled by default)
_profiler = SimpleProfiler(enabled=False)


def get_profiler() -> SimpleProfiler:
    """Return the module-level profiler singleton.

    Enable it before use::

        from vllm.utils.profiler import get_profiler
        get_profiler().enabled = True
    """
    return _profiler


# ---------------------------------------------------------------------------
# StepLatencyCollector — per-step profiling with auto-flush to disk
# ---------------------------------------------------------------------------

import json
import os
import threading

import torch


class StepLatencyCollector:
    """Collect per-inference-step profiling data and auto-flush to disk.

    Each step records its *num_tokens* (as the grouping key) along with a
    dict of named latency measurements (e.g. ``{"q_proj": 0.287, ...}``).
    Data is organised per *num_tokens* bucket and per rank.

    A background daemon thread checks every second whether new data has
    been added.  When the collector has been idle for *flush_idle_sec*
    seconds **and** the buffer is non-empty, all accumulated data is
    written to ``{output_dir}/profile_rank_{RANK}.json`` and the buffer
    is cleared.

    Parameters
    ----------
    output_dir:
        Directory where per-rank JSON files will be written.
    flush_idle_sec:
        Seconds of inactivity after which data is flushed to disk.
        Default: 30.0.
    enabled:
        When ``False``, ``record_step`` is a no-op and no background
        thread is started.
    """

    def __init__(
        self,
        output_dir: str,
        *,
        flush_idle_sec: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self._output_dir = output_dir
        self._flush_idle_sec = flush_idle_sec
        self._enabled = enabled

        # num_tokens (int) -> list of per-step measurement dicts
        self._data: dict[int, list[dict[str, float]]] = {}
        self._last_record_time = time.time()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        if self._enabled:
            self._flush_thread = threading.Thread(
                target=self._flush_loop, daemon=True, name="step-profiler-flush"
            )
            self._flush_thread.start()
        else:
            self._flush_thread = None

    # -- public API --------------------------------------------------------

    def record_step(self, num_tokens: int, measurements: dict[str, float]) -> None:
        """Record one inference step.

        Parameters
        ----------
        num_tokens:
            Input token count for this step (grouping key).
        measurements:
            Dict mapping operation name → elapsed milliseconds.
        """
        if not self._enabled:
            return
        with self._lock:
            bucket = self._data.setdefault(num_tokens, [])
            bucket.append(dict(measurements))
            self._last_record_time = time.time()

    def snapshot_and_reset(self, profiler: SimpleProfiler) -> None:
        """Convenience: take a snapshot of *profiler*, record the step,
        then reset the profiler for the next step.

        *num_tokens* must be stored in the profiler via a ``"num_tokens"``
        mark (see :meth:`SimpleProfiler.mark`), or pass it directly.
        """
        if not self._enabled:
            return
        measurements = {name: profiler.stats(name)["total_ms"] for name in profiler.names()}
        # Try to read num_tokens from profiler records; fall back to -1.
        num_tokens_rec = profiler._records.get("num_tokens")
        num_tokens = int(num_tokens_rec.total_ms) if num_tokens_rec else -1
        self.record_step(num_tokens, measurements)
        profiler.reset()

    def flush(self) -> str | None:
        """Force-flush accumulated data to disk immediately.

        Returns the file path if data was written, else ``None``.
        """
        with self._lock:
            if not self._data:
                return None
            return self._flush_locked()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def stop(self) -> None:
        """Signal the background thread to stop and flush remaining data."""
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
        self.flush()

    # -- internals ---------------------------------------------------------

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=1.0)
            with self._lock:
                if self._data and (
                    time.time() - self._last_record_time > self._flush_idle_sec
                ):
                    self._flush_locked()

    def _flush_locked(self) -> str:
        """Must be called with ``self._lock`` held and ``self._data`` non-empty."""
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        os.makedirs(self._output_dir, exist_ok=True)
        filepath = os.path.join(self._output_dir, f"profile_rank_{rank}.json")

        # Merge with existing file if present (for long-running services that
        # flush multiple times).
        merged: dict[int, list[dict[str, float]]] = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, encoding="utf-8") as f:
                    existing = json.load(f)
                # json keys are strings; convert back to int
                for k, v in existing.items():
                    merged[int(k)] = v
            except (json.JSONDecodeError, OSError):
                pass

        for num_tokens, steps in self._data.items():
            if num_tokens in merged:
                merged[num_tokens].extend(steps)
            else:
                merged[num_tokens] = list(steps)

        # Write atomically: temp file then rename.
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

        os.replace(tmp_path, filepath)
        self._data.clear()
        return filepath


# module-level collector singleton (disabled by default)
_collector: StepLatencyCollector | None = None


def get_collector(output_dir: str = "", *, enabled: bool = True) -> StepLatencyCollector:
    """Return (or create) the module-level :class:`StepLatencyCollector` singleton.

    On first call, *output_dir* and *enabled* configure the singleton.
    Subsequent calls return the same instance, ignoring the arguments.
    """
    global _collector
    if _collector is None:
        _collector = StepLatencyCollector(
            output_dir=output_dir,
            flush_idle_sec=30.0,
            enabled=enabled,
        )
    return _collector


@contextlib.contextmanager
def profile(name: str, *, enabled: bool = True) -> Generator[None, None, None]:
    """Convenience context manager using the module-level profiler.

    Usage::

        from vllm.utils.profiler import profile

        with profile("my_block"):
            # ... code to measure ...
    """
    if not enabled:
        yield
        return
    _profiler.start(name)
    try:
        yield
    finally:
        _profiler.end(name)
