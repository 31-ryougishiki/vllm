# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lightweight call-stack tracer for debugging vLLM inference code paths.

Designed to answer: "Which functions are called during a single inference
step, in what order, and (optionally) how long does each take?"

Key design decisions:
- **First-step only** — the code path doesn't change across steps, so we
  record the first step and skip the rest (configurable via
  ``first_step_only``).
- **Timing off by default** — by default only the call tree structure is
  recorded; enable ``enable_timing=True`` or pass ``timing=True`` on a
  per-span basis to add wall-clock durations.

Usage::

    from vllm.utils.callstack import CallStackTracer, get_tracer

    # Quick-start: print call path for the first step only, no timing.
    tracer = get_tracer(enabled=True)
    tracer.step_begin("step_0", metadata={"num_tokens": 120})

    with tracer.span("model_forward"):
        with tracer.span("attention"):
            ...

    tracer.step_end()
    print(tracer.tree_text())

    # Later: re-enable for a single step with timing.
    tracer.enable_timing = True
    tracer.reset_first_step()
    tracer.step_begin("step_N")
    ...
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import traceback
from collections.abc import Generator
from typing import Any

# ---------------------------------------------------------------------------
# Level 1: one-shot call-stack snapshot (diagnostic)
# ---------------------------------------------------------------------------


def snapshot(
    label: str = "",
    *,
    skip_frames: int = 0,
    max_depth: int = 20,
    module_filter: str | None = None,
) -> list[str]:
    """Capture the current Python call stack and return it as a list.

    Args:
        label: Human label printed as a header line (``[CallStack] <label>``).
        skip_frames: Extra stack frames to omit beyond the ``snapshot()``
            call itself.  Default 0.
        max_depth: Maximum number of frames to capture.
        module_filter: If given, only frames whose filename contains this
            string are expanded; other frames are collapsed to ``... in
            <name>``.  E.g. ``"vllm"`` shows vllm / vllm-ascend frames.

    Returns:
        List of ``"file:line in func_name"`` strings, innermost frame first.
    """
    frames = traceback.extract_stack(limit=max_depth + skip_frames + 1)
    # Remove the snapshot() frame itself + any caller-requested extras.
    frames = frames[: -(skip_frames + 2)]

    lines: list[str] = []
    for f in frames:
        fname = f.filename.replace("\\", "/")
        if module_filter is None or module_filter in fname:
            short = fname.split("/")[-1] if "/" in fname else fname
            lines.append(f"{short}:{f.lineno} in {f.name}")
        else:
            lines.append(f"... in {f.name}")

    if label:
        header = f"[CallStack] {label}"
        print(header)
        for line in lines:
            print(f"  {line}")
    return lines


# ---------------------------------------------------------------------------
# Level 2: structured span-tree tracer
# ---------------------------------------------------------------------------


class _Span:
    """Single node in the call tree.  Internal — not part of the public API."""

<<<<<<< ours
    __slots__ = (
        "name", "start_us", "end_us", "children", "attrs",
        "_start_event", "_end_event",
    )
=======
    __slots__ = ("name", "start_us", "end_us", "children", "attrs")
>>>>>>> theirs

    def __init__(
        self,
        name: str,
        start_us: float = 0.0,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.start_us = start_us  # 0 = untimed
        self.end_us: float = 0.0  # 0 = untimed
        self.children: list[_Span] = []
        self.attrs: dict[str, Any] = attrs or {}
<<<<<<< ours
        self._start_event: Any = None  # torch.npu.Event when NPU timing
        self._end_event: Any = None
=======
>>>>>>> theirs

    @property
    def is_timed(self) -> bool:
        return self.start_us > 0 or self.end_us > 0

    @property
    def duration_us(self) -> float:
        return max(0.0, self.end_us - self.start_us)


<<<<<<< ours
# ---------------------------------------------------------------------------
# Zero-overhead no-op span used when tracing is disabled.
# Using a pre-allocated singleton avoids creating context-manager objects
# on the hot path and skips Python's context-manager protocol overhead.
# ---------------------------------------------------------------------------


class _NoopSpan:
    """Zero-overhead no-op context manager for the disabled-tracer fast path.

    Callable — ``_NOOP_SPAN("name", **kw)`` returns ``self`` so it can
    replace ``tracer.span("name", **kw)`` in ``with`` statements directly.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object, **kwargs: object) -> None:
        pass

    def __call__(self, _name: str = "", **_kw: object) -> "_NoopSpan":
        return self


_NOOP_SPAN = _NoopSpan()


=======
>>>>>>> theirs
class CallStackTracer:
    """Lightweight tracer that records a tree of named spans.

    By default it captures the **first step only** without timing.  Both
    behaviours can be changed at construction time or via properties.

    Parameters
    ----------
    enabled:
        Master on/off switch.  When ``False`` every method is a no-op.
    first_step_only:
        When ``True`` (default), ``step_begin`` records the first call and
        then sets ``active=False`` so subsequent steps are skipped.
    enable_timing:
        When ``True``, every span records wall-clock enter / exit timestamps.
        When ``False`` (default), spans record structure only (zero overhead
        from ``perf_counter`` calls).  Per-span override via ``timing=True``.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        first_step_only: bool = True,
        enable_timing: bool = False,
<<<<<<< ours
        use_npu_timing: bool = False,
        step_interval: int = 1,
=======
>>>>>>> theirs
    ) -> None:
        self._enabled = enabled
        self._first_step_only = first_step_only
        self._enable_timing = enable_timing
<<<<<<< ours
        self._use_npu_timing = use_npu_timing
        self._step_interval = step_interval
=======
>>>>>>> theirs
        self._active: bool = False
        self._step_count: int = 0
        self._step_id: str = ""
        self._step_metadata: dict[str, Any] = {}
        self._root: _Span | None = None
        self._stack: list[_Span] = []
        self._t0_us: float = 0.0
        self._tid: int = threading.get_ident()
        self._lock: threading.Lock = threading.Lock()

    # -- properties ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            self._active = False

    @property
    def enable_timing(self) -> bool:
        return self._enable_timing

    @enable_timing.setter
    def enable_timing(self, value: bool) -> None:
        self._enable_timing = value

    @property
    def active(self) -> bool:
        """Is the tracer currently recording (for the *current* step)?"""
        return self._active

    # -- step lifecycle ------------------------------------------------------

    def step_begin(
        self,
        step_id: str = "",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Start recording a new inference step.

        If ``first_step_only`` is set and this is not the first call, the
        tracer moves to ``active=False`` and all subsequent ``span()`` calls
        become no-ops.
        """
        if not self._enabled:
            return
        with self._lock:
            if self._first_step_only and self._step_count > 0:
                self._active = False
                return
            self._step_count += 1
<<<<<<< ours
            # Sample: only record every step_interval steps
            # (step 1 is always recorded for the first-step log).
            if self._step_interval > 1:
                if (
                    self._step_count != 1
                    and self._step_count % self._step_interval != 0
                ):
                    self._active = False
                    return
            self._step_id = step_id or f"step_{self._step_count}"
            self._step_metadata = dict(metadata or {})
            if self._use_npu_timing:
                self._t0_us = 0.0
                self._root = _Span(self._step_id, start_us=0.0)
                self._root._start_event = self._create_npu_event()
                self._root._start_event.record()
            else:
                self._t0_us = time.perf_counter_ns() / 1000.0
                self._root = _Span(self._step_id, start_us=0.0)
=======
            self._step_id = step_id or f"step_{self._step_count}"
            self._step_metadata = dict(metadata or {})
            self._t0_us = time.perf_counter_ns() / 1000.0
            self._root = _Span(self._step_id, start_us=0.0)
>>>>>>> theirs
            self._stack = [self._root]
            self._active = True

    def step_end(self) -> None:
        """Finish the current step and close any unclosed spans."""
        if not self._active:
            return
        with self._lock:
            while len(self._stack) > 1:
                self._pop_span()
            if self._root is not None:
<<<<<<< ours
                if (
                    self._use_npu_timing
                    and self._root._start_event is not None
                ):
                    self._root._end_event = self._create_npu_event()
                    self._root._end_event.record()
                    self._synchronize_npu()
                    self._compute_npu_durations(self._root)
                else:
                    self._root.end_us = (
                        time.perf_counter_ns() / 1000.0 - self._t0_us
                    )
=======
                self._root.end_us = (
                    time.perf_counter_ns() / 1000.0 - self._t0_us
                )
>>>>>>> theirs

    def reset_first_step(self) -> None:
        """Reset the step counter so the next ``step_begin`` records again.

        Useful after calling with ``first_step_only=True`` when you want to
        re-capture (e.g. after changing ``enable_timing``).
        """
        with self._lock:
            self._step_count = 0

    # -- span tracking -------------------------------------------------------

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        *,
        timing: bool = False,
        **attrs: Any,
    ) -> Generator[None, None, None]:
        """Record a named span that encloses a block of code.

        Parameters
        ----------
        name:
            Span label.  Use ``"/"`` to hint at nesting in output (e.g.
            ``"layer_0/attn/q_proj"``).
        timing:
            When ``True``, record wall-clock timestamps for this span even
            when the global ``enable_timing`` is ``False`` — useful for
            profiling a single suspect span without paying the overhead on
            every span.
        **attrs:
            Arbitrary key-value pairs stored on the span node and shown in
            ``tree_text()`` (e.g. ``ratio=4``).
        """
        if not self._active:
            yield
            return

        timed = self._enable_timing or timing
<<<<<<< ours
        node = _Span(name, start_us=0.0, attrs=attrs if attrs else None)

        if timed:
            if self._use_npu_timing:
                node._start_event = self._create_npu_event()
                node._start_event.record()
            else:
                # All timestamps are relative to step_begin's _t0_us so that
                # Chrome Trace output and duration calculations are consistent.
                node.start_us = (
                    time.perf_counter_ns() / 1000.0 - self._t0_us
                )
=======
        # All timestamps are relative to step_begin's _t0_us so that
        # Chrome Trace output and duration calculations are consistent.
        _now_us = (time.perf_counter_ns() / 1000.0 - self._t0_us) if timed else 0.0
        node = _Span(name, start_us=_now_us, attrs=attrs if attrs else None)
>>>>>>> theirs

        with self._lock:
            if self._stack:
                self._stack[-1].children.append(node)
            self._stack.append(node)

        try:
            yield
        finally:
<<<<<<< ours
            if timed:
                if self._use_npu_timing:
                    node._end_event = self._create_npu_event()
                    node._end_event.record()
                else:
                    node.end_us = (
                        time.perf_counter_ns() / 1000.0 - self._t0_us
                    )
            with self._lock:
=======
            _end_us = (time.perf_counter_ns() / 1000.0 - self._t0_us) if timed else 0.0
            with self._lock:
                node.end_us = _end_us
>>>>>>> theirs
                self._stack.pop()

    def mark(self, name: str, **attrs: Any) -> None:
        """Record a zero-duration marker (e.g. branch taken, metadata)."""
        if not self._active:
            return
        with self._lock:
            if self._stack:
                node = _Span(name, start_us=0.0, end_us=0.0, attrs=attrs if attrs else None)
                self._stack[-1].children.append(node)

    # -- output --------------------------------------------------------------

    def tree_text(self) -> str:
        """Render the recorded call tree as indented text.

        If timing data is present (any span has ``start_us > 0``) each line
        includes the duration and percentage of parent.  Otherwise a
        timing-free structural tree is returned.
        """
        if self._root is None or not self._root.children:
            return "CallStackTracer: no data recorded."

        meta = ", ".join(
            f"{k}={v}" for k, v in self._step_metadata.items()
        )
        total_us = self._root.end_us
        has_timing = total_us > 0

        if has_timing:
            header = f"{self._root.name}  [{meta}]  {total_us / 1000:.1f}ms"
        else:
            header = f"{self._root.name}  [{meta}]"

        lines = [header]
        self._render_tree(
            self._root, lines, indent="", has_timing=has_timing,
            parent_us=total_us,
        )
        return "\n".join(lines)

    def to_chrome_trace(self, *, filepath: str | None = None) -> dict[str, Any]:
        """Export as Chrome Trace Format JSON.

        If *filepath* is provided the JSON is written directly to disk
        (atomic write via temp file + rename).

        Returns the JSON-serialisable dict (also written to *filepath* if
        given).
        """
        events: list[dict[str, Any]] = []
        if self._root is not None:
            ts0_ns = 0
            self._chrome_events(self._root, ts0_ns, pid=0, events=events)

        obj: dict[str, Any] = {
            "traceEvents": events,
            "displayTimeUnit": "ns",
        }

        if filepath:
            tmp = filepath + ".tmp"
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
            os.replace(tmp, filepath)

        return obj

    def to_json(self, *, filepath: str | None = None) -> dict[str, Any]:
        """Export the call tree as a simple nested JSON object.

        Useful for programmatic consumption or post-processing.
        """
        if self._root is None:
            return {}

        def _node_to_dict(node: _Span) -> dict[str, Any]:
            d: dict[str, Any] = {"name": node.name}
            if node.attrs:
                d["attrs"] = node.attrs
            if node.is_timed:
                d["duration_us"] = round(node.duration_us, 3)
            if node.children:
                d["children"] = [_node_to_dict(c) for c in node.children]
            return d

        obj = {
            "step_id": self._step_id,
            "metadata": self._step_metadata,
            "tree": _node_to_dict(self._root),
        }

        if filepath:
            tmp = filepath + ".tmp"
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
            os.replace(tmp, filepath)

        return obj

<<<<<<< ours
    def flatten_spans(self) -> dict[str, float]:
        """Walk the span tree and return a flat dict of ``path → duration_us``.

        Paths are built by joining span names with ``"/"``, skipping the
        root step span itself.  Only timed spans are included.

        Example::

            {"model_forward": 1100000.0,
             "model_forward/post_process": 45000.0}
        """
        if self._root is None:
            return {}
        result: dict[str, float] = {}
        for child in self._root.children:
            self._flatten_child(child, "", result)
        return result

    @staticmethod
    def _flatten_child(
        span: _Span, prefix: str, result: dict[str, float]
    ) -> None:
        path = f"{prefix}/{span.name}" if prefix else span.name
        if span.is_timed and span.duration_us > 0:
            result[path] = span.duration_us
        for child in span.children:
            CallStackTracer._flatten_child(child, path, result)

=======
>>>>>>> theirs
    # -- internals -----------------------------------------------------------

    def _pop_span(self) -> None:
        """Pop the innermost span, setting its end time."""
        s = self._stack.pop()
<<<<<<< ours
        if self._use_npu_timing:
            return  # Events already recorded in span() finally block
        if s.start_us > 0:
            s.end_us = time.perf_counter_ns() / 1000.0 - self._t0_us

    # -- NPU event helpers ----------------------------------------------------

    @staticmethod
    def _create_npu_event() -> Any:
        """Create an NPU event with timing enabled.

        Uses lazy import so ``callstack`` remains importable on platforms
        without ``torch_npu`` (e.g. upstream CI, CPU-only environments).
        """
        try:
            import torch
            return torch.npu.Event(enable_timing=True)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "use_npu_timing=True requires torch.npu.Event support "
                "(torch_npu must be installed)"
            ) from exc

    @staticmethod
    def _synchronize_npu() -> None:
        """Synchronize the NPU device so all recorded events are complete."""
        import torch
        torch.npu.synchronize()

    @staticmethod
    def _compute_npu_durations(span: "_Span") -> None:
        """Walk the span tree and compute ``end_us`` from NPU event elapsed times.

        Post-condition: for every timed span, ``start_us`` stays 0 and
        ``end_us`` is set to ``elapsed_ms * 1000`` (microseconds).
        """
        if span._start_event is not None and span._end_event is not None:
            elapsed_ms = span._start_event.elapsed_time(span._end_event)
            span.end_us = elapsed_ms * 1000.0
            span.start_us = 0.0
        for child in span.children:
            CallStackTracer._compute_npu_durations(child)

=======
        if s.start_us > 0:
            s.end_us = time.perf_counter_ns() / 1000.0 - self._t0_us

>>>>>>> theirs
    @staticmethod
    def _render_tree(
        span: _Span,
        lines: list[str],
        indent: str,
        has_timing: bool,
        parent_us: float,
    ) -> None:
        """Recursively render *span* children into *lines*."""
        n_children = len(span.children)
        for i, child in enumerate(span.children):
            is_last = i == n_children - 1
            branch = "└── " if is_last else "├── "
            next_indent = indent + ("    " if is_last else "│   ")

            # build the line
            parts = [indent, branch, child.name]

            if has_timing and child.is_timed:
                dur_ms = child.duration_us / 1000.0
                pct = (
                    f" ({dur_ms / (parent_us / 1000.0) * 100:.0f}%)"
                    if parent_us > 0
                    else ""
                )
                parts.append(f"  {dur_ms:.1f}ms{pct}")
            elif has_timing:
                parts.append("  (untimed)")

            if child.attrs:
                attr_str = ", ".join(
                    f"{k}={v}" for k, v in child.attrs.items()
                )
                parts.append(f"  [{attr_str}]")

            lines.append("".join(parts))
            CallStackTracer._render_tree(
                child, lines, next_indent, has_timing, parent_us,
            )

    @staticmethod
    def _chrome_events(
        span: _Span,
        ts0_ns: int,
        pid: int,
        events: list[dict[str, Any]],
    ) -> None:
        """Recursively emit Chrome Trace events for *span*."""
        tid = threading.get_ident()
        ts_b = ts0_ns + int(span.start_us * 1000)
        ts_e = ts0_ns + int(span.end_us * 1000)

        if span.is_timed and ts_e > ts_b:
            events.append({
                "name": span.name,
                "ph": "B",
                "ts": ts_b,
                "pid": pid,
                "tid": tid,
                "args": span.attrs or {},
            })
            events.append({
                "name": span.name,
                "ph": "E",
                "ts": ts_e,
                "pid": pid,
                "tid": tid,
            })
        else:
            # Untimed spans become instant events at t=0 (or parent start).
            events.append({
                "name": span.name,
                "ph": "I",
                "ts": ts_b if ts_b > 0 else ts0_ns,
                "pid": pid,
                "tid": tid,
                "args": span.attrs or {},
                "s": "g",
            })

        for child in span.children:
            CallStackTracer._chrome_events(child, ts0_ns, pid, events)


# ---------------------------------------------------------------------------
<<<<<<< ours
# CallStackCollector — per-step accumulation with auto-flush to disk
# ---------------------------------------------------------------------------


class CallStackCollector:
    """Collect per-step flattened span data and flush to disk periodically.

    Each call to :meth:`collect_step` stores a snapshot of the tracer's
    current step tree (flattened to ``path → duration_us``).  Once the
    buffer reaches *flush_interval* steps, all accumulated data is written
    to disk and the buffer is cleared.

    The output file includes per-step detail and cross-step aggregate
    statistics (average, min, max) — both raw and per-token.

    Parameters
    ----------
    output_dir:
        Directory where per-rank JSON files are written.
    flush_interval:
        Number of steps to buffer before flushing to disk.  Default 100.
        Set to 1 to flush every step (zero buffering).
    enabled:
        When ``False``, ``collect_step`` is a no-op.
    """

    def __init__(
        self,
        output_dir: str,
        *,
        flush_interval: int = 100,
        enabled: bool = True,
    ) -> None:
        self._output_dir = output_dir
        self._flush_interval = flush_interval
        self._enabled = enabled

        # Accumulated step records
        self._steps: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def collect_step(
        self,
        tracer: CallStackTracer,
        num_tokens: int,
    ) -> None:
        """Snapshot *tracer*'s current step tree and buffer it.

        Flushes to disk when the buffer reaches ``flush_interval`` steps.

        Parameters
        ----------
        tracer:
            The tracer whose :meth:`~CallStackTracer.flatten_spans` is
            called to extract per-span durations.
        num_tokens:
            Number of tokens for this step (used for per-token stats).
        """
        if not self._enabled:
            return
        spans = tracer.flatten_spans()
        if not spans:
            return  # nothing timed, skip
        record: dict[str, Any] = {
            "step_id": tracer._step_id,
            "num_tokens": num_tokens,
            "spans": spans,  # {path: duration_us}
            "timestamp": time.time(),
        }
        with self._lock:
            self._steps.append(record)
            if len(self._steps) >= self._flush_interval:
                self._flush_locked()

    def flush(self) -> str | None:
        """Force-flush all buffered steps to disk.

        Returns the output file path, or ``None`` if there was nothing to
        write.
        """
        with self._lock:
            if not self._steps:
                return None
            return self._flush_locked()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def pending_steps(self) -> int:
        """Number of buffered steps not yet flushed."""
        with self._lock:
            return len(self._steps)

    def stop(self) -> None:
        """Flush any remaining buffered steps to disk."""
        self.flush()

    # -- internals -----------------------------------------------------------

    def _flush_locked(self) -> str:
        """Must be called with ``self._lock`` held and ``self._steps`` non-empty."""
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        os.makedirs(self._output_dir, exist_ok=True)
        filepath = os.path.join(
            self._output_dir, f"callstack_rank_{rank}.json"
        )

        # Merge with existing file if present.
        existing_steps: list[dict[str, Any]] = []
        if os.path.exists(filepath):
            try:
                with open(filepath, encoding="utf-8") as f:
                    prev = json.load(f)
                existing_steps = prev.get("steps", [])
            except (json.JSONDecodeError, OSError):
                pass

        all_steps = existing_steps + list(self._steps)
        self._steps.clear()

        # Compute aggregate stats across all (existing + new) steps.
        aggregate = self._compute_aggregate(all_steps)

        output: dict[str, Any] = {
            "steps": all_steps,
            "aggregate": aggregate,
        }

        tmp = filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        os.replace(tmp, filepath)
        return filepath

    @staticmethod
    def _compute_aggregate(
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute cross-step aggregate stats.

        Returns a dict with ``total_steps`` and per-span-name stats:
        count, avg_us, avg_us_per_token, min_us, max_us,
        min_us_per_token, max_us_per_token.
        """
        # accumulator: path → [total_us, total_us_per_token, count, min_us, max_us, min_upt, max_upt]
        acc: dict[
            str, list[float]
        ] = {}  # [sum_us, sum_upt, count, min_us, max_us, min_upt, max_upt]

        for step in steps:
            num_tokens = step.get("num_tokens", 1)
            spans: dict[str, float] = step.get("spans", {})
            for path, dur_us in spans.items():
                upt = dur_us / num_tokens if num_tokens > 0 else 0.0
                if path not in acc:
                    acc[path] = [0.0, 0.0, 0, float("inf"), float("-inf"), float("inf"), float("-inf")]
                a = acc[path]
                a[0] += dur_us
                a[1] += upt
                a[2] += 1
                if dur_us < a[3]:
                    a[3] = dur_us
                if dur_us > a[4]:
                    a[4] = dur_us
                if upt < a[5]:
                    a[5] = upt
                if upt > a[6]:
                    a[6] = upt

        spans_agg: dict[str, dict[str, float]] = {}
        for path, a in acc.items():
            cnt = a[2]
            spans_agg[path] = {
                "count": cnt,
                "avg_us": round(a[0] / cnt, 1),
                "avg_us_per_token": round(a[1] / cnt, 4),
                "min_us": round(a[3], 1),
                "max_us": round(a[4], 1),
                "min_us_per_token": round(a[5], 4),
                "max_us_per_token": round(a[6], 4),
            }

        return {"total_steps": len(steps), "spans": spans_agg}


# ---------------------------------------------------------------------------
=======
>>>>>>> theirs
# Module-level singleton
# ---------------------------------------------------------------------------

_tracer: CallStackTracer | None = None
_tracer_lock: threading.Lock = threading.Lock()

<<<<<<< ours
_collector: CallStackCollector | None = None
_collector_lock: threading.Lock = threading.Lock()

=======
>>>>>>> theirs

def get_tracer(
    *,
    enabled: bool = True,
    first_step_only: bool = True,
    enable_timing: bool = False,
<<<<<<< ours
    use_npu_timing: bool = False,
    step_interval: int = 1,
=======
>>>>>>> theirs
) -> CallStackTracer:
    """Return (or create) the module-level :class:`CallStackTracer` singleton.

    On first call the arguments configure the singleton.  Subsequent calls
    return the same instance, ignoring the arguments (use the properties on
    the returned object to change behaviour at runtime).
    """
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = CallStackTracer(
                    enabled=enabled,
                    first_step_only=first_step_only,
                    enable_timing=enable_timing,
<<<<<<< ours
                    use_npu_timing=use_npu_timing,
                    step_interval=step_interval,
=======
>>>>>>> theirs
                )
    return _tracer


<<<<<<< ours
def get_collector(
    output_dir: str = "",
    *,
    flush_interval: int = 100,
    enabled: bool = True,
) -> CallStackCollector:
    """Return (or create) the module-level :class:`CallStackCollector` singleton.

    On first call the arguments configure the singleton.  Subsequent calls
    return the same instance.
    """
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = CallStackCollector(
                    output_dir=output_dir,
                    flush_interval=flush_interval,
                    enabled=enabled,
                )
    return _collector


=======
>>>>>>> theirs
@contextlib.contextmanager
def span(
    name: str,
    *,
    timing: bool = False,
    **attrs: Any,
) -> Generator[None, None, None]:
    """Convenience context manager using the global tracer singleton.

    Usage::

        from vllm.utils.callstack import span

        with span("my_block"):
            ...
    """
    t = get_tracer()
    with t.span(name, timing=timing, **attrs):
        yield
