"""The lumen profiler, modeled on ``torch.profiler`` (bindings in
``lumen/profiler/python.rs``, core in ``lumen/profiler/mod.rs``)::

    from lumen.profiler import ProfilerActivity, profile, record_function

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.MPS],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        with record_function("init"):
            x = lumen.zeros([1024, 1024], device="mps")

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")  # open in Perfetto / chrome://tracing

It records lumen's tensor ops (``lumen::zeros``, ``lumen::fill_``, ...) and
``record_function`` ranges on the CPU clock, allocations and frees with
``profile_memory``, and device-side timing of copies and fills for the CUDA
and MPS activities. Times are in microseconds, as in PyTorch.
"""

import enum
import functools

from lumen import _C

__all__ = [
    "ProfilerActivity",
    "profile",
    "record_function",
    "supported_activities",
    "FunctionEventAvg",
    "EventList",
]


class ProfilerActivity(enum.Enum):
    """What to record (PyTorch: ``torch.profiler.ProfilerActivity``)."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def supported_activities():
    """The activities this build and machine can record: CPU, plus CUDA or
    MPS when such a device is usable (PyTorch: ``supported_activities``).
    Timing CUDA also needs a build with CUPTI."""
    import lumen

    activities = {ProfilerActivity.CPU}
    for activity in (ProfilerActivity.CUDA, ProfilerActivity.MPS):
        if activity == ProfilerActivity.CUDA and not _C._profiler_cuda_timing():
            continue
        try:
            lumen.empty([0], device=activity.value)
        except RuntimeError:
            continue
        activities.add(activity)
    return activities


class FunctionEventAvg:
    """One row of ``key_averages()``: totals for all events with one name
    (PyTorch: ``FunctionEventAvg``). Times in microseconds, memory in bytes;
    "self" excludes nested ops."""

    __slots__ = (
        "key",
        "is_device_event",
        "count",
        "cpu_time_total",
        "self_cpu_time_total",
        "cpu_time",
        "device_time_total",
        "self_device_time_total",
        "device_time",
        "cpu_memory_usage",
        "self_cpu_memory_usage",
        "device_memory_usage",
        "self_device_memory_usage",
    )

    def __init__(self, fields):
        for name in self.__slots__:
            setattr(self, name, fields[name])

    def __repr__(self):
        return (
            f"<FunctionEventAvg key={self.key} self_cpu_time={self.self_cpu_time_total:.3f}us "
            f"cpu_time={self.cpu_time_total:.3f}us self_device_time={self.self_device_time_total:.3f}us "
            f"count={self.count}>"
        )


class EventList(list):
    """``key_averages()`` result: a list of :class:`FunctionEventAvg` that
    also renders PyTorch's summary table."""

    def __init__(self, rows, native):
        super().__init__(rows)
        self._native = native

    def table(self, sort_by=None, row_limit=100):
        """PyTorch-style table, sorted descending by ``sort_by`` (e.g.
        ``"cpu_time_total"``, ``"self_device_time_total"``; ``cuda_*`` names
        work too) and cut to ``row_limit`` rows (negative: no limit)."""
        return self._native.table(sort_by, row_limit)


class profile:
    """Profile the code in a ``with`` block (PyTorch:
    ``torch.profiler.profile``). ``activities`` defaults to CPU plus every
    usable device, as in PyTorch. ``record_shapes`` records the dtype and
    shape of each op's inputs and outputs (PyTorch records the inputs')."""

    def __init__(self, activities=None, profile_memory=False, record_shapes=False):
        if activities is None:
            activities = supported_activities()
        self.activities = {ProfilerActivity(a) for a in activities}
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self._result = None

    def start(self):
        _C._profiler_start(
            sorted(a.value for a in self.activities),
            self.profile_memory,
            self.record_shapes,
        )

    def stop(self):
        self._result = _C._profiler_stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def _profile(self):
        if self._result is None:
            raise RuntimeError("the profiler has not finished; use it as a `with` block first")
        return self._result

    def events(self):
        """Every recorded event as a dict (``name``, ``kind`` = op /
        user_range / memory / gpu / host_kernel (a plan step's run on the
        host, inside its op), ``start_us``, ``duration_us``, ``thread``,
        ``parent``, ``device``, ``inputs`` and ``outputs`` (with
        ``record_shapes``: each a list of ``(dtype, shape)``), ``accum``
        (with ``record_shapes``: the dtypes the op accumulates in, a dot's
        or sum's ``accum_dtype``, a max's or softmax's dtype, each of a
        fusion's reductions'), ``bytes``, ...), by start time. A gpu event
        has the types of the op that issued it."""
        return self._profile().events()

    def key_averages(self):
        """Per-name totals (PyTorch: ``prof.key_averages()``)."""
        native = self._profile()
        return EventList([FunctionEventAvg(r) for r in native.key_averages()], native)

    def export_chrome_trace(self, path):
        """Write a Chrome trace, viewable in Perfetto or ``chrome://tracing``
        (PyTorch: ``prof.export_chrome_trace``)."""
        with open(path, "w") as f:
            f.write(self._profile().chrome_trace())


class record_function:
    """Time a named range; lumen ops inside it nest under it (PyTorch:
    ``torch.profiler.record_function``). Use as a ``with`` block or a
    decorator. Free when no profiler is running.

    Inside a compiled function (while ``lumen.compile`` traces it), the ops
    traced in the range are in it: when the plan runs, the profiler shows
    their steps (kernels) inside a range of that name, one each time the
    plan runs through them, as uncompiled code would."""

    def __init__(self, name):
        self.name = name
        self._handle = None

    def __enter__(self):
        from lumen.graph import tracer  # it imports lumen.profiler's package

        tracer._enter_scope(self.name)
        self._handle = _C._record_function_enter(self.name)
        return self

    def __exit__(self, *exc):
        from lumen.graph import tracer

        self._handle.exit()
        self._handle = None
        tracer._exit_scope()
        return False

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with record_function(self.name):
                return fn(*args, **kwargs)

        return wrapper
