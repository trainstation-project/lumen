"""PyTorch-compatible profiler API (``profile``, ``record_function``, ``key_averages``) over lumen's native profiler."""

import enum
import functools

from lumen.profiler import _C

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
    TPU = "tpu"


def supported_activities():
    """The activities this build and machine can record: CPU, plus CUDA,
    MPS or TPU when such a device is usable (PyTorch: ``supported_activities``).
    Timing CUDA also needs a build with CUPTI."""
    import lumen

    activities = {ProfilerActivity.CPU}
    for activity in (ProfilerActivity.CUDA, ProfilerActivity.MPS, ProfilerActivity.TPU):
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
        "input_shapes",
        "input_strides",
    )

    def __init__(self, fields):
        """Copies each slot's value from the native row dict ``fields``."""
        for name in self.__slots__:
            setattr(self, name, fields[name])

    def __repr__(self):
        """Short summary with the row's key, main times and count."""
        return (
            f"<FunctionEventAvg key={self.key} self_cpu_time={self.self_cpu_time_total:.3f}us "
            f"cpu_time={self.cpu_time_total:.3f}us self_device_time={self.self_device_time_total:.3f}us "
            f"count={self.count}>"
        )


class EventList(list):
    """``key_averages()`` result: a list of :class:`FunctionEventAvg` that
    also renders PyTorch's summary table."""

    def __init__(self, rows, native, group_by_input_shape=False):
        """Wraps ``rows`` and keeps the native profile ``native`` to render the table from."""
        super().__init__(rows)
        self._native = native
        self._group_by_input_shape = group_by_input_shape

    def table(self, sort_by=None, row_limit=100):
        """PyTorch-style table, sorted descending by ``sort_by`` (e.g.
        ``"cpu_time_total"``, ``"self_device_time_total"``; ``cuda_*`` names
        work too) and cut to ``row_limit`` rows (negative: no limit); grouped
        by input shape, with each row's input dtypes and shapes and its
        strides."""
        return self._native.table(sort_by, row_limit, self._group_by_input_shape)


class profile:
    """Profile the code in a ``with`` block (PyTorch:
    ``torch.profiler.profile``). ``activities`` defaults to CPU plus every
    usable device, as in PyTorch. ``record_shapes`` records the dtype and
    shape of each op's inputs and outputs (PyTorch records the inputs'),
    and each kernel's strides: as a plan step reads and writes them."""

    def __init__(self, activities=None, profile_memory=False, record_shapes=False):
        """Picks the activities to record (all supported ones when ``None``) and whether to track memory and shapes."""
        if activities is None:
            activities = supported_activities()
        self.activities = {ProfilerActivity(a) for a in activities}
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self._result = None

    def start(self):
        """Starts recording; events go to the native profiler until :meth:`stop`."""
        _C._profiler_start(
            sorted(a.value for a in self.activities),
            self.profile_memory,
            self.record_shapes,
        )

    def stop(self):
        """Stops recording and keeps the finished profile."""
        self._result = _C._profiler_stop()

    def __enter__(self):
        """Starts profiling and returns the profiler."""
        self.start()
        return self

    def __exit__(self, *exc):
        """Stops profiling; exceptions from the block propagate."""
        self.stop()
        return False

    def _profile(self):
        """Returns the finished native profile, raising if profiling has not stopped yet."""
        if self._result is None:
            raise RuntimeError("the profiler has not finished; use it as a `with` block first")
        return self._result

    def events(self):
        """Every recorded event as a dict (``name``, ``kind`` = op /
        user_range / memory / gpu / host_kernel (a plan step's run on the
        host, inside its op), ``start_us``, ``duration_us``, ``thread``,
        ``parent``, ``device``, ``inputs`` and ``outputs`` (with
        ``record_shapes``: each a list of ``(dtype, shape)``),
        ``input_strides`` and ``output_strides`` (with ``record_shapes``,
        a plan step's: each a list of strides in elements, as its kernel
        reads its inputs and writes its outputs: a slice read in place, an
        in-place concatenate's operand written in its result), ``accum``
        (with ``record_shapes``: the dtypes the op accumulates in, a dot's
        or sum's ``accum_dtype``, a max's or softmax's dtype, each of a
        fusion's reductions'), ``bytes``, ...), by start time. A gpu event
        has the types of the op that issued it."""
        return self._profile().events()

    def key_averages(self, group_by_input_shape=False):
        """Per-name totals (PyTorch: ``prof.key_averages()``); with
        ``group_by_input_shape``, per name and input dtypes, shapes and
        strides (recorded with ``record_shapes``)."""
        native = self._profile()
        rows = [FunctionEventAvg(r) for r in native.key_averages(group_by_input_shape)]
        return EventList(rows, native, group_by_input_shape)

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
        """Prepares a range named ``name``; it starts on ``__enter__``."""
        self.name = name
        self._handle = None

    def __enter__(self):
        """Opens the range in the profiler and as a scope in any active trace."""
        from lumen.graph import tracer

        tracer._enter_scope(self.name)
        self._handle = _C._record_function_enter(self.name)
        return self

    def __exit__(self, *exc):
        """Closes the profiler range and the trace scope."""
        from lumen.graph import tracer

        self._handle.exit()
        self._handle = None
        tracer._exit_scope()
        return False

    def __call__(self, fn):
        """Wraps ``fn`` so each call runs inside this range."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            """Calls ``fn`` inside a fresh range of the same name."""
            with record_function(self.name):
                return fn(*args, **kwargs)

        return wrapper
