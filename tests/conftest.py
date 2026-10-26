"""Pytest hooks: skip tests for missing devices, skip known TPU gaps and pin each xdist worker to one chip."""

import json
import os
import re
import subprocess
import sys

import pytest

import lumen


def pytest_collection_modifyitems(config, items):
    """Tests marked with a device (``mps``, ``cuda``, ``tpu``) skipped where it
    is not available: those that plan for it without placing a tensor there
    would fail, not skip."""
    for device in ("mps", "cuda", "tpu"):
        if available(config, device):
            continue
        skip = pytest.mark.skip(reason=f"{device} is not available")
        for item in items:
            if item.get_closest_marker(device):
                item.add_marker(skip)
    if hasattr(config, "workerinput"):
        skip = pytest.mark.skip(reason="a worker sees one device of each kind: run it without -n")
        for item in items:
            if item.get_closest_marker("all_devices"):
                item.add_marker(skip)


# Error messages of features the TPU backend doesn't support yet; failing TPU tests matching one are skipped.
TPU_GAPS = re.compile(
    r"(does|do) not run on TPU yet"
    r"|is not stored on a TPU"
    r"|moves elements between tiles"
    r"|an operand whose rows are not its tiles' rows"
    r"|a strided view on tpu"
    r"|TPU programs read and write whole rows"
    r"|operands of one dtype accumulating in float32"
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Turns a TPU-parametrized test's failure into a skip when it hits a known `TPU_GAPS` message."""
    report = (yield).get_result()
    on_tpu = getattr(item, "callspec", None) and item.callspec.params.get("device") == "tpu"
    if on_tpu and report.when == "call" and report.failed and call.excinfo:
        gap = TPU_GAPS.search(str(call.excinfo.value))
        if gap:
            report.outcome = "skipped"
            report.wasxfail = str(call.excinfo.value).splitlines()[0]


# Environment variables that restrict a process to one chip of each device kind, per xdist worker.
PINNED = {
    "tpu": lambda chip, worker: {
        "TPU_VISIBLE_CHIPS": str(chip),
        "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_PROCESS_PORT": str(8476 + worker),
    },
    "cuda": lambda chip, worker: {"CUDA_VISIBLE_DEVICES": str(chip)},
}


def device_counts(config):
    """Device count per kind in `PINNED`, probed once in a subprocess so this process isn't pinned to a chip."""
    if not hasattr(config, "device_counts"):
        probe = "import json, lumen; print(json.dumps({d: getattr(lumen, d).device_count() for d in %r}))"
        run = [sys.executable, "-c", probe % list(PINNED)]
        out = subprocess.run(run, capture_output=True, text=True, check=True).stdout
        config.device_counts = json.loads(out.splitlines()[-1])
    return config.device_counts


def available(config, device):
    """Whether `device` can hold tensors, using the controller's counts on xdist workers."""
    if device not in PINNED:
        return getattr(lumen, device).is_available()
    worker = getattr(config, "workerinput", None)
    counts = worker["device_counts"] if worker else device_counts(config)
    return counts[device] > 0


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    """Defaults `-n auto` to one worker per chip of the first present device kind, else per CPU."""
    counts = [n for n in device_counts(config).values() if n]
    return counts[0] if counts else os.cpu_count()


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """Hands the controller's device counts to each new xdist worker."""
    node.workerinput["device_counts"] = device_counts(node.config)


def pytest_configure(config):
    """Pins an xdist worker to chip `worker id % count` of each device kind before lumen initialises it."""
    worker = getattr(config, "workerinput", None)
    if worker is None:
        return
    k = int(worker["workerid"].removeprefix("gw"))
    for device, count in worker["device_counts"].items():
        if count:
            os.environ.update(PINNED[device](k % count, k))


@pytest.fixture
def triton_on_cuda(device):
    """``use_triton_backend`` on for CUDA, whose plans run with Triton: its
    GEMM and generated loops."""
    lumen.config.compiler.cuda.use_triton_backend = device == "cuda"
    yield
    lumen.config.compiler.cuda.use_triton_backend = False
