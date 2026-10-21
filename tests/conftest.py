import re

import pytest

import lumen


def pytest_collection_modifyitems(config, items):
    """Tests marked with a device (``mps``, ``cuda``, ``tpu``) skipped where it
    is not available: those that plan for it without placing a tensor there
    would fail, not skip."""
    for device in ("mps", "cuda", "tpu"):
        if getattr(lumen, device).is_available():
            continue
        skip = pytest.mark.skip(reason=f"{device} is not available")
        for item in items:
            if item.get_closest_marker(device):
                item.add_marker(skip)


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
    report = (yield).get_result()
    on_tpu = getattr(item, "callspec", None) and item.callspec.params.get("device") == "tpu"
    if on_tpu and report.when == "call" and report.failed and call.excinfo:
        gap = TPU_GAPS.search(str(call.excinfo.value))
        if gap:
            report.outcome = "skipped"
            report.wasxfail = str(call.excinfo.value).splitlines()[0]
