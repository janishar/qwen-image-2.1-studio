"""The CLI's markers reach the studio whole, though tqdm leaves its bar unterminated on the same pipe.

Run: uv run python web/test_markers.py
"""

import os
import subprocess
import sys
from types import SimpleNamespace

from server import STEP_RE, Runner

# The real marker() beside a real bar that refreshes every step, as a render's slow steps make it.
CLI = """
from tqdm import tqdm
from qwen_image_2_1.generate import marker, stage

stage("denoise")
with tqdm(total=8, mininterval=0, miniters=1) as bar:
    for i in range(8):
        marker(f"@step {i + 1}/8")
        bar.update()
stage("decode")
marker("Saved /tmp/take.png (1024x1024, mode=RGB)")
"""


def main() -> None:
    assert not STEP_RE.match(" 12%|█| 1/8 [00:08<00:58]@step 1/8")  # glued, it is lost

    logged: list[tuple[str, bool]] = []
    steps: list[int] = []
    runner = Runner.__new__(Runner)  # no helmstudio, no queue thread
    runner.state = SimpleNamespace(job_progress=lambda job: None)
    runner.emit = lambda kind, data: steps.append(data["progress"]["step"])
    runner.log = lambda job, line, rewrites=False: logged.append((line, rewrites))
    job = {"id": "x", "progress": {"stage": None, "step": 0, "total": 0}, "result": {}}

    proc = subprocess.Popen(
        [sys.executable, "-c", CLI],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    runner._pump(job, proc.stdout, [])
    assert proc.wait() == 0, logged

    assert job["result"] == {"steps": 8, "width": 1024, "height": 1024, "mode": "RGB"}
    assert [s for s in steps if s] == list(range(1, 9)), steps  # the page's bar moves
    assert job["progress"]["stage"] == "decode"

    bars = [(line, rewrites) for line, rewrites in logged if "/8 [" in line]
    # The bar stays in the log, rewriting itself, and no marker shows there.
    assert any("1/8 [" in line for line, _ in bars), logged
    assert all(rewrites for _, rewrites in bars[:-1]), bars
    assert not any("@" in line for line, _ in logged), logged
    print("ok")


if __name__ == "__main__":
    main()
