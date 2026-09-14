"""GPU lease for shell scripts (tools/gpu_lease.py), CPU only.

The pipeline already guards its own GPU steps with an O_EXCL lease, but sweep and comparison
scripts call the tools directly and were guarded by hand-written checks. One of those checks
was ``tasklist /v | findstr train_gsplat``, which matches window titles rather than command
lines, so it never fired and a compare render started while a trainer was mid-run.

What is pinned:

* ``--check`` distinguishes a free lease from a held one by exit code, so a script can look
  without waiting;
* a command runs only after the lease is taken, and the child's exit code is passed through;
* the lease is released even when the child fails, so a failing step cannot strand a queue;
* a lease whose holder pid is dead is reclaimed rather than blocking forever;
* ``--wait-seconds`` gives up with its own exit code instead of hanging an overnight queue;
* refusing to run nothing: no command and no --check is an error, not a silent success.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "gpu_lease.py"
EXIT_BUSY = 3
EXIT_TIMEOUT = 4


def _run(*args, timeout=60):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


class GpuLeaseCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.lock = self.root / "gpu.lock"
        self.addCleanup(self._tmp.cleanup)

    def _hold(self, pid: int, owner: str = "someone else") -> None:
        self.lock.write_text(json.dumps({
            "pid": pid,
            "started_at": time.time(),
            "started_at_text": "2026-09-15 02:44:56",
            "host": "test",
            "device": "cuda:0",
            "owner": owner,
            "command": ["python", "train.py"],
            "command_sha256": "0" * 64,
            "token": "test",
        }), encoding="utf-8")

    def test_check_reports_a_free_lease(self) -> None:
        done = _run("--run-root", str(self.root), "--check")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("free", done.stdout)

    def test_check_reports_a_held_lease_with_its_owner(self) -> None:
        self._hold(os.getpid(), owner="train tile1")
        done = _run("--run-root", str(self.root), "--check")
        self.assertEqual(done.returncode, EXIT_BUSY)
        self.assertIn("train tile1", done.stdout)
        self.assertIn(str(os.getpid()), done.stdout)

    def test_a_free_lease_runs_the_command_and_passes_its_exit_code(self) -> None:
        done = _run("--run-root", str(self.root), "--owner", "sweep", "--",
                    sys.executable, "-c", "print('ran'); raise SystemExit(7)")
        self.assertEqual(done.returncode, 7, done.stderr)
        self.assertIn("ran", done.stdout)
        self.assertIn("gpu lease taken by sweep", done.stdout)
        self.assertIn("gpu lease released", done.stdout)
        self.assertFalse(self.lock.exists(), "a failing child must still release the lease")

    def test_a_held_lease_blocks_the_command_until_it_is_released(self) -> None:
        self._hold(os.getpid(), owner="train tile1")
        marker = self.root / "ran.txt"

        def release_soon():
            time.sleep(2.0)
            self.lock.unlink(missing_ok=True)

        timer = threading.Thread(target=release_soon)
        timer.start()
        started = time.monotonic()
        done = _run("--run-root", str(self.root), "--poll-seconds", "0.25", "--",
                    sys.executable, "-c",
                    "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ok')", str(marker))
        elapsed = time.monotonic() - started
        timer.join()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(marker.is_file(), "the command must not run before the lease is taken")
        self.assertGreaterEqual(elapsed, 1.5, "it should have waited for the holder")
        self.assertIn("waiting for the gpu lease", done.stdout)

    def test_a_lease_whose_holder_is_dead_is_reclaimed(self) -> None:
        # A crashed job must not block the queue. 2**22 is above Windows' and Linux' default
        # pid ranges, so nothing is alive there.
        self._hold(2 ** 22, owner="a trainer that crashed")
        done = _run("--run-root", str(self.root), "--wait-seconds", "10", "--poll-seconds", "0.25",
                    "--", sys.executable, "-c", "print('reclaimed')")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("reclaimed", done.stdout)

    def test_waiting_gives_up_with_its_own_exit_code(self) -> None:
        self._hold(os.getpid(), owner="train tile1")
        done = _run("--run-root", str(self.root), "--wait-seconds", "1", "--poll-seconds", "0.25",
                    "--", sys.executable, "-c", "print('should not run')")
        self.assertEqual(done.returncode, EXIT_TIMEOUT)
        self.assertNotIn("should not run", done.stdout)
        self.assertIn("giving up", done.stdout)

    def test_no_command_and_no_check_is_an_error(self) -> None:
        done = _run("--run-root", str(self.root))
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("nothing to run", done.stderr)


if __name__ == "__main__":
    unittest.main()
