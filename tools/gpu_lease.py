"""Take the pipeline's GPU lease from a shell script.

``tools/pipeline.py`` already guards every GPU step with an ``O_EXCL`` lease on
``<run_root>/gpu.lock``, but the lease only protects work that goes through the pipeline. The
orchestration scripts that drive sweeps and comparisons call the tools directly, so they were
guarded by whatever the script author wrote by hand - and on 2026-09-15 that guard was
``tasklist /v | findstr train_gsplat``, which matches window titles rather than command lines,
never fired, and let a compare render start while a trainer was mid-run. VRAM reached 13.2 of
16.3 GiB before it was stopped.

This exposes the same lease to those scripts:

    rem block until the GPU is free, then run one job under the lease
    python tools/gpu_lease.py --run-root RUN --owner "fill sweep axis20c1" -- ^
        python tools/evaluate_probe_views.py ...

``--wait-seconds`` bounds the wait (default: wait forever). A lease whose holder pid is dead is
stale and reclaimed, so a crashed job does not block the queue. The child's exit code is this
process's exit code, and the lease is released even when the child fails, so a failing step
cannot strand the queue either.

``--check`` exits 0 when the lease is free and 3 when it is held, printing the holder, for a
script that wants to look rather than wait.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pipeline import GpuLeaseBusy, acquire_gpu_lease, read_gpu_lease  # noqa: E402

EXIT_BUSY = 3
EXIT_TIMEOUT = 4


def _describe(record) -> str:
    if not record:
        return "held by an unreadable lease"
    return "held by pid %s (%s) since %s" % (
        record.get("pid"),
        record.get("owner") or "no owner recorded",
        record.get("started_at_text") or "unknown time",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-root", type=pathlib.Path, required=True,
                        help="the run root holding gpu.lock")
    parser.add_argument("--owner", default="", help="what to record as the lease owner")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait-seconds", type=float, default=None,
                        help="give up after this long (default: wait indefinitely)")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--check", action="store_true",
                        help="report whether the lease is free and exit; run nothing")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the command to run while holding the lease, after --")
    args = parser.parse_args(argv)

    lock = args.run_root / "gpu.lock"

    if args.check:
        record = read_gpu_lease(lock)
        if record is None and not lock.exists():
            print("gpu lease free")
            return 0
        print("gpu lease %s" % _describe(record))
        return EXIT_BUSY

    command = [item for item in args.command if item != "--"]
    if not command:
        parser.error("nothing to run: pass the command after --, or use --check")

    deadline = None if args.wait_seconds is None else time.monotonic() + args.wait_seconds
    announced = False
    while True:
        try:
            lease = acquire_gpu_lease(lock, command=command, device=args.device, owner=args.owner)
            break
        except GpuLeaseBusy:
            if deadline is not None and time.monotonic() >= deadline:
                print("gpu lease still %s after %.0f s; giving up"
                      % (_describe(read_gpu_lease(lock)), args.wait_seconds))
                return EXIT_TIMEOUT
            if not announced:
                # Print once, not every poll: an overnight queue should not fill its log with
                # the same line four times a minute.
                print("waiting for the gpu lease, %s" % _describe(read_gpu_lease(lock)),
                      flush=True)
                announced = True
            time.sleep(args.poll_seconds)

    try:
        print("gpu lease taken by %s" % (args.owner or "this process"), flush=True)
        return subprocess.call(command)
    finally:
        lease.release()
        print("gpu lease released", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
