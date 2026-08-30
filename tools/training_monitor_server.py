#!/usr/bin/env python3
"""Serve a read-only multi-run CloudStudio 3DGS training dashboard."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _gpu_status() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        row = subprocess.run(
            command, capture_output=True, text=True, timeout=2, check=True
        ).stdout.strip().splitlines()[0]
        values = [part.strip() for part in row.split(",")]
        return {
            "utilization_percent": float(values[0]),
            "memory_used_mib": float(values[1]),
            "memory_total_mib": float(values[2]),
            "temperature_c": float(values[3]),
            "power_w": float(values[4]),
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def _morphology(params: dict) -> dict:
    """Shape and opacity distributions, which the per-step metrics never carry.

    The step metrics report loss terms; whether the population is actually
    thin, disk-shaped and opaque enough to survive a background change is a
    property of the parameters themselves, and only a checkpoint has those.
    """
    scales = params.get("scales")
    opacities = params.get("opacities")
    if scales is None or opacities is None:
        return {}
    try:
        import numpy as np
        import torch

        scale_m = torch.exp(torch.as_tensor(scales).detach().float()).numpy()
        opacity = torch.sigmoid(
            torch.as_tensor(opacities).detach().float().reshape(-1)
        ).numpy()
        if scale_m.ndim != 2 or scale_m.shape[1] != 3:
            return {}
        ordered = np.sort(scale_m, axis=1)
        shortest, middle, longest = ordered[:, 0], ordered[:, 1], ordered[:, 2]
        ratio = longest / np.maximum(shortest, 1e-12)
        tangent = longest / np.maximum(middle, 1e-12)
        return {
            "shape_longest_axis_p50_mm": float(np.percentile(longest, 50) * 1000.0),
            "shape_shortest_axis_p50_mm": float(np.percentile(shortest, 50) * 1000.0),
            "shape_axis_ratio_p50": float(np.percentile(ratio, 50)),
            "shape_tangent_ratio_p50": float(np.percentile(tangent, 50)),
            "shape_opacity_p50": float(np.percentile(opacity, 50)),
            "shape_opacity_below_0p1_fraction": float(np.mean(opacity < 0.1)),
        }
    except (ImportError, RuntimeError, ValueError, TypeError):
        return {}


class CompareRunner:
    """One-click three-way comparison, at most one render at a time.

    The render is a subprocess so a CUDA failure there can never take the
    dashboard down with it; state is what the page polls.
    """

    def __init__(self, command_path: Path | None, output_dir: Path | None) -> None:
        self.command_path = command_path
        self.output_dir = output_dir
        self.lock = threading.Lock()
        self.state: dict = {"state": "idle"}

    def start(self) -> dict:
        if self.command_path is None or not self.command_path.is_file():
            return {"accepted": False, "reason": "compare command is not configured"}
        with self.lock:
            if self.state.get("state") == "running":
                return {"accepted": False, "reason": "a render is already running"}
            self.state = {"state": "running", "started_unix": time.time()}
        threading.Thread(target=self._run, daemon=True).start()
        return {"accepted": True}

    def _run(self) -> None:
        try:
            completed = subprocess.run(
                ["cmd", "/c", str(self.command_path)],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            tail = "\n".join(
                (completed.stdout + "\n" + completed.stderr).strip().splitlines()[-12:]
            )
            with self.lock:
                self.state = {
                    "state": "done" if completed.returncode == 0 else "error",
                    "returncode": completed.returncode,
                    "finished_unix": time.time(),
                    "output_tail": tail,
                }
        except (OSError, subprocess.SubprocessError) as error:
            with self.lock:
                self.state = {
                    "state": "error",
                    "finished_unix": time.time(),
                    "output_tail": str(error),
                }

    def status(self) -> dict:
        with self.lock:
            status = dict(self.state)
        images: list[dict] = []
        summary: dict = {}
        if self.output_dir is not None and self.output_dir.is_dir():
            for path in sorted(self.output_dir.glob("*.png")):
                images.append(
                    {"name": path.name, "mtime_unix": path.stat().st_mtime}
                )
            summary = _read_json(self.output_dir / "compare_summary.json")
        status["images"] = images
        status["summary"] = summary
        return status

    def image_bytes(self, name: str) -> bytes | None:
        if self.output_dir is None or "/" in name or "\\" in name or ".." in name:
            return None
        path = self.output_dir / name
        if not path.is_file() or path.suffix.lower() != ".png":
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None


class MonitorState:
    def __init__(self, *, data_root: Path, cache_path: Path, configs: list[Path]) -> None:
        self.data_root = data_root.resolve()
        self.cache_path = cache_path.resolve()
        self.config_paths = [path.resolve() for path in configs]
        self.lock = threading.Lock()
        self.cache = _read_json(self.cache_path)
        self.cache.setdefault("series", {})
        self.checkpoint_mtimes: dict[str, int] = {}

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)

    def _configs(self) -> list[dict]:
        values: list[dict] = []
        for path in self.config_paths:
            value = _read_json(path)
            if value.get("run_id") and value.get("output_dir"):
                value["_config_path"] = str(path)
                values.append(value)
        return values

    def _progress_jsonl(self, run_dir: Path) -> list[dict]:
        path = run_dir / "monitor" / "progress.jsonl"
        if not path.is_file():
            return []
        records: list[dict] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        except (OSError, ValueError):
            return []
        return records

    def _checkpoint_record(self, run_id: str, run_dir: Path) -> dict | None:
        path = run_dir / "checkpoints" / "latest.pt"
        if not path.is_file():
            return None
        mtime = path.stat().st_mtime_ns
        if self.checkpoint_mtimes.get(run_id) == mtime:
            series = self.cache["series"].get(run_id, [])
            return None if not series else series[-1]
        try:
            import torch

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            training = checkpoint.get("training_state", {})
            params = checkpoint.get("params", {})
            step = int(checkpoint.get("step", 0))
            metrics = dict(training.get("last_metrics", {}))
            metrics.update(_morphology(params))
            record = {
                "timestamp_unix": path.stat().st_mtime,
                "completed_steps": step,
                "gaussian_count": len(params.get("means", [])),
                "metrics": metrics,
                "latest_optimization_audit": (
                    training.get("optimization_audits", [])[-1]
                    if training.get("optimization_audits")
                    else None
                ),
                "source": "checkpoint",
            }
            self.checkpoint_mtimes[run_id] = mtime
            return record
        except (OSError, RuntimeError, ValueError, EOFError):
            return None

    @staticmethod
    def _merge_series(existing: list[dict], incoming: list[dict]) -> list[dict]:
        by_step = {
            int(record.get("completed_steps", 0)): record
            for record in existing
            if int(record.get("completed_steps", 0)) > 0
        }
        for record in incoming:
            step = int(record.get("completed_steps", 0))
            if step > 0:
                by_step[step] = record
        return [by_step[step] for step in sorted(by_step)]

    def snapshot(self) -> dict:
        with self.lock:
            runs: list[dict] = []
            cache_changed = False
            for config in self._configs():
                run_id = str(config["run_id"])
                run_dir = Path(config["output_dir"])
                incoming = self._progress_jsonl(run_dir)
                checkpoint = self._checkpoint_record(run_id, run_dir)
                if checkpoint is not None:
                    incoming.append(checkpoint)
                existing = self.cache["series"].get(run_id, [])
                merged = self._merge_series(existing, incoming)
                if merged != existing:
                    self.cache["series"][run_id] = merged
                    cache_changed = True
                manifest = _read_json(run_dir / "run_manifest.json")
                latest = merged[-1] if merged else {}
                step = int(
                    manifest.get("training", {}).get(
                        "completed_steps", latest.get("completed_steps", 0)
                    )
                )
                max_steps = int(config.get("max_steps", 0))
                status = (
                    "COMPLETE"
                    if manifest
                    else ("RUNNING_OR_PAUSED" if step else "NOT_STARTED")
                )
                runs.append(
                    {
                        "run_id": run_id,
                        "output_dir": str(run_dir),
                        "config_path": config["_config_path"],
                        "status": status,
                        "completed_steps": step,
                        "max_steps": max_steps,
                        "progress_fraction": 0.0
                        if max_steps <= 0
                        else min(1.0, step / max_steps),
                        "phase": latest.get("phase")
                        or latest.get("metrics", {}).get("optimization_phase"),
                        "topology_mode": config.get("topology_policy", {}).get("mode"),
                        "gaussian_count": latest.get("gaussian_count"),
                        "latest_metrics": latest.get("metrics", {}),
                        "latest_optimization_audit": latest.get(
                            "latest_optimization_audit"
                        ),
                        "series": merged,
                        "checkpoint_mtime_unix": (
                            (run_dir / "checkpoints" / "latest.pt").stat().st_mtime
                            if (run_dir / "checkpoints" / "latest.pt").is_file()
                            else None
                        ),
                    }
                )
            self.cache["updated_at_unix"] = time.time()
            if cache_changed:
                self._write_cache()
            return {
                "schema_version": 1,
                "updated_at_unix": time.time(),
                "refresh_seconds": 5,
                "gpu": _gpu_status(),
                "runs": runs,
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--config", action="append", default=[], type=Path)
    parser.add_argument("--html", type=Path, default=Path(__file__).with_name("training_monitor.html"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument(
        "--compare-cmd",
        type=Path,
        help="batch file that renders the photo/ours/reference strips",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        help="directory the compare command writes its PNG strips into",
    )
    args = parser.parse_args()
    state = MonitorState(
        data_root=args.data_root, cache_path=args.cache, configs=args.config
    )
    compare = CompareRunner(args.compare_cmd, args.compare_dir)
    html = args.html.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route == "/api/compare/run":
                payload = json.dumps(
                    compare.start(), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route == "/api/compare/status":
                payload = json.dumps(
                    compare.status(), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route.startswith("/compare/"):
                image = compare.image_bytes(route[len("/compare/"):])
                if image is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(image)))
                self.end_headers()
                self.wfile.write(image)
                return
            if route == "/api/status":
                payload = json.dumps(
                    state.snapshot(), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            self.send_error(404)

        def log_message(self, format: str, *values: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"training monitor ready: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
