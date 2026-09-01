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
            record = {
                "timestamp_unix": path.stat().st_mtime,
                "completed_steps": step,
                "gaussian_count": len(params.get("means", [])),
                "metrics": training.get("last_metrics", {}),
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
                latest_timestamp = float(latest.get("timestamp_unix", 0.0) or 0.0)
                if manifest:
                    status = "COMPLETE"
                elif step <= 0:
                    status = "NOT_STARTED"
                elif time.time() - latest_timestamp <= 30.0:
                    status = "RUNNING"
                else:
                    status = "PAUSED"
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
    args = parser.parse_args()
    state = MonitorState(
        data_root=args.data_root, cache_path=args.cache, configs=args.config
    )
    html = args.html.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
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
