"""Load a prebuilt gsplat JIT module before running a Python entry point."""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


def preload_gsplat(extension_path: Path) -> object:
    # On Windows, importing torch registers its bundled DLL directories before
    # the extension loader resolves c10/torch CUDA dependencies.  Loading the
    # prebuilt module first otherwise fails with an unhelpful missing-DLL error.
    import torch  # noqa: F401

    extension_path = Path(extension_path).resolve()
    if not extension_path.is_file():
        raise FileNotFoundError(f"prebuilt gsplat extension is missing: {extension_path}")
    spec = importlib.util.spec_from_file_location("gsplat_cuda", extension_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prebuilt gsplat extension: {extension_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gsplat_cuda"] = module
    spec.loader.exec_module(module)
    # gsplat's normal import first asks for the packaged `gsplat.csrc` and only
    # then falls back to JIT. Alias the verified JIT module so no Ninja cache
    # metadata is consulted after a successful direct link.
    sys.modules["gsplat.csrc"] = module
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("script", nargs="?")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    preload_gsplat(args.extension)
    import gsplat

    if args.probe:
        print(f"GSPLAT_READY {gsplat.__version__}")
        return 0
    if args.script is None:
        parser.error("script is required unless --probe is used")
    script = Path(args.script).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Python entry point is missing: {script}")
    sys.argv = [str(script), *args.script_args]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
