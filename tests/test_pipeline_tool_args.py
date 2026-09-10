"""Argument handling of the tools whose machine paths moved to the CLI.

They must import and print --help without torch, and refuse clearly when a
required input is missing instead of falling back to a path that only
exists on machine B.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import audit_colocated_morphology, build_offtrajectory_compare, score_offtrajectory_strips

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_CV2 = importlib.util.find_spec("cv2") is not None


class _NumpyCv2:
    """Just enough of cv2 for the strip scorer, so the CPU channel (where
    opencv is not a locked dependency) still exercises the scoring path
    instead of skipping it. Integer-factor INTER_AREA is an exact block mean;
    the Laplacian uses the same 4-neighbour kernel with reflected borders."""

    CV_64F = 6
    COLOR_BGR2GRAY = 6
    INTER_AREA = 3

    def imread(self, path):
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()

    def imwrite(self, path, image):
        from PIL import Image

        Image.fromarray(image[:, :, ::-1].copy()).save(path)
        return True

    def resize(self, image, size, fx, fy, interpolation):
        factor = int(round(1.0 / fx))
        height = image.shape[0] // factor * factor
        width = image.shape[1] // factor * factor
        block = image[:height, :width].reshape(height // factor, factor, width // factor, factor, 3)
        return block.mean(axis=(1, 3)).round().astype(np.uint8)

    def cvtColor(self, image, code):
        blue, green, red = image[..., 0], image[..., 1], image[..., 2]
        return (0.114 * blue + 0.587 * green + 0.299 * red).round().astype(np.uint8)

    def Laplacian(self, gray, ddepth):
        values = gray.astype(np.float64)
        padded = np.pad(values, 1, mode="reflect")
        return padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * values


class OfftrajectoryCompareArgsTests(unittest.TestCase):
    def test_reference_inputs_are_required(self) -> None:
        parser = build_offtrajectory_compare.build_parser()
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            parser.parse_args(["cfg.json", "ckpt.pt", "out"])
        self.assertIn("--reference-ply", err.getvalue())

    def test_positionals_keep_the_historical_order(self) -> None:
        args = build_offtrajectory_compare.build_parser().parse_args(
            ["cfg.json", "ckpt.pt", "out", "--reference-ply", "ref.ply", "--reference-alignment", "a.json"]
        )
        self.assertEqual((args.config, args.checkpoint, args.output, args.frames), (Path("cfg.json"), Path("ckpt.pt"), Path("out"), 6))
        args = build_offtrajectory_compare.build_parser().parse_args(
            ["cfg.json", "ckpt.pt", "out", "3", "--reference-ply", "ref.ply", "--reference-alignment", "a.json"]
        )
        self.assertEqual(args.frames, 3)

    def test_missing_reference_file_fails_before_torch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "nope.ply"
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code = build_offtrajectory_compare.main(
                    ["cfg.json", "ckpt.pt", "out", "--reference-ply", str(missing), "--reference-alignment", str(missing)]
                )
        self.assertEqual(code, 2)
        self.assertIn("--reference-ply not found", err.getvalue())


class ScoreOfftrajectoryArgsTests(unittest.TestCase):
    def test_at_least_one_pair_is_required(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            score_offtrajectory_strips.build_parser().parse_args([])

    def test_pairs_must_be_name_equals_dir(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            score_offtrajectory_strips.build_parser().parse_args(["just_a_dir"])
        self.assertIn("NAME=DIR", err.getvalue())
        args = score_offtrajectory_strips.build_parser().parse_args(["F6=a/b", "R1=c=d"])
        self.assertEqual(dict(args.arms), {"F6": Path("a/b"), "R1": Path("c=d")})
        self.assertEqual(args.baseline, "F6")
        self.assertIsNone(args.json)

    def test_missing_directory_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code = score_offtrajectory_strips.main([f"X={Path(temporary) / 'absent'}"])
        self.assertEqual(code, 2)
        self.assertIn("X: directory not found", err.getvalue())

    def test_scores_synthetic_strips_and_counts_wins(self) -> None:
        if HAS_CV2:
            import cv2
        else:
            cv2 = _NumpyCv2()
            sys.modules["cv2"] = cv2
            self.addCleanup(sys.modules.pop, "cv2", None)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(0)
            texture = rng.integers(0, 255, size=(32, 48, 3), dtype=np.uint8)
            for name, degrade in (("F6", 40), ("R1", 0)):
                strips = root / name
                strips.mkdir()
                ours = np.clip(texture.astype(np.int16) + degrade, 0, 255).astype(np.uint8)
                strip = np.full((32, 48 * 2 + 8, 3), 24, np.uint8)
                strip[:, :48] = ours
                strip[:, 56:] = texture
                cv2.imwrite(str(strips / "offtraj_00_lateral_1m_abc.png"), strip)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = score_offtrajectory_strips.main([f"F6={root / 'F6'}", f"R1={root / 'R1'}", "--json", str(root / "scores.json")])
            self.assertEqual(code, 0)
            lines = out.getvalue().splitlines()
            self.assertTrue(lines[0].startswith("F6 n 1 median psnr full"))
            self.assertTrue(lines[1].startswith("R1 n 1 median psnr full 100.00  1/4 100.00  1/8 100.00  sharpness ours/ref 1.000"))
            self.assertEqual(lines[2], "R1 wins vs F6 at 1/4 res 1 / 1")
            self.assertTrue((root / "scores.json").exists())


class AuditColocatedArgsTests(unittest.TestCase):
    REQUIRED = ["--checkpoint", "m.pt", "--reference-ply", "r.ply", "--reference-alignment", "a.json"]

    def test_lidar_source_is_required(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            audit_colocated_morphology.build_parser().parse_args(self.REQUIRED)
        self.assertIn("--tile-inputs-root", err.getvalue())

    def test_tile_inputs_root_derives_the_four_tile_clouds(self) -> None:
        args = audit_colocated_morphology.build_parser().parse_args([*self.REQUIRED, "--tile-inputs-root", "inputs"])
        self.assertEqual(
            audit_colocated_morphology.lidar_ply_paths(args),
            [Path("inputs") / f"Tile_{t}" / "initialization_full_lidar.ply" for t in range(4)],
        )
        self.assertEqual(args.label, "ours")
        self.assertEqual(args.voxel, 0.5)
        args = audit_colocated_morphology.build_parser().parse_args([*self.REQUIRED, "--tile-inputs-root", "inputs", "--tiles", "0", "2"])
        self.assertEqual(len(audit_colocated_morphology.lidar_ply_paths(args)), 2)

    def test_explicit_lidar_plys_are_used_verbatim(self) -> None:
        args = audit_colocated_morphology.build_parser().parse_args([*self.REQUIRED, "--lidar-ply", "a.ply", "--lidar-ply", "b.ply"])
        self.assertEqual(audit_colocated_morphology.lidar_ply_paths(args), [Path("a.ply"), Path("b.ply")])

    def test_missing_checkpoint_fails_before_torch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "m.pt"
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code = audit_colocated_morphology.main(
                    ["--checkpoint", str(missing), "--reference-ply", "r.ply", "--reference-alignment", "a.json", "--lidar-ply", "l.ply"]
                )
        self.assertEqual(code, 2)
        self.assertIn("--checkpoint not found", err.getvalue())


class HelpWithoutTorchTests(unittest.TestCase):
    """Each tool must print usage under the plain interpreter (no torch)."""

    TOOLS = (
        "audit_colocated_morphology.py",
        "build_offtrajectory_compare.py",
        "score_offtrajectory_strips.py",
        "checkpoint_morphology.py",
        "pipeline.py",
    )

    def test_help_exits_zero(self) -> None:
        for tool in self.TOOLS:
            with self.subTest(tool=tool):
                completed = subprocess.run(
                    [sys.executable, str(REPO_ROOT / "tools" / tool), "--help"],
                    capture_output=True, text=True, check=False, cwd=str(REPO_ROOT),
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
