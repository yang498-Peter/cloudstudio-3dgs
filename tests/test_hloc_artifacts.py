from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cloudstudio_3dgs.ba.hloc_artifacts import (
    FEATURES_NAME,
    MATCHES_NAME,
    RUNTIME_MANIFEST_NAME,
    prepare_hloc_output,
)


class HlocArtifactTests(unittest.TestCase):
    def test_fresh_and_overwrite_modes_return_upstream_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hloc"
            self.assertFalse(prepare_hloc_output(root))
            (root / FEATURES_NAME).write_bytes(b"partial")
            self.assertTrue(prepare_hloc_output(root, overwrite=True))

    def test_resume_accepts_only_known_unsigned_partial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hloc"
            root.mkdir()
            (root / FEATURES_NAME).write_bytes(b"features")
            (root / MATCHES_NAME).write_bytes(b"matches")
            self.assertFalse(prepare_hloc_output(root, resume=True))

    def test_resume_rejects_default_unknown_and_completed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hloc"
            root.mkdir()
            (root / FEATURES_NAME).write_bytes(b"features")
            with self.assertRaisesRegex(FileExistsError, "pass --resume or --overwrite"):
                prepare_hloc_output(root)
            (root / "unexpected.tmp").write_bytes(b"unknown")
            with self.assertRaisesRegex(ValueError, "unknown artifacts"):
                prepare_hloc_output(root, resume=True)
            (root / "unexpected.tmp").unlink()
            (root / RUNTIME_MANIFEST_NAME).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already signed complete"):
                prepare_hloc_output(root, resume=True)

    def test_resume_rejects_matches_without_features_and_conflicting_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hloc"
            root.mkdir()
            (root / MATCHES_NAME).write_bytes(b"matches")
            with self.assertRaisesRegex(ValueError, "without the feature artifact"):
                prepare_hloc_output(root, resume=True)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                prepare_hloc_output(root, overwrite=True, resume=True)


if __name__ == "__main__":
    unittest.main()
