"""The env script loader must survive argv quoting on Windows.

The first version passed `call "<path>" && set` as one argument; Python's
CreateProcess quoting turned the inner quotes into \\" and cmd answered
"is not recognized" with exit 1, so every real run died before its first
step. A generated wrapper batch file is what the loader runs now.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tools.pipeline import PipelineError, load_env_script


@unittest.skipUnless(os.name == "nt", "cmd env scripts only exist on Windows")
class EnvScriptLoaderTests(unittest.TestCase):
    def test_variables_set_by_the_script_are_captured(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "env with space.cmd"
            script.write_text("@echo off\r\nset CS3DGS_TEST_MARKER=marker-value\r\necho noise\r\n", encoding="ascii")
            env = load_env_script(script)
        self.assertEqual(env.get("CS3DGS_TEST_MARKER"), "marker-value")
        self.assertIn("PATH", env)

    def test_failing_script_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "broken.cmd"
            script.write_text("@echo off\r\nexit /b 3\r\n", encoding="ascii")
            with self.assertRaises(PipelineError):
                load_env_script(script)


if __name__ == "__main__":
    unittest.main()
