"""Build-only compatibility for localized Windows compiler output."""

import os


if os.name == "nt":
    try:
        import torch.utils.cpp_extension as cpp_extension

        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    except Exception:
        # The build process will report the original import/compiler error.
        pass
