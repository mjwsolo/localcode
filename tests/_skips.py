"""Skip markers for tests whose feature is genuinely POSIX- or macOS-only."""
import sys

import pytest

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only: shell script, file modes or ps/signals")
