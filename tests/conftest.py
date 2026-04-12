from __future__ import annotations

import sys
from pathlib import Path

# Allow tests to import top-level modules like scripts/ without requiring
# the project to be installed into the active environment first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
