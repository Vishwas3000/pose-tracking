"""Desktop wrapper around the offline bundle reader.

The mobile loader will mmap the file and slice sections lazily; this Python
implementation just calls the shared reader. The byte format is identical
(see shared/bundle_format.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reach into offline/tools/bundle_writer for the Bundle / ReferenceView dataclasses.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from offline.tools.bundle_writer import Bundle, ReferenceView, load_bundle  # noqa: E402,F401


def load(path: Path | str) -> Bundle:
    return load_bundle(Path(path))
