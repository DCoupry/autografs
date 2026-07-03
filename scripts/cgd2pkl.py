#!/usr/bin/env python3
"""Deprecated location: the converter lives in autografs.cgd.

Use the installed console command instead::

    autografs-topologies --use_rcsr -o topologies.json.gz

This shim remains so existing instructions keep working.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autografs.cgd import main  # noqa: E402

if __name__ == "__main__":
    main()
