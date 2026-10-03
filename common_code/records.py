"""Run identity and bounded numerical output for the PRA comparison."""

import json
from pathlib import Path
import platform
import resource
import sys
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]


def provenance():
    return dict(
        python=sys.executable,
        python_version=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


def peak_memory_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    path = Path(path)
    path.write_text(
        json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n"
    )
