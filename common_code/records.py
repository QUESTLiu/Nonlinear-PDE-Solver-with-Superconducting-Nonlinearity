"""Run identity and bounded numerical output for the PRA comparison."""

import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]


def provenance():
    files = sorted([p for folder in ("common_code", "duffing_code", "sine_gordon_code", "double_sine_gordon_code") for p in (ROOT/folder).glob("*.py")])
    return dict(python=sys.executable, python_version=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                tracked_diff_sha256=hashlib.sha256(subprocess.check_output(["git","diff","--binary"],cwd=ROOT)).hexdigest(),
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def peak_memory_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak*1024)


def write_json(path, value):
    def convert(x):
        if isinstance(x,np.ndarray):
            return x.tolist()
        if isinstance(x,np.generic):
            return x.item()
        if isinstance(x,Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.write_text(json.dumps(value,indent=2,default=convert,allow_nan=False)+"\n")
