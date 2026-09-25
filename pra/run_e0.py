"""Record the authorized small-dimension algebra and implementation checks."""

import io
import time
import unittest
from pathlib import Path

from pra import test_e0
from pra.records import provenance, peak_memory_bytes, write_json


def main():
    started, cpu = time.perf_counter(), time.process_time()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_e0)
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    target=Path(__file__).parent/"records"
    target.mkdir(exist_ok=True)
    (target/"e0.log").write_text(stream.getvalue())
    record=dict(scope="E0 finite algebra and implementation checks; no scientific pilot",provenance=provenance(),
                dimensions=[8,12,16],passed=result.wasSuccessful(),tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),checks=test_e0.RECORDS,
                wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                peak_memory_bytes=peak_memory_bytes())
    write_json(target/"e0.json",record)
    print(stream.getvalue())
    print(record["provenance"]["python"],record["provenance"]["python_version"])
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
