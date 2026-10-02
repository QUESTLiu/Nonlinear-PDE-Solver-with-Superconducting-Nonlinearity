"""Execute a reviewed run manifest with per-process resource caps.

Without --execute this prints commands and makes no run directories. Human
approval of a manifest is external to this program. The script does not infer
approval from the existence of a manifest or a previous run.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common_code.records import ROOT, write_json, provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--batch",choices=["B0","B1","B2","B3-D","B3-S"],required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    jobs=[job for job in manifest["jobs"] if job["batch"]==args.batch]
    if not args.execute:
        for job in jobs:
            print(job["id"],job["wall_cap_seconds"],"seconds:"," ".join(job["command"]))
        return
    output=ROOT/manifest["output_root"]
    output.mkdir(parents=True,exist_ok=True)
    run_ledger=output/manifest.get("execution_record",f"{args.batch}_execution.json")
    if run_ledger.exists():
        raise FileExistsError("a batch record exists; preserve it and use a new reviewed manifest for a rerun")
    results=[]
    env=os.environ.copy()
    env.update({name:"1" for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS")})
    for job in jobs:
        condition=job.get("requires_summary_true")
        if condition:
            path,key=condition
            if not (ROOT/path).exists() or not json.loads((ROOT/path).read_text()).get(key,False):
                results.append(dict(id=job["id"],status="not_run",reason="continuation evidence absent or insufficient"))
                continue
        started=time.monotonic()
        command=[str(ROOT/x[2:]) if x.startswith("./") else x for x in job["command"]]
        with (output/(job["id"]+".log")).open("x") as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            reason=None;peak=0
            while process.poll() is None:
                elapsed=time.monotonic()-started
                rss=subprocess.run(["/bin/ps","-o","rss=","-p",str(process.pid)],capture_output=True,text=True)
                if rss.returncode==0 and rss.stdout.strip():
                    peak=max(peak,int(rss.stdout.strip())*1024)
                storage=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                if elapsed>job["wall_cap_seconds"]:
                    reason="wall_time_cap"
                elif peak>manifest["memory_cap_bytes"]:
                    reason="memory_cap"
                elif storage>manifest["storage_cap_bytes"]:
                    reason="storage_cap"
                if reason:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait()
                    break
                time.sleep(.5)
            result=dict(id=job["id"],status="completed" if process.returncode==0 else "budget_limited" if reason else "failed",returncode=process.returncode,reason=reason,
                        wall_seconds=time.monotonic()-started,peak_rss_bytes=peak,command=command)
            results.append(result)
            write_json(run_ledger,dict(batch=args.batch,results=results,provenance=provenance()))
            if result["status"]!="completed":
                break
    write_json(run_ledger,dict(batch=args.batch,results=results,provenance=provenance()))
    if any(result["status"] in ("failed","budget_limited") for result in results):
        raise SystemExit(1)


if __name__=="__main__":
    main()
