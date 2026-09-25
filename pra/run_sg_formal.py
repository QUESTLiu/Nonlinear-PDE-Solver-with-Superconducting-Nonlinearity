"""Execute the predefined two-site correction comparison at fixed cutoffs."""

import argparse
from pathlib import Path
import json
import shutil
import time
import numpy as np
from scipy.linalg import expm
from duffing import duffing_padded_hybrid_validation as df
from sine_gordon.run_snapd_end_to_end import build_two_site_problem
from pra.run_pilot import evaluate_sg
from pra.records import provenance, write_json, peak_memory_bytes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--pilot",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text());identity=provenance()
    started,cpu=time.perf_counter(),time.process_time();spent=0.;runs=[];chosen=[]
    problem=build_two_site_problem(48,np.sqrt(3),1.,1.,.3)
    inputs=np.eye(48,dtype=complex)[:,:16]
    for r in (10,20):
        targets=expm(-1j*(5.2/np.sqrt(3)/r)*problem["correction_local"])@inputs
        for layers in (2,4,6):
            candidates=[]
            for seed in (20260729,20260730,20260731):
                directory=args.outdir/f"r{r}_L{layers}_seed{seed}";directory.mkdir()
                if (r,layers,seed)==(10,4,20260729):
                    with np.load(args.pilot/"E4_optimizer.npz") as a:vector=a["parameter_vector"].copy()
                    info=json.loads((args.pilot/"summary.json").read_text())["optimization"]
                    info=dict(info,reused_pilot=True)
                    shutil.copyfile(args.pilot/"E4_optimizer.npz",directory/"optimizer.npz")
                else:
                    compiler=df.SnapDisplacementCompiler(48,24,layers,inputs,targets,1e-8)
                    remaining=16200-spent
                    if remaining<=0:raise RuntimeError("SG optimization cap reached")
                    vector,info=df.optimize_common_phase(compiler,seed,min(900.,remaining),directory/"optimizer.npz")
                spent+=info["elapsed_seconds"]
                write_json(directory/"optimization.json",info)
                runs.append(dict(info,r=r,L=layers,seed=seed));candidates.append((info["objective"],seed,vector))
                write_json(args.outdir/"progress.json",dict(phase="training",completed=len(runs),optimization_seconds=spent))
            loss,seed,vector=min(candidates,key=lambda row:row[0])
            chosen.append((r,layers,seed,vector))
    write_json(args.outdir/"selection.json",dict(criterion="minimum training objective within each r,L",selected=[dict(r=r,L=l,seed=s) for r,l,s,v in chosen]))
    reference_cache={};configurations=[];validation_started=time.perf_counter()
    pilot_summary=json.loads((args.pilot/"summary.json").read_text())
    for point in pilot_summary["configurations"]:
        if point["route"]=="H_SNAP":continue
        original=point["name"]
        name=original.replace(f"_N{point['cutoff']}",f"_r10_N{point['cutoff']}")
        for suffix in (".npz","_schedule.json"):
            shutil.copyfile(args.pilot/(original+suffix),args.outdir/(name+suffix))
        configurations.append(dict(point,name=name,r=10,layers=None,reused_pilot=True,pilot_source=str(args.pilot)))
    for r,layers,seed,vector in chosen:
        configs=[("H_SNAP",2,1)]
        if layers==2:
            configs += [(route,order,s) for route in ("H_trig","G_trig","H_trig_ret") for order in (1,2) for s in (1,2,4,8,16,32) if not (r==10 and s in (1,4))]
        result=evaluate_sg(args.outdir,vector,layers,r,args.reference,configs,reference_cache,extended_names=True,deadline=validation_started+3600)
        configurations.extend(result["configurations"])
        write_json(args.outdir/"progress.json",dict(phase="validation",configurations=len(configurations),optimization_seconds=spent,validation_seconds=time.perf_counter()-validation_started))
        if time.perf_counter()-validation_started>3600:
            raise RuntimeError("SG deterministic validation cap reached")
    write_json(args.outdir/"summary.json",dict(protocol=protocol,provenance=identity,runs=runs,configurations=configurations,
        training_cutoff=48,validation_cutoffs=[96,128],reference_source=str(args.reference),optimization_seconds=spent,
        validation_seconds=time.perf_counter()-validation_started,wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_memory_bytes=peak_memory_bytes(),
        evidence_scope="Protocol-fixed comparison on the development physical input at EJ=1, lambda2=0.3; no independent input-family or hardware claim."))


if __name__=="__main__":main()
