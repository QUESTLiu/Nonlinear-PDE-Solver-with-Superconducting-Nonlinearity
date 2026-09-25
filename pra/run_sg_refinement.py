"""Bounded initialization and size comparisons for local SG correction."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm

from duffing import duffing_padded_hybrid_validation as df
from sine_gordon.run_snapd_end_to_end import build_two_site_problem
from pra.run_pilot import evaluate_sg
from pra.records import provenance,write_json,peak_memory_bytes


def fit_diagnostics(compiler,vector):
    theta,alpha=df.unpack_parameters(vector,compiler.layers,compiler.active_dim)
    output,cache=compiler._forward(theta,alpha)
    states=[compiler.inputs,output]+[state for entry in cache for state in entry[:2]]
    return dict(common_phase_action_mse=float(np.linalg.norm(np.exp(-1j*vector[-1])*output-compiler.targets)**2/compiler.n_states),
                above_active_at_primitive_endpoints=max(float(np.sum(abs(state[compiler.active_dim:])**2,axis=0).max()) for state in states),
                top_eight_at_primitive_endpoints=max(float(np.sum(abs(state[-8:])**2,axis=0).max()) for state in states),
                max_abs_alpha=float(np.hypot(alpha[:,0],alpha[:,1]).max()),
                displacement_components_at_bound=int(np.count_nonzero(np.isclose(abs(alpha),1.5,rtol=0,atol=1e-10))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    args.outdir.mkdir(parents=True,exist_ok=False)
    identity=provenance();started,cpu=time.perf_counter(),time.process_time()
    n,d,k=protocol["training_dimensions"]
    fit_check=protocol.get("fit_check_cutoff",96)
    local={cut:build_two_site_problem(cut,np.sqrt(3),1.,1.,.3)["correction_local"] for cut in sorted({n,fit_check})}
    runs=[];chosen=[];spent=0.
    for r in protocol["outer_steps"]:
        unitary={cut:expm(-1j*(5.2/np.sqrt(3)/r)*ham) for cut,ham in local.items()}
        for layers in protocol["layers"]:
            candidates=[]
            for seed in protocol["seeds"]:
                directory=args.outdir/f"r{r}_L{layers}_seed{seed}";directory.mkdir()
                inputs=np.eye(n,dtype=complex)[:,:k]
                compiler=df.SnapDisplacementCompiler(n,d,layers,inputs,unitary[n]@inputs,1e-8)
                initial_source=protocol.get("initial_parameters_file")
                if initial_source is None:
                    initial=df.paired_frame_parameters(unitary[n],d,layers,seed)
                else:
                    source=Path(initial_source)
                    if hashlib.sha256(source.read_bytes()).hexdigest()!=protocol["initial_parameters_sha256"]:
                        raise ValueError("continuation checkpoint differs from its specified source")
                    with np.load(source) as saved:initial=saved["parameter_vector"].copy()
                remaining=protocol["optimizer_cap_seconds"]-spent
                if remaining<=0:raise RuntimeError("SG refinement optimizer cap reached")
                vector,info=df.optimize_common_phase(compiler,seed,min(900.,remaining),directory/"optimizer.npz",initial_vector=initial,
                                                    maxiter=protocol.get("maxiter",1200),record_history=True)
                spent+=info["elapsed_seconds"]
                diagnostics={str(n):fit_diagnostics(compiler,vector)}
                high_inputs=np.eye(fit_check,dtype=complex)[:,:k]
                high=df.SnapDisplacementCompiler(fit_check,d,layers,high_inputs,unitary[fit_check]@high_inputs,1e-8)
                diagnostics[str(fit_check)]=fit_diagnostics(high,vector)
                info.update(r=r,L=layers,seed=seed,fit_diagnostics=diagnostics,initial_parameters_source=initial_source)
                write_json(directory/"optimization.json",info)
                runs.append(info);candidates.append((info["objective"],seed,vector))
                write_json(args.outdir/"progress.json",dict(phase="training",completed=len(runs),optimization_seconds=spent))
            loss,seed,vector=min(candidates,key=lambda row:row[0])
            chosen.append((r,layers,seed,vector))
    write_json(args.outdir/"selection.json",dict(criterion="minimum training objective within r,L",selected=[dict(r=r,L=l,seed=s) for r,l,s,v in chosen]))
    reference_cache={};configurations=[];validation_started=time.perf_counter()
    for r,layers,seed,vector in chosen:
        result=evaluate_sg(args.outdir,vector,layers,r,args.reference,[("H_SNAP",2,1)],reference_cache,
                           extended_names=True,deadline=validation_started+protocol["validation_cap_seconds"],training_dimensions=(n,d,k))
        configurations.extend(result["configurations"])
        write_json(args.outdir/"progress.json",dict(phase="validation",configurations=len(configurations),optimization_seconds=spent,validation_seconds=time.perf_counter()-validation_started))
    write_json(args.outdir/"summary.json",dict(protocol=protocol,provenance=identity,runs=runs,configurations=configurations,
        training_cutoff=n,active_levels=d,training_columns=k,validation_cutoffs=[96,128],reference_source=str(args.reference),optimization_seconds=spent,
        validation_seconds=time.perf_counter()-validation_started,wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_memory_bytes=peak_memory_bytes(),
        evidence_scope="Post-formal bounded SNAP refinement on the original development physical input. The protocol states changes to initialization, dimensions or iterations. Original runs and thresholds are unchanged. No capacity or family-wide guarantee follows from optimizer termination."))


if __name__=="__main__":main()
