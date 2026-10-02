"""Bounded iteration and phase-coverage refinements of the Duffing fit."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from duffing_code import duffing_padded_hybrid_validation as df
from duffing_code.run_duffing_formal import evaluation_inputs,evaluate_pair
from common_code.records import write_json,provenance,peak_memory_bytes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text())
    args.outdir.mkdir(parents=True,exist_ok=False)
    started,cpu=time.perf_counter(),time.process_time();identity=provenance()
    ops=df.hard_sandwiched_operators(96,1.,.1);gates=df.build_fixed_gates(ops,.1)
    target_step=df.hermitian_exponential(ops["H_D"],-.1j)
    spent=0.;runs=[];chosen=[]
    for arm in protocol["arms"]:
        amplitudes=[radius*np.exp(1j*phase) for radius in (1.,1.5,2.25) for phase in arm["training_phases"]]
        x,y,_=df.build_training_snapshots(96,amplitudes,10,gates,target_step,80)
        directory=args.outdir/arm["name"];directory.mkdir()
        np.savez_compressed(directory/"training.npz",inputs=x,targets=y,initial_amplitudes=np.array(amplitudes))
        candidates=[]
        for seed in arm["seeds"]:
            run_dir=directory/f"seed{seed}";run_dir.mkdir()
            compiler=df.SnapDisplacementCompiler(96,48,arm["layers"],x,y,1e-8)
            initial=(df.paired_frame_parameters(gates.u4_exact,48,arm["layers"],seed)
                     if arm["initialization"]=="paired_frames" else None)
            remaining=protocol["optimizer_cap_seconds"]-spent
            if remaining<=0:raise RuntimeError("Duffing refinement optimizer cap reached")
            vector,info=df.optimize_common_phase(compiler,seed,min(900.,remaining),run_dir/"optimizer.npz",
                                                initial_vector=initial,maxiter=5000,record_history=True)
            spent+=info["elapsed_seconds"]
            info.update(arm=arm["name"],r=10,L=arm["layers"],training_columns=x.shape[1])
            write_json(run_dir/"optimization.json",info);runs.append(info)
            candidates.append((info["objective"],seed,vector))
            write_json(args.outdir/"progress.json",dict(phase="training",completed=len(runs),optimization_seconds=spent))
        loss,seed,vector=min(candidates,key=lambda row:row[0])
        chosen.append((arm["name"],arm["layers"],seed,vector))
    write_json(args.outdir/"selection.json",dict(criterion="minimum training objective within each arm",selected=[dict(arm=a,L=l,seed=s) for a,l,s,v in chosen]))
    # All original evaluation inputs now have a development role in this new study.
    old_inputs=[(name,state,"development") for name,state,role in evaluation_inputs()]
    fresh_file=Path(protocol["independent_inputs_file"])
    if hashlib.sha256(fresh_file.read_bytes()).hexdigest()!=protocol["independent_inputs_sha256"]:
        raise ValueError("independent input specification changed after protocol definition")
    fresh=json.loads(fresh_file.read_text())
    new_inputs=[(f"fresh_{i:02d}",df.coherent_state(80,complex(*alpha))[0],"independent_validation") for i,alpha in enumerate(fresh["amplitudes_real_imag"])]
    comparisons=[];validation_started=time.perf_counter()
    for name,layers,seed,vector in chosen:
        directory=args.outdir/f"evaluation_{name}";directory.mkdir()
        comparison=evaluate_pair(directory,10,layers,vector,old_inputs+new_inputs,
                                 deadline=validation_started+protocol["validation_cap_seconds"],boundary_at_both_cutoffs=True)
        comparison.update(arm=name,selected_seed=seed)
        comparison["scope_acceptance"]={scope:{route:{str(t):all(row["acceptance"][str(t)] for row in comparison["rows"] if row["route"]==route and row["input_role"]==scope) for t in (1e-2,1e-3,1e-4)} for route in ("hybrid","replacement")} for scope in ("development","independent_validation")}
        write_json(directory/"summary.json",comparison);comparisons.append(comparison)
        if time.perf_counter()-validation_started>protocol["validation_cap_seconds"]:
            raise RuntimeError("Duffing refinement validation cap reached")
    write_json(args.outdir/"summary.json",dict(protocol=protocol,provenance=identity,runs=runs,comparisons=comparisons,
        optimization_seconds=spent,validation_seconds=time.perf_counter()-validation_started,
        wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_memory_bytes=peak_memory_bytes(),
        evidence_scope="The iteration arm changes only maxiter relative to its archived seed. The coverage arm jointly changes training phases, depth, initialization and iteration allowance. The fresh twelve-state coherent family is evaluated only after both selections, with no further tuning."))


if __name__=="__main__":main()
