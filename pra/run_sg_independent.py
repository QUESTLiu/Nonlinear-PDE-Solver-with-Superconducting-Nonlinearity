"""One fixed second-input check after SG circuit selection."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp

from sine_gordon.run_snapd_end_to_end import build_two_site_problem,exact_trajectory,embed_trajectory
from pra.run_pilot import _sg_initial,_sg_observations,evaluate_sg
from pra.metrics import projective_distance
from pra.records import write_json,provenance,peak_memory_bytes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text())
    args.outdir.mkdir(parents=True,exist_ok=False)
    reference=args.outdir/"reference";reference.mkdir()
    comparison=args.outdir/"comparison";comparison.mkdir()
    started,cpu=time.perf_counter(),time.process_time();identity=provenance()
    preparation=tuple(protocol["preparation"])
    write_json(args.outdir/"selection.json",protocol)
    horizon=5.2/np.sqrt(3);times=np.linspace(0,horizon,257);step_times=np.linspace(0,horizon,21)
    cache={};observations={};step_observations={};records=[]
    c=lambda data:data["C12_real"]+1j*data["C12_imag"]
    for n in (128,160):
        problem=build_two_site_problem(n,np.sqrt(3),1.,1.,.3)
        initial=_sg_initial(problem["basis"],*preparation)
        exact=exact_trajectory(problem["H_target"],initial,horizon,256)
        observed=_sg_observations(exact,problem["basis"]);observations[n]=observed
        steps=exact_trajectory(problem["H_target"],initial,horizon,20)
        step_observed=_sg_observations(steps,problem["basis"]);step_observations[n]=step_observed
        cache[(n,20,preparation)]=steps;cache[(n,10,preparation)]=steps[::2]
        solver=None;solver_arrays={}
        if n==160:
            check_times=np.unique(np.concatenate((times,step_times)))
            checked=solve_ivp(lambda t,x:-1j*(problem["H_target"]@x),(0.,horizon),initial.ravel(),method="DOP853",t_eval=check_times,rtol=2e-11,atol=2e-13)
            if not checked.success:raise RuntimeError(checked.message)
            observed_indices=np.searchsorted(check_times,times);step_indices=np.searchsorted(check_times,step_times)
            solver_observed=_sg_observations(checked.y.T[observed_indices],problem["basis"])
            solver_step_observed=_sg_observations(checked.y.T[step_indices],problem["basis"])
            solver_distances=np.array([projective_distance(a,b) for a,b in zip(exact,checked.y.T[observed_indices])])
            solver_step_distances=np.array([projective_distance(a,b) for a,b in zip(steps,checked.y.T[step_indices])])
            solver=dict(state_distance=float(max(solver_distances.max(),solver_step_distances.max())),
                        C12=float(max(np.max(abs(c(observed)-c(solver_observed))),np.max(abs(c(step_observed)-c(solver_step_observed))))),
                        S2=float(max(np.max(abs(observed["S2"]-solver_observed["S2"])),np.max(abs(step_observed["S2"]-solver_step_observed["S2"])))))
            solver_arrays={"solver_20step_states":checked.y.T[step_indices].copy(),"solver_state_distances_257":solver_distances,"solver_state_distances_21":solver_step_distances,
                           **{"solver_"+key:value for key,value in solver_observed.items()},**{"solver_step_"+key:value for key,value in solver_step_observed.items()}}
            del checked,solver_observed
        np.savez_compressed(reference/f"E4_reference_lambda0.3_N{n}.npz",t=times,step_t=step_times[::2],full_step_t=step_times,target_step_states=steps[::2],target_20step_states=steps,
                            **{"target_"+key:value for key,value in observed.items()},**{"target_step_"+key:value for key,value in step_observed.items()},**solver_arrays)
        records.append(dict(N=n,lambda2=.3,solver=solver))
        del exact,problem
        write_json(args.outdir/"progress.json",dict(phase="reference",completed_cutoff=n,elapsed_seconds=time.perf_counter()-started))
    margins=dict(state_distance=max(projective_distance(a,b) for a,b in zip(embed_trajectory(cache[(128,20,preparation)],128,160),cache[(160,20,preparation)])),
                 C12=float(max(np.max(abs(c(observations[128])-c(observations[160]))),np.max(abs(c(step_observations[128])-c(step_observations[160]))))),
                 S2=float(max(np.max(abs(observations[128]["S2"]-observations[160]["S2"])),np.max(abs(step_observations[128]["S2"]-step_observations[160]["S2"])))))
    solver=records[-1]["solver"]
    resolved=margins["state_distance"]+solver["state_distance"]<=np.sqrt(.001)/10 and margins["C12"]+solver["C12"]<=.001 and margins["S2"]+solver["S2"]<=.005
    write_json(reference/"summary.json",dict(preparation=list(preparation),compared_cutoffs=[128,160],records=records,empirical_cutoff_estimates=margins,resolution_ok=resolved,provenance=identity))
    if not resolved:raise RuntimeError("second-input reference does not resolve the unchanged criteria")
    configurations=[]
    for group in protocol["groups"]:
        n,d,k=group["training_dimensions"];layers=group["layers"]
        if group["parameter_file"] is None:
            vector=np.zeros(layers*d+2*layers+1)
        else:
            with np.load(group["parameter_file"]) as data:vector=data["parameter_vector"].copy()
        result=evaluate_sg(comparison,vector,layers,group["r"],reference,[tuple(row) for row in group["configurations"]],cache,
                           extended_names=True,deadline=started+protocol["wall_cap_seconds"],validation_cutoffs=(128,160),
                           training_dimensions=(n,d,k),preparation=preparation)
        configurations.extend(result["configurations"])
        write_json(args.outdir/"progress.json",dict(phase="circuits",completed=len(configurations),elapsed_seconds=time.perf_counter()-started))
    write_json(args.outdir/"summary.json",dict(protocol=protocol,provenance=identity,preparation=list(preparation),configurations=configurations,
        reference=margins,solver=solver,validation_cutoffs=[128,160],wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_memory_bytes=peak_memory_bytes(),
        evidence_scope="One second coherent preparation declared before refinement, evaluated after fixed circuit selection without retuning. This does not establish uniform input-family accuracy."))


if __name__=="__main__":main()
