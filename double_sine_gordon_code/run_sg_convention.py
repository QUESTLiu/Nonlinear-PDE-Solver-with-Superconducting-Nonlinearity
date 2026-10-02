"""Compare two finite SG conventions on the same physical preparation."""

import argparse
import time
import numpy as np

from double_sine_gordon_code.run_snapd_end_to_end import build_two_site_problem, exact_trajectory
from sine_gordon_code import native_correction as nc
from common_code.run_pilot import _sg_initial, _sg_observations, embed_trajectory
from common_code.metrics import projective_distance
from common_code.records import provenance, write_json, peak_memory_bytes
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=False)
    identity=provenance();started,cpu=time.perf_counter(),time.process_time()
    states={};observations={};rows=[];horizon=5.2/np.sqrt(3)
    for n in (96,128):
        instance=build_two_site_problem(n,np.sqrt(3),1.,1.,.3,convention="physical_sandwich")
        initial=_sg_initial(instance["basis"])
        full=exact_trajectory(instance["H_target"],initial,horizon,256)
        states[n]=exact_trajectory(instance["H_target"],initial,horizon,10)
        finite_observations=_sg_observations(full,instance["basis"])
        vertex=nc.vertex_sandwich(n,np.sqrt(3))
        physical_rows=[nc.two_site_observables(state.reshape(n,n),instance["basis"],vertex=vertex) for state in full]
        observations[n]={key:np.array([row[key] for row in physical_rows]) for key in ("C12_real","C12_imag","S2")}
        cv=lambda data:data["C12_real"]+1j*data["C12_imag"]
        observable_convention_change=float(np.max(abs(cv(observations[n])-cv(finite_observations))))
        with np.load(args.reference/f"E4_reference_lambda0.3_N{n}.npz") as spectral:
            distance=max(projective_distance(a,b) for a,b in zip(states[n],spectral["target_step_states"]))
            cphys=observations[n]["C12_real"]+1j*observations[n]["C12_imag"]
            cspec=spectral["target_C12_real"]+1j*spectral["target_C12_imag"]
            cdiff=float(np.max(abs(cphys-cspec)))
            sdiff=float(np.max(abs(observations[n]["S2"]-spectral["target_S2"])))
        rows.append(dict(N=n,spectral_to_physical_state_distance=distance,C12_difference=cdiff,S2_difference=sdiff,C12_observable_convention_change_on_physical_states=observable_convention_change,
                         max_boundary_population=max(nc.populations(state.reshape(n,n))["boundary_band_population"] for state in full)))
        np.savez_compressed(args.outdir/f"physical_N{n}.npz",t=np.linspace(0,horizon,257),step_t=np.linspace(0,horizon,11),step_states=states[n],**observations[n])
    state_change=max(projective_distance(a,b) for a,b in zip(embed_trajectory(states[96],96,128),states[128]))
    c=lambda n:observations[n]["C12_real"]+1j*observations[n]["C12_imag"]
    write_json(args.outdir/"summary.json",dict(parameters=dict(E_J=1.,lambda2=.3,kappa=1.,omega_b=float(np.sqrt(3)),T=horizon),
        conventions="spectral Hamiltonian and exp(i phi_N) versus physical-sandwich Hamiltonian and P exp(i phi) P, at fixed basis, input family and horizon",rows=rows,
        physical_cutoff_change=dict(state_distance=state_change,C12=float(np.max(abs(c(96)-c(128)))),S2=float(np.max(abs(observations[96]["S2"]-observations[128]["S2"])))),
        scope="State comparison on 11 common times and observable comparison on 257 times. These empirical differences do not prove infinite-dimensional convergence.",
        wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_memory_bytes=peak_memory_bytes(),provenance=identity))


if __name__=="__main__":main()
