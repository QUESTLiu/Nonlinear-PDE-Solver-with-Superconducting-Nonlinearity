"""Align the recorded Duffing T2 action sum with its N=96 reference."""

from pathlib import Path
import json
import time
import numpy as np
from duffing import duffing_padded_hybrid_validation as df
from duffing.native_comparison import kerr_snap
from pra.run_duffing_formal import evaluation_inputs
from pra.metrics import projective_distance
from pra.records import ROOT, provenance, write_json, peak_memory_bytes


def main():
    out=ROOT/"pra/results/formal/E2"
    result_path=out/"same_space_action_check.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    selected=json.loads((out/"selection.json").read_text())["selected"]
    complete=json.loads((out/"summary.json").read_text())
    started,cpu=time.perf_counter(),time.process_time();rows=[]
    ops=df.hard_sandwiched_operators(96,1.,.1)
    for choice in selected:
        r,layers,seed=choice["r"],choice["layers"],choice["seed"]
        with np.load(out/f"r{r}_L{layers}_seed{seed}/optimizer.npz") as a:
            vector=a["parameter_vector"]
        gates=df.build_fixed_gates(ops,1/r)
        residual=df.dense_ansatz_unitary(96,48,layers,vector)
        target=df.hermitian_exponential(ops["H_D"],-1j/r)
        sk=kerr_snap(np.diag(ops["H_K"]).real,1/(2*r),48)[1]
        matrices={"hybrid":gates.k_half@gates.v2_half@residual@gates.v2_half@gates.k_half,
                  "replacement":sk@gates.v2_half@residual@gates.v2_half@sk}
        for label,state,role in evaluation_inputs():
            initial=df.embed_state(state,96)
            reference=df.apply_repeated_step(initial,target,r)
            with np.load(out/f"evaluation_r{r}_L{layers}/{label}.npz") as original:
                for route,matrix in matrices.items():
                    trajectory=df.apply_repeated_step(initial,matrix,r)
                    distances=np.array([projective_distance(reference[:,k],trajectory[:,k]) for k in range(r+1)])
                    bounds=np.cumsum(original[route+"_action_errors"])
                    rows.append(dict(r=r,L=layers,input=label,input_role=role,route=route,distances=distances.tolist(),prefix_action_sums=bounds.tolist(),
                                     max_distance_minus_prefix_bound=float(np.max(distances[1:]-bounds))))
    write_json(result_path,dict(reference_cutoff=96,rows=rows,source_run_provenance=complete["provenance"],
               purpose="Compare the same finite target and implementation as T2. N80 reference and cutoff margins remain the primary numerical acceptance data.",
               wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_memory_bytes=peak_memory_bytes(),provenance=provenance()))


if __name__=="__main__":main()
