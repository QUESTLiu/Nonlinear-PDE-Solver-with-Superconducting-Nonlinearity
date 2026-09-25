"""Same-parameter padding checks of the selected phase-coverage pair."""

import json
import time
import numpy as np
from pra.records import ROOT,provenance,write_json,peak_memory_bytes
from pra.run_duffing_formal import evaluation_inputs,evaluate_pair
from duffing import duffing_padded_hybrid_validation as df


def main():
    out=ROOT/"pra/results/duffing_padding/comparison";out.mkdir(parents=True,exist_ok=False)
    started,cpu=time.perf_counter(),time.process_time();identity=provenance()
    source=ROOT/"pra/results/duffing_refinement/comparison"
    old=json.loads((source/"summary.json").read_text())
    chosen=next(row for row in old["comparisons"] if row["arm"]=="phase_coverage")
    parameter_file=source/f"phase_coverage/seed{chosen['selected_seed']}/optimizer.npz"
    with np.load(parameter_file) as data:vector=data["parameter_vector"].copy()
    fresh=json.loads((ROOT/"pra/duffing_refinement_inputs.json").read_text())
    inputs=[(name,state,"development") for name,state,role in evaluation_inputs()]
    inputs += [(f"fresh_{i:02d}",df.coherent_state(80,complex(*alpha))[0],"independent_validation") for i,alpha in enumerate(fresh["amplitudes_real_imag"])]
    comparisons=[]
    for cuts in ((96,112),(112,128),(128,160)):
        directory=out/f"N{cuts[0]}_{cuts[1]}";directory.mkdir()
        result=evaluate_pair(directory,10,12,vector,inputs,deadline=started+120,boundary_at_both_cutoffs=True,padding_cutoffs=cuts)
        write_json(directory/"summary.json",result);comparisons.append(result)
        write_json(out/"progress.json",dict(completed_cutoffs=cuts,elapsed_seconds=time.perf_counter()-started))
    reproduction=max(abs(row[key]-next(previous for previous in chosen["rows"] if previous["input"]==row["input"] and previous["route"]==row["route"])[key])
                     for row in comparisons[0]["rows"] if row["route"] in ("hybrid","replacement")
                     for key in ("max_common_distance","padding_margin","reference_margin","solver_margin","boundary_population_both_cutoffs","action_sum"))
    write_json(out/"summary.json",dict(provenance=identity,source_parameters=str(parameter_file),comparisons=comparisons,reproduced_default_max_difference=reproduction,
        wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_memory_bytes=peak_memory_bytes(),
        scope="Same selected parameters and all22 previously evaluated inputs. The original96/112 pair checks unchanged default behavior, followed by112/128 and128/160. No new training or independent input evidence."))


if __name__=="__main__":main()
