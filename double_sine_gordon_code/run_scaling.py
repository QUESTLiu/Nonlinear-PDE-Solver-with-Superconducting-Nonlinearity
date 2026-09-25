"""The separately approved B0 timing sample, with no optimizer."""

import argparse
from pathlib import Path
import time
import numpy as np
from double_sine_gordon_code.run_snapd_end_to_end import build_two_site_problem, exact_trajectory
from sine_gordon_code import native_correction as nc, sglib as sg
from common_code.records import provenance, write_json, peak_memory_bytes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=False)
    rows=[]
    for n in (16,24):
        start,cpu=time.perf_counter(),time.process_time()
        problem=build_two_site_problem(n,np.sqrt(3),1.,1.,.3)
        initial=sg.make_initial_condition("coherent_single",2,n,np.sqrt(3),1.,1.,1.,0.,0,False)[0]
        target=exact_trajectory(problem["H_target"],initial,5.2/np.sqrt(3),10)
        for route in ("H_trig","G_trig"):
            for order in (1,2):
                step=nc.make_step(problem,1.,1.,.3,5.2/(10*np.sqrt(3)),route,order,1)
                joint=np.zeros((n,n,2),complex);joint[:,:,0]=initial
                for _ in range(10):
                    joint=step(joint)
                rows.append(dict(cutoff=n,route=route,order=order,**nc.reduced_metrics(joint,target[-1])))
        rows.append(dict(cutoff=n,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
    write_json(args.outdir/"summary.json",dict(scope="B0 scaling sample only, not scientific pilot evidence",rows=rows,provenance=provenance(),peak_memory_bytes=peak_memory_bytes()))


if __name__=="__main__":
    main()
