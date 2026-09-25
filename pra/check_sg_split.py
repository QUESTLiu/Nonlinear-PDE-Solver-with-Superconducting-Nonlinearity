"""Compare the exact symmetric correction at the formal 20-step resolution."""

import time
import numpy as np
from sine_gordon.run_snapd_end_to_end import build_two_site_problem
from sine_gordon import native_correction as nc
from pra.run_pilot import _sg_initial
from pra.metrics import projective_distance
from pra.records import ROOT, write_json, provenance, peak_memory_bytes


def main():
    directory=ROOT/"pra/results/sg_formal/exact_split"
    directory.mkdir(parents=True,exist_ok=False)
    started,cpu=time.perf_counter(),time.process_time();rows=[];saved={}
    horizon=5.2/np.sqrt(3)
    for n in (96,128):
        instance=build_two_site_problem(n,np.sqrt(3),1.,1.,.3)
        joint=np.zeros((n,n,2),complex);joint[:,:,0]=_sg_initial(instance["basis"])
        with np.load(ROOT/f"pra/results/sg128/reference/E4_reference_lambda0.3_N{n}.npz") as archive:
            target=archive["target_step_states"]
        step=nc.make_step(instance,1.,1.,.3,horizon/20,"exact_split")
        errors=[];states=[]
        for k in range(21):
            if time.perf_counter()-started>180:raise TimeoutError("exact-split comparison cap reached")
            if k%2==0:
                errors.append(projective_distance(joint[:,:,0],target[k//2]))
                states.append(joint.copy())
            if k<20:joint=step(joint)
        saved[n]=states
        np.savez_compressed(directory/f"N{n}.npz",common_t=np.linspace(0,horizon,11),distance=errors,joint_states=states)
        rows.append(dict(N=n,max_common_distance=max(errors)))
    padded=[]
    for low,high in zip(saved[96],saved[128]):
        embedded=np.zeros_like(high);embedded[:96,:96,:]=low
        padded.append(projective_distance(embedded,high))
    write_json(directory/"summary.json",dict(r=20,rows=rows,padding_distance=max(padded),wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_memory_bytes=peak_memory_bytes(),provenance=provenance()))


if __name__=="__main__":main()
