"""Run fixed native-error and two-site correction comparisons.

The batch runner supplies process time, memory and output limits. The additional
Duffing validation inputs are handled separately by run_duffing_formal.
"""

import argparse
import json
import shutil
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from duffing_code import duffing_padded_hybrid_validation as df
from duffing_code.native_comparison import constructive_pair, moments, moment_operators, kerr_snap, residual_factors
from sine_gordon_code import sglib as sg
from sine_gordon_code import native_correction as nc
from double_sine_gordon_code.run_snapd_end_to_end import build_two_site_problem, exact_trajectory, embed_trajectory
from common_code.metrics import projective_distance, projective_distances, native_residual_integral, native_verdict, family_action_sum, first_crossing
from common_code.records import provenance, write_json, peak_memory_bytes


def save_arrays(out, name, **arrays):
    np.savez_compressed(out/f"{name}.npz",**arrays)


def e1(out,formal=False):
    summaries=[]
    cutoffs=(48,64,80) if formal else (48,64)
    for beta in ((0.,.02,.05,.10,.20) if formal else (.02,.05,.10)):
        operators={n:df.hard_sandwiched_operators(n,1.,beta) for n in cutoffs}
        eigen={n:np.linalg.eigh(operators[n]["H_D"]) for n in operators}
        conditions=[(xi,1.) for xi in ((.5,1.,1.75,2.,2.5) if formal else (.5,2.))]
        if formal and beta==.1:
            conditions.extend([(1.75,2*np.pi),(2.,2*np.pi)])
        for xi,horizon in conditions:
            times=np.linspace(0,horizon,401)
            data={}
            for n,ops in operators.items():
                psi,_=df.coherent_state(n,xi)
                vals,vecs=eigen[n]
                target=vecs@((vecs.conj().T@psi)[:,None]*np.exp(-1j*vals[:,None]*times))
                native=psi[:,None]*np.exp(-1j*np.diag(ops["H_K"]).real[:,None]*times)
                residual=ops["H_D"]-ops["H_K"]
                bounds=native_residual_integral(residual,native,times)
                fine_times=np.linspace(0,horizon,801)
                fine_native=psi[:,None]*np.exp(-1j*np.diag(ops["H_K"]).real[:,None]*fine_times)
                refined=native_residual_integral(residual,fine_native,fine_times)
                quadrature_difference=float(abs(refined["centered_integral"][-1]-bounds["centered_integral"][-1]))
                distance=projective_distances(target,native)
                infidelity=distance**2*(1-distance**2/4)
                diagonal_ceiling=np.sum(abs(target)*abs(psi[:,None]),axis=0)**2
                prepared=moment_operators(n,1.)
                m=[moments(target[:,i],1.,beta,ops,prepared) for i in range(len(times))]
                q=np.array([x["q"] for x in m]);p=np.array([x["p"] for x in m])
                classical=solve_ivp(lambda t,z:[z[1],-z[0]-beta*z[0]**3],(0.,horizon),[q[0],p[0]],t_eval=times,method="DOP853",rtol=2e-10,atol=2e-12)
                if not classical.success:
                    raise RuntimeError(classical.message)
                qstar=max(1.,abs(q[0]),abs(p[0]))
                mf=np.maximum(abs(q-classical.y[0]),abs(p-classical.y[1]))/qstar
                name=f"E1_beta{beta:g}_xi{xi:g}_N{n}"+("_long" if horizon>1 else "")
                save_arrays(out,name,t=times,infidelity=infidelity,diagonal_ceiling=diagonal_ceiling,q=q,p=p,classical=classical.y,mean_field_error=mf,
                            **bounds,**{key:np.array([x[key] for x in m]) for key in ("variance_q","central_third_q","closure_p","cutoff_q","cutoff_p")})
                data[n]=dict(target=target,native=native,bound=float(bounds["centered_integral"][-1]),quadrature=quadrature_difference,infidelity=float(infidelity.max()),name=name,mean_field_crossing=first_crossing(times,mf,.1))
            lower,high=data[cutoffs[-2]],data[cutoffs[-1]]
            target_difference=max(projective_distance(df.embed_state(lower["target"][:,i],cutoffs[-1]),high["target"][:,i]) for i in range(len(times)))
            native_difference=max(projective_distance(df.embed_state(lower["native"][:,i],cutoffs[-1]),high["native"][:,i]) for i in range(len(times)))
            previous_target_difference=None
            if formal:
                previous_target_difference=max(projective_distance(df.embed_state(data[48]["target"][:,i],64),data[64]["target"][:,i]) for i in range(len(times)))
            bound_difference=abs(high["bound"]-lower["bound"])
            amplitude_margin=bound_difference+high["quadrature"]+target_difference+native_difference
            verdicts=[native_verdict(high["bound"],threshold,amplitude_margin=amplitude_margin,observed_infidelity=high["infidelity"],infidelity_uncertainty=2*(target_difference+native_difference)) for threshold in (1e-3,1e-2,1e-4)]
            summaries.append(dict(beta=beta,xi=xi,T=horizon,cutoffs=list(cutoffs),native_bound=high["bound"],max_native_infidelity=high["infidelity"],target_cutoff_difference=target_difference,previous_target_cutoff_difference=previous_target_difference,native_cutoff_difference=native_difference,
                                  quadrature_difference=high["quadrature"],bound_cutoff_difference=bound_difference,verdicts=verdicts,mean_field_crossing=high["mean_field_crossing"],
                                  evidence="finite-family numerical estimates; cutoff and quadrature differences are empirical, not rigorous physical margins",records=[data[n]["name"]+".npz" for n in cutoffs]))
    return dict(points=summaries,primary_threshold=1e-3,development=not formal)


def e2(out):
    n_ref,n_pad,d,r,layers=80,96,48,10,8
    h=.1
    ops=df.hard_sandwiched_operators(n_pad,1.,.1)
    gates=df.build_fixed_gates(ops,h)
    exact=df.hermitian_exponential(ops["H_D"],-1j*h)
    inputs,targets,rows=df.build_training_snapshots(n_pad,[1.,1.5,2.25],r,gates,exact,n_ref)
    compiler=df.SnapDisplacementCompiler(n_pad,d,layers,inputs,targets,1e-8)
    vector,optimization=df.optimize_common_phase(compiler,240731,900.,out/"E2_optimizer.npz")
    write_json(out/"E2_training.json",rows)
    summaries=[]
    ref_ops={n:df.hard_sandwiched_operators(n,1.,.1) for n in (64,80)}
    times=np.linspace(0,1,11)
    restricted=None
    for xi in (1.75,2.):
        reference={n:df.evolve_hermitian(ref_ops[n]["H_D"],df.coherent_state(n,xi)[0],times) for n in ref_ops}
        reference_difference=max(projective_distance(df.embed_state(reference[64][:,j],80),reference[80][:,j]) for j in range(11))
        independent=expm_multiply(-1j*ref_ops[80]["H_D"],df.coherent_state(80,xi)[0],start=0.,stop=1.,num=11,traceA=-1j*np.trace(ref_ops[80]["H_D"]))
        solver_difference=max(projective_distance(reference[80][:,j],independent[j]) for j in range(11))
        pairs={};step_errors={}
        for n in (96,112):
            local_ops=ops if n==96 else df.hard_sandwiched_operators(n,1.,.1)
            local_gates=gates if n==96 else df.build_fixed_gates(local_ops,h)
            residual=df.dense_ansatz_unitary(n,d,layers,vector)
            initial=df.embed_state(df.coherent_state(n_ref,xi)[0],n)
            pairs[n]=constructive_pair(local_ops,local_gates,residual,initial,h,r,d,residual_factors(n,d,layers,vector))
            if n==96:
                w=local_gates.k_half@local_gates.v2_half@residual@local_gates.v2_half@local_gates.k_half
                sk_half=kerr_snap(np.diag(local_ops["H_K"]).real,h/2,d)[1]
                sk=sk_half@local_gates.v2_half@residual@local_gates.v2_half@sk_half
                for name,matrix in (("hybrid",w),("replacement",sk)):
                    step_errors[name]=family_action_sum(lambda a,matrix=matrix:matrix@a,lambda a:exact@a,initial,r,phase=float(vector[-1]))
                access=np.eye(n)[:,:24]
                if restricted is None:
                    restricted={name:float(np.linalg.norm((matrix-np.exp(1j*vector[-1])*exact)@access,2)) for name,matrix in (("hybrid",w),("replacement",sk))}
                controls={"native":df.diagonal_kerr_trajectory(local_ops["H_K"],initial,times),
                          "gaussian":df.apply_repeated_step(initial,local_gates.k_v2_step,r),
                          "exact_split":df.apply_repeated_step(initial,local_gates.exact_split_step,r)}
        pair_summary={}
        arrays={"t":times,"reference_access_population":np.sum(abs(reference[80][24:,:])**2,axis=0)}
        for name in ("hybrid","replacement"):
            states=pairs[96][name]
            errors=np.array([projective_distance(df.embed_state(reference[80][:,j],96),states[:,j]) for j in range(11)])
            pad_difference=max(projective_distance(df.embed_state(states[:,j],112),pairs[112][name][:,j]) for j in range(11))
            total=float(errors.max()+reference_difference+solver_difference+pad_difference)
            boundary=pairs[96]["gate_populations"][name]["boundary_population"]
            pair_summary[name]=dict(max_distance=float(errors.max()),empirical_reference_margin=reference_difference,empirical_solver_margin=solver_difference,empirical_padding_margin=pad_difference,
                                    total_amplitude_estimate=total,max_boundary_gate_population=boundary,max_above_active_gate_population=pairs[96]["gate_populations"][name]["above_active_population"],action_sum=float(step_errors[name].sum()),rectangular_action_norm=restricted[name],
                                    thresholds={str(t):bool(total<=np.sqrt(t) and boundary<=1e-6) for t in (1e-2,1e-3,1e-4)})
            arrays[name+"_distance"]=errors
            arrays[name+"_action_errors"]=step_errors[name]
        for name,states in controls.items():
            arrays[name+"_distance"]=np.array([projective_distance(df.embed_state(reference[80][:,j],96),states[:,j]) for j in range(11)])
        save_arrays(out,f"E2_xi{xi:g}",**arrays)
        write_json(out/f"E2_xi{xi:g}_replacement_locations.json",{str(n):pairs[n]["locations"] for n in pairs})
        summaries.append(dict(xi=xi,routes=pair_summary,replacement_bound=pairs[96]["replacement_bound"],extra_snap_count=11,
                              caveat="development finite-family validation with empirical margins; finite-matrix primitive endpoints do not bound a physical pulse's continuous-time path"))
    return dict(optimization=optimization,points=summaries,development=True,
                resource_counts=dict(system_modes=1,residual_snap=r*layers,residual_displacements=r*layers,gaussian_half_pulses=2*r,native_waits_after_merging=r+1,extra_replacement_snap=r+1,native_simulated_time=1.,physical_duration="requires K_phys and pulse durations"))


def _sg_initial(basis,phi0=1.,pi0=0.):
    return sg.make_initial_condition("coherent_single",2,basis.n_cut,basis.omega,1.,1.,phi0,pi0,0,False)[0]


def _sg_observations(states,basis):
    rows=[nc.two_site_observables(s.reshape(basis.n_cut,basis.n_cut),basis) for s in states]
    return {key:np.array([row[key] for row in rows]) for key in ("C12_real","C12_imag","S2")}


def e4_reference(out,cutoffs=(32,48),previous=None,lambdas=(0.,.3)):
    horizon=5.2/np.sqrt(3)
    records={}
    if previous is not None:
        prior=json.loads((previous/"summary.json").read_text())
        prior_cut=max(entry["N"] for entry in prior["records"])
        for entry in prior["records"]:
            if entry["N"]!=prior_cut:
                continue
            with np.load(previous/(entry["name"]+".npz")) as data:
                observations={route:{key:data[route+"_"+key] for key in ("C12_real","C12_imag","S2")} for route in ("target","native","retuned_native")}
                records[(entry["lambda2"],prior_cut)]=dict(name=str(previous/(entry["name"])),observations=observations,steps=data["target_step_states"],summary=entry["routes"],solver=entry["solver"],comparator_solver=entry["comparator_solver"],mean_field_crossing=entry["mean_field_crossing"],amplitude=entry["classical_amplitude"])
    for lam in lambdas:
        for n in cutoffs:
            problem=build_two_site_problem(n,np.sqrt(3),1.,1.,lam)
            basis=problem["basis"];initial=_sg_initial(basis)
            target=exact_trajectory(problem["H_target"],initial,horizon,256)
            times=np.linspace(0,horizon,257)
            trajectories={"target":target}
            if lam:
                trajectories["native"]=exact_trajectory(problem["H_native"],initial,horizon,256)
                retuned=build_two_site_problem(n,np.sqrt(3),1+4*lam,1.,0.)
                trajectories["retuned_native"]=exact_trajectory(retuned["H_native"],initial,horizon,256)
            else:
                trajectories["native"]=target;trajectories["retuned_native"]=target
            observations={name:_sg_observations(values,basis) for name,values in trajectories.items()}
            prepared=nc.prepare_moment_diagnostics(basis,1.,lam,np.zeros(2))
            moment_rows=[nc.moment_diagnostics(state.reshape(n,n),basis,1.,1.,lam,lambda x:(problem["H_target"]@x.ravel()).reshape(n,n),prepared=prepared) for state in target]
            diagnostics={key:np.array([row[key] for row in moment_rows]) for key in moment_rows[0]}
            classical_phi,classical_pi=sg.classical_sine_gordon(diagnostics["mean_phi"][0],diagnostics["mean_pi"][0],1.,1.,times,False,lambda2=lam)
            amplitude=float(np.max(np.ptp(classical_phi,axis=0))/2)
            if amplitude==0:
                amplitude=1.
            mf=np.max(abs(diagnostics["mean_phi"]-classical_phi),axis=1)/amplitude
            solver=solve_ivp(lambda t,x:-1j*(problem["H_target"]@x),(0.,horizon),initial.ravel(),method="DOP853",t_eval=times,rtol=2e-11,atol=2e-13)
            if not solver.success:
                raise RuntimeError(solver.message)
            solver_states=solver.y.T
            solver_distance=max(projective_distance(a,b) for a,b in zip(target,solver_states))
            solver_obs=_sg_observations(solver_states,basis)
            complex_c=lambda x:x["C12_real"]+1j*x["C12_imag"]
            solver_c=float(np.max(abs(complex_c(observations["target"])-complex_c(solver_obs))))
            solver_s=float(np.max(abs(observations["target"]["S2"]-solver_obs["S2"])))
            comparator_solver={}
            if lam:
                for route,ham in (("native",problem["H_native"]),("retuned_native",retuned["H_native"])):
                    checked=solve_ivp(lambda t,x:-1j*(ham@x),(0.,horizon),initial.ravel(),method="DOP853",t_eval=times,rtol=2e-11,atol=2e-13)
                    if not checked.success:
                        raise RuntimeError(checked.message)
                    checked_obs=_sg_observations(checked.y.T,basis)
                    comparator_solver[route]=dict(state_distance=max(projective_distance(a,b) for a,b in zip(trajectories[route],checked.y.T)),C12=float(np.max(abs(complex_c(observations[route])-complex_c(checked_obs)))),S2=float(np.max(abs(observations[route]["S2"]-checked_obs["S2"]))))
            target_steps=exact_trajectory(problem["H_target"],initial,horizon,10)
            joint=np.zeros((n,n,2),complex);joint[:,:,0]=initial
            step=nc.make_step(problem,1.,1.,lam,horizon/10,"exact_split")
            split=[joint[:,:,0].ravel()]
            for _ in range(10):
                joint=step(joint);split.append(joint[:,:,0].ravel())
            arrays={"t":times,"step_t":np.linspace(0,horizon,11),"target_step_states":target_steps,"exact_split_step_states":np.array(split),"classical_phi":classical_phi,"classical_pi":classical_pi,"mean_field_error":mf,**diagnostics}
            arrays["reference_access_population"]=np.array([nc.populations(state.reshape(n,n))["outside_joint_access"] for state in target])
            arrays["reference_boundary_population"]=np.array([nc.populations(state.reshape(n,n))["boundary_band_population"] for state in target])
            info={}
            for name,states in trajectories.items():
                for key,value in observations[name].items():
                    arrays[name+"_"+key]=value
                distances=np.array([projective_distance(x,y) for x,y in zip(target,states)])
                arrays[name+"_distance"]=distances
                info[name]=dict(max_distance=float(distances.max()))
            info["exact_split"]={"max_distance":max(projective_distance(x,y) for x,y in zip(target_steps,split))}
            name=f"E4_reference_lambda{lam:g}_N{n}"
            save_arrays(out,name,**arrays)
            records[(lam,n)]=dict(name=name,observations=observations,steps=target_steps,summary=info,solver=dict(state_distance=solver_distance,C12=solver_c,S2=solver_s),comparator_solver=comparator_solver,mean_field_crossing=first_crossing(times,mf,.1),amplitude=amplitude)
    available=sorted(n for lam,n in records if lam==.3)
    low_cut,high_cut=available[-2:]
    low,high=records[(.3,low_cut)],records[(.3,high_cut)]
    embedded=embed_trajectory(low["steps"],low_cut,high_cut)
    state_margin=max(projective_distance(a,b) for a,b in zip(embedded,high["steps"]))
    c=lambda d:d["C12_real"]+1j*d["C12_imag"]
    target_obs=high["observations"]["target"]
    c_margin=float(np.max(abs(c(target_obs)-c(low["observations"]["target"]))))
    s_margin=float(np.max(abs(target_obs["S2"]-low["observations"]["target"]["S2"])))
    effects={name:float(np.max(abs(c(target_obs)-c(high["observations"][name])))) for name in ("native","retuned_native")}
    combined={name:c_margin+float(np.max(abs(c(high["observations"][name])-c(low["observations"][name]))))+high["solver"]["C12"]+high["comparator_solver"][name]["C12"] for name in effects}
    resolved=any(effects[name]>5*combined[name] for name in effects)
    state_total=state_margin+high["solver"]["state_distance"]
    c_total=c_margin+high["solver"]["C12"]
    s_total=s_margin+high["solver"]["S2"]
    previous_difference=None
    if len(available)>2:
        prev=available[-3]
        previous_difference=max(projective_distance(a,b) for a,b in zip(embed_trajectory(records[(.3,prev)]["steps"],prev,low_cut),low["steps"]))
    return dict(development=True,T=horizon,omega_b=float(np.sqrt(3)),Omega_time=float(np.sqrt(3)),
                records=[dict(lambda2=lam,N=n,name=entry["name"],routes=entry["summary"],solver=entry["solver"],comparator_solver=entry["comparator_solver"],mean_field_crossing=entry["mean_field_crossing"],classical_amplitude=entry["amplitude"]) for (lam,n),entry in records.items()],
                empirical_cutoff_estimates=dict(state_distance=state_margin,C12=c_margin,S2=s_margin),C12_effects=effects,
                compared_cutoffs=[low_cut,high_cut],previous_state_cutoff_difference=previous_difference,
                combined_signal_uncertainty=combined,empirical_total_estimates=dict(state_distance=state_total,C12=c_total,S2=s_total),
                continuation_signal_resolved=bool(resolved and c_total<=.001 and s_total<=.005 and state_total<=np.sqrt(1e-3)/10),
                continuation_scope="cutoff comparison and independent DOP853 reference check; development numerical evidence")


def e4_compiled(out,previous=None,optimizer_source=None):
    n,d,layers,r=48,24,4,10
    horizon=5.2/np.sqrt(3);h=horizon/r
    problem=build_two_site_problem(n,np.sqrt(3),1.,1.,.3)
    inputs=np.eye(n,dtype=complex)[:,:16]
    targets=expm(-1j*h*problem["correction_local"])@inputs
    compiler=df.SnapDisplacementCompiler(n,d,layers,inputs,targets,1e-8)
    if optimizer_source is None:
        vector,optimization=df.optimize_common_phase(compiler,20260729,900.,out/"E4_optimizer.npz")
    else:
        with np.load(optimizer_source/"E4_optimizer.npz") as data:
            if data["status"].item() not in ("solver_success","solver_terminated","budget_limited"):
                raise ValueError("optimizer checkpoint is incomplete")
            vector=data["parameter_vector"].copy()
            optimization={key:data[key].item() for key in ("seed","objective","evaluations","elapsed_seconds","status")}
        bounds=np.array([np.pi]*(layers*d)+[1.5]*(2*layers)+[np.pi])
        optimization.update(iterations=None,cpu_seconds=None,message="Completed checkpoint reused after interruption of downstream validation; iteration and CPU counters were not serialized.",constraint_violation=float(np.maximum(abs(vector)-bounds,0).max()),common_phase=float(vector[-1]),method="SLSQP",maxiter=1200,ftol=1e-12,regularization=1e-8,source=str(optimizer_source),training_provenance=json.loads((optimizer_source/"identity.json").read_text()))
        shutil.copyfile(optimizer_source/"E4_optimizer.npz",out/"E4_optimizer.npz")
    configs=[("H_SNAP",2,1)]+[(route,order,s) for route in ("H_trig","G_trig","H_trig_ret") for order in (1,2) for s in (1,4)]
    return dict(development=True,optimization=optimization,**evaluate_sg(out,vector,layers,r,previous,configs))


def evaluate_sg(out,vector,layers,r,previous,configs,reference_cache=None,extended_names=False,deadline=None,validation_cutoffs=None,reuse_low=None,training_dimensions=(48,24,16),preparation=(1.,0.)):
    if r not in (10,20) or (r==20 and previous is None):
        raise ValueError("use 10 steps, or 20 steps with the archived converged reference")
    n,d,access=training_dimensions
    horizon=5.2/np.sqrt(3);h=horizon/r
    problem=build_two_site_problem(n,np.sqrt(3),1.,1.,.3)
    reference_cache={} if reference_cache is None else reference_cache
    selected=[]
    references={}
    low_cut,high_cut=validation_cutoffs or ((96,128) if previous is not None else (48,64))
    previous_summary=json.loads((previous/"summary.json").read_text()) if previous is not None else None
    if previous_summary is not None and tuple(previous_summary.get("preparation",(1.,0.)))!=tuple(preparation):
        raise ValueError("reference preparation does not match the circuit input")
    if reuse_low is not None and tuple(reuse_low.get("preparation",(1.,0.)))!=tuple(preparation):
        raise ValueError("reused circuit preparation does not match the requested input")
    for cutoff in (low_cut,high_cut):
        instance=problem if cutoff==n else build_two_site_problem(cutoff,np.sqrt(3),1.,1.,.3)
        basis=instance["basis"];initial=_sg_initial(basis,*preparation)
        key=(cutoff,r,tuple(preparation))
        if key in reference_cache:
            target=reference_cache[key]
        elif previous is not None and r==10:
            with np.load(previous/f"E4_reference_lambda0.3_N{cutoff}.npz") as archive:
                target=archive["target_step_states"].copy()
            reference_cache[key]=target
        else:
            target=exact_trajectory(instance["H_target"],initial,horizon,r)
            reference_cache[key]=target
        references[cutoff]=target
        local=df.dense_ansatz_unitary(cutoff,d,layers,vector)
        for route,order,s in configs:
            suffix=(f"_r{r}_L{layers}" if route=="H_SNAP" else f"_r{r}") if extended_names else ""
            name=f"E4_{route}_o{order}_s{s}{suffix}_N{cutoff}"
            if cutoff==low_cut and reuse_low is not None:
                reused=next(point for point in reuse_low["configurations"] if point["name"]==name)
                for extension in (".npz","_schedule.json"):
                    shutil.copyfile(reuse_low["directory"]/(name+extension),out/(name+extension))
                selected.append(dict(reused,reused_same_parameter_record=str(reuse_low["directory"])))
                continue
            joint=np.zeros((cutoff,cutoff,2),complex);joint[:,:,0]=initial
            maximum_populations=nc.populations(joint,access=access,active=d)
            def observe(state):
                for key,value in nc.populations(state,access=access,active=d).items():
                    maximum_populations[key]=max(maximum_populations[key],value)
            factors=residual_factors(cutoff,d,layers,vector) if route=="H_SNAP" else None
            step=nc.make_step(instance,1.,1.,.3,h,route,order,s,local_correction=local,observer=observe,local_factors=factors)
            rows=[];t2=[];states=[]
            phase=2*vector[-1] if route=="H_SNAP" else 0.
            for k in range(r+1):
                if deadline is not None and time.perf_counter()>deadline:
                    raise TimeoutError("SG validation time cap reached")
                metrics=nc.reduced_metrics(joint,target[k])
                obs=nc.two_site_observables(joint,basis,joint=True)
                refobs=nc.two_site_observables(target[k].reshape(cutoff,cutoff),basis)
                rows.append(dict(t=k*h,**metrics,**obs,**nc.populations(joint,access=access,active=d),mean_phi=np.array([sg.expect_mode(joint,basis.phi,site).real for site in (0,1)]),C12_error=abs(complex(obs["C12_real"]-refobs["C12_real"],obs["C12_imag"]-refobs["C12_imag"])),S2_error=abs(obs["S2"]-refobs["S2"])))
                states.append(joint.copy())
                if k<r:
                    exact_input=np.zeros_like(joint);exact_input[:,:,0]=target[k].reshape(cutoff,cutoff)
                    expected=np.zeros_like(joint);expected[:,:,0]=target[k+1].reshape(cutoff,cutoff)
                    t2.append(float(np.linalg.norm(step(exact_input,record=False)-np.exp(1j*phase)*expected)))
                    joint=step(joint)
            save_arrays(out,name,**{key:np.array([row[key] for row in rows]) for key in rows[0]},joint_step_states=np.array(states),action_errors=np.array(t2))
            coefficients=(1.,.3) if route=="G_trig" else (-1.2,.3) if route=="H_trig_ret" else (0.,.3)
            pulses=nc.trig_schedule(2,basis.omega,coefficients,h,order,s) if "trig" in route else []
            write_json(out/(name+"_schedule.json"),dict(per_step_unmerged=pulses,per_step_merged=nc.merge_pulses(pulses),outer_steps=r,
                       gaussian_primitives_per_step=10 if route=="G_trig" else 0,native_half_waits_per_step=0 if route=="G_trig" else 2,
                       full_circuit_unmerged=dict(gaussian_primitives=10*r if route=="G_trig" else 0,native_waits=0 if route=="G_trig" else 2*r),
                       full_circuit_merged=dict(gaussian_primitives=5*r+2 if route=="G_trig" else 0,native_waits=0 if route=="G_trig" else r+1,additional_Ry_boundary_cancellations=2*(r-1) if "trig" in route else 0),
                       merging_scope="Potential pulses merge within each block. Bond halves commute through the potential and merge; adjacent kinetic or native half waits merge between outer steps.",
                       local_snap_per_step=2*layers if route=="H_SNAP" else 0,local_displacement_per_step=2*layers if route=="H_SNAP" else 0,
                       system_modes=2,coherently_reused_qubits=1 if "trig" in route else 0,SNAP_control_qubits="device dependent",
                       physical_duration="parametric: pulse durations and native time mapping required"))
            common_rows=rows[::r//10]
            selected.append(dict(name=name,route=route,r=r,layers=layers if route=="H_SNAP" else None,order=order,subdivisions=s,cutoff=cutoff,max_bures_distance=max(row["bures_distance"] for row in common_rows),max_endpoint_bures_distance=max(row["bures_distance"] for row in rows),max_C12_error=max(row["C12_error"] for row in common_rows),max_S2_error=max(row["S2_error"] for row in common_rows),action_sum=sum(t2),max_population_at_monitored_locations=maximum_populations,population_scope="native-half and potential-block endpoints; SNAP route also each local SNAP and displacement"))
    if previous is None:
        checked=solve_ivp(lambda t,x:-1j*(instance["H_target"]@x),(0.,horizon),_sg_initial(instance["basis"],*preparation).ravel(),method="DOP853",t_eval=np.linspace(0,horizon,11),rtol=2e-11,atol=2e-13)
        if not checked.success:
            raise RuntimeError(checked.message)
        solver_margin=max(projective_distance(a,b) for a,b in zip(references[high_cut],checked.y.T))
        ref_obs={cut:_sg_observations(states,sg.build_mode_basis(cut,np.sqrt(3))) for cut,states in references.items()}
        solver_obs=_sg_observations(checked.y.T,instance["basis"])
        c=lambda data:data["C12_real"]+1j*data["C12_imag"]
        cref=float(np.max(abs(c(ref_obs[low_cut])-c(ref_obs[high_cut]))))
        sref=float(np.max(abs(ref_obs[low_cut]["S2"]-ref_obs[high_cut]["S2"])))
        csolver=float(np.max(abs(c(ref_obs[high_cut])-c(solver_obs))))
        ssolver=float(np.max(abs(ref_obs[high_cut]["S2"]-solver_obs["S2"])))
    else:
        high_record=next(row for row in previous_summary["records"] if row["N"]==high_cut and row["lambda2"]==.3)
        solver_margin=high_record["solver"]["state_distance"]
        cref,sref=(previous_summary["empirical_cutoff_estimates"][key] for key in ("C12","S2"))
        csolver,ssolver=(high_record["solver"][key] for key in ("C12","S2"))
        c=lambda data:data["C12_real"]+1j*data["C12_imag"]
    ref_margin=max(projective_distance(a,b) for a,b in zip(embed_trajectory(references[low_cut],low_cut,high_cut),references[high_cut]))
    for point in selected:
        if point["cutoff"]!=low_cut:
            continue
        high_name=point["name"].replace(f"_N{low_cut}",f"_N{high_cut}")
        high_point=next(row for row in selected if row["name"]==high_name)
        with np.load(out/(point["name"]+".npz")) as low,np.load(out/(high_name+".npz")) as high:
            embedded=np.zeros((r+1,high_cut,high_cut,2),complex);embedded[:,:low_cut,:low_cut,:]=low["joint_step_states"]
            pad=max(projective_distance(a,b) for a,b in zip(embedded,high["joint_step_states"]))
            cp=float(np.max(abs(c(low)-c(high))))
            sp=float(np.max(abs(low["S2"]-high["S2"])))
        total=point["max_bures_distance"]+ref_margin+solver_margin+pad
        uc,us=cref+csolver+cp,sref+ssolver+sp
        point["empirical_margins"]=dict(reference=ref_margin,solver=solver_margin,padding_joint_distance=pad,C12=uc,S2=us)
        boundary=max(row["max_population_at_monitored_locations"]["boundary_band_population"] for row in (point,high_point))
        point["numerical_acceptance"]=dict(state=bool(total<=np.sqrt(1e-3) and ref_margin<=np.sqrt(1e-3)/10 and solver_margin<=np.sqrt(1e-3)/10 and boundary<=1e-6),C12=bool(uc<=.001 and point["max_C12_error"]+uc<=.01),S2=bool(us<=.005 and point["max_S2_error"]+us<=.02),amplitude_total=total,boundary_population_both_cutoffs=boundary)
    return dict(training_cutoff=n,active_levels=d,training_columns=access,preparation=list(preparation),validation_cutoffs=[low_cut,high_cut],reference_source=str(previous) if previous else None,configurations=selected,
                acceptance="finite-family numerical assessment with empirical reference/solver/padding estimates; no continuum or hardware guarantee")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job",choices=["e1","e1-formal","e2","e4-reference","e4-refinement","e4-reference128","e4-compiled"])
    parser.add_argument("--previous-reference",type=Path)
    parser.add_argument("--optimizer-source",type=Path)
    parser.add_argument("--outdir",type=Path,required=True)
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=False)
    identity=provenance();started,cpu=time.perf_counter(),time.process_time()
    write_json(args.outdir/"identity.json",dict(job=args.job,development=args.job!="e1-formal",provenance=identity))
    if args.job=="e1-formal":
        result=e1(args.outdir,formal=True)
    elif args.job=="e4-compiled":
        result=e4_compiled(args.outdir,args.previous_reference,args.optimizer_source)
    elif args.job in ("e4-refinement","e4-reference128"):
        if args.previous_reference is None:
            raise ValueError("refinement needs the previous reference records")
        result=e4_reference(args.outdir,cutoffs=(64,) if args.job=="e4-refinement" else (96,128),previous=args.previous_reference,lambdas=(0.,.3) if args.job=="e4-refinement" else (.3,))
    else:
        result={"e1":e1,"e2":e2,"e4-reference":e4_reference,"e4-compiled":e4_compiled}[args.job](args.outdir)
    write_json(args.outdir/"summary.json",dict(**result,wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_memory_bytes=peak_memory_bytes(),provenance=identity))


if __name__=="__main__":
    main()
