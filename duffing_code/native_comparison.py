"""Constructive Kerr replacement and finite-space Duffing diagnostics."""

import numpy as np

from . import duffing_padded_hybrid_validation as df


def kerr_snap(energies, duration, active_dim):
    """SNAP phase vector and whole-space scalar phase for a Kerr wait."""
    energies = np.asarray(energies)
    if energies.ndim != 1 or not 1 <= active_dim <= energies.size:
        raise ValueError("SNAP cutoff must lie in the padded space")
    e0 = energies[0]
    phases = (-duration*(energies[:active_dim]-e0)+np.pi) % (2*np.pi)-np.pi
    diagonal = np.ones(energies.size, dtype=complex)
    diagonal[:active_dim] = np.exp(1j*phases)
    return phases, np.exp(-1j*duration*e0)*np.diag(diagonal)


def constructive_pair(ops, gates, residual, initial, dt, repetitions, active_dim, residual_factors=None):
    """H and S_K endpoints, with tails at the r+1 merged H wait locations."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    energies = np.diag(ops["H_K"]).real
    middle = gates.v2_half @ residual @ gates.v2_half
    half_phases, replacement_half = kerr_snap(energies, dt/2, active_dim)
    full_phases, _ = kerr_snap(energies, dt, active_dim)
    hstates, skstates = [initial.copy()], [initial.copy()]
    factors = [residual] if residual_factors is None else residual_factors
    maxima={name:dict(boundary_population=float(np.sum(abs(initial[-4:])**2)),above_active_population=float(np.sum(abs(initial[active_dim:])**2))) for name in ("hybrid","replacement")}
    for name,half,history in (("hybrid",gates.k_half,hstates),("replacement",replacement_half,skstates)):
        for _ in range(repetitions):
            state=history[-1]
            for gate in [half,gates.v2_half,*factors,gates.v2_half,half]:
                state=gate@state
                maxima[name]["boundary_population"]=max(maxima[name]["boundary_population"],float(np.sum(abs(state[-4:])**2)))
                maxima[name]["above_active_population"]=max(maxima[name]["above_active_population"],float(np.sum(abs(state[active_dim:])**2)))
            history.append(state)
    state = initial.copy()
    tails, locations = [], []
    for j in range(repetitions+1):
        duration = dt/2 if j in (0, repetitions) else dt
        tails.append(float(np.linalg.norm(state[active_dim:])))
        phases = half_phases if j in (0, repetitions) else full_phases
        locations.append(dict(index=j, simulated_wait=duration, tail_amplitude=tails[-1], phase_vector=phases.tolist()))
        state = np.exp(-1j*duration*energies)*state
        if j < repetitions:
            state = middle @ state
    return dict(hybrid=np.column_stack(hstates), replacement=np.column_stack(skstates), locations=locations, replacement_bound=2*sum(tails), extra_snap_count=repetitions+1,gate_populations=maxima)


def residual_factors(dim,active_dim,layers,vector):
    phases,xy=df.unpack_parameters(vector,layers,active_dim)
    a=df.ladder(dim)
    factors=[]
    for phase,components in zip(phases,xy):
        diagonal=np.ones(dim,complex);diagonal[:active_dim]=np.exp(1j*phase)
        factors.append(np.diag(diagonal))
        factors.append(df.finite_displacement(complex(*components),a)[0])
    return factors


def three_sector_constant(ops, h0):
    lam = ops["lambda"]
    sectors = [ops["H_K"]-ops["H_K"][0, 0]*np.eye(len(ops["H_K"])), lam*ops["V_2"], lam*ops["V_4"]]
    m = sum(float(np.linalg.norm(h, 2)) for h in sectors)
    log_constant = 3*np.log(m)+h0*m-np.log(3) if m else -np.inf
    return dict(norm_sum=m, h0=h0, log_constant=float(log_constant), constant=float(np.exp(log_constant)) if log_constant < np.log(np.finfo(float).max) else None)


def moment_operators(n, omega0):
    """Prepare physical moment sandwiches once per reference cutoff."""
    a = df.ladder(n+4)
    qext = (a+a.conj().T)/np.sqrt(2*omega0)
    q = qext[:n, :n]
    p = (1j*np.sqrt(omega0/2)*(a.conj().T-a))[:n, :n]
    q2, q3 = (qext@qext)[:n, :n], (qext@qext@qext)[:n, :n]
    return q,p,q2,q3


def moments(state, omega0, beta, operators, prepared=None):
    """Physical-sandwich force expectation and actual commutator residual."""
    q,p,q2,q3=moment_operators(state.size,omega0) if prepared is None else prepared
    mean = float(np.vdot(state, q@state).real)
    mom = float(np.vdot(state, p@state).real)
    second, third = [float(np.vdot(state, op@state).real) for op in (q2, q3)]
    variance = second-mean**2
    central_third = third-3*mean*second+2*mean**3
    hpsi = operators["H_D"]@state
    dotq = float(2*np.vdot(q@state, hpsi).imag)
    dotp = float(2*np.vdot(p@state, hpsi).imag)
    return dict(q=mean, p=mom, variance_q=variance, central_third_q=central_third, closure_p=-beta*(3*mean*variance+central_third), cutoff_q=dotq-mom, cutoff_p=dotp+omega0**2*mean+beta*third)
