"""Finite spectral SG corrections with coherent reuse of one ancilla.

Matrices multiply right to left. Pulse schedules are stored in execution
order. Joint states have oscillator axes followed by the ancillary axis.
"""

import numpy as np
from scipy.linalg import expm
from scipy.special import eval_genlaguerre, gammaln
from scipy.sparse.linalg import expm_multiply, LinearOperator

if __package__:
    from . import sglib as sg
else:
    import sglib as sg

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1., -1.])


def vertex_sandwich(n_cut, omega, harmonic=1):
    """Exact matrix elements of P exp(i m phi) P in the physical Fock basis."""
    alpha = 1j * harmonic / np.sqrt(2 * omega)
    displacement = np.zeros((n_cut, n_cut), dtype=complex)
    for row in range(n_cut):
        for col in range(n_cut):
            low, high = min(row, col), max(row, col)
            coefficient = np.exp(-abs(alpha)**2 / 2 + (gammaln(low+1)-gammaln(high+1))/2)
            power = alpha**(row-col) if row >= col else (-alpha.conjugate())**(col-row)
            displacement[row, col] = coefficient * power * eval_genlaguerre(low, high-low, abs(alpha)**2)
    return displacement


def cosine_sandwich(n_cut, omega, harmonic=1):
    """Exact matrix elements of P cos(m phi) P in the physical Fock basis."""
    displacement=vertex_sandwich(n_cut,omega,harmonic)
    return (displacement + displacement.conj().T) / 2


def sandwich_local_operators(basis):
    """Physical quadratic sandwiches, including the finite boundary term."""
    top = np.zeros_like(basis.phi)
    top[-1, -1] = basis.n_cut
    return basis.phi @ basis.phi + top / (2*basis.omega), basis.pi @ basis.pi + basis.omega*top/2


def apply_joint_local(operator, state, site):
    moved = np.moveaxis(state, (site, state.ndim-1), (0, 1))
    shape = moved.shape
    output = (operator @ moved.reshape(shape[0]*2, -1)).reshape(shape)
    return np.moveaxis(output, (0, 1), (site, state.ndim-1))


def elementary_trig(phi, harmonic, v, sign):
    d = expm(.5j * harmonic * np.kron(phi, X))
    rz = np.kron(np.eye(phi.shape[0]), expm(1j * v * Z))
    return d @ rz @ d.conj().T if sign == 1 else d.conj().T @ rz @ d


def trig_block(phi, harmonic, u, order, subdivisions):
    """Joint Q, including its scalar phase, without ancillary reset."""
    if order not in (1, 2) or not isinstance(subdivisions, (int, np.integer)) or subdivisions < 1 or harmonic not in (1, 2):
        raise ValueError("expected harmonic/order 1 or 2 and positive subdivisions")
    if not np.isfinite(u):
        raise ValueError("harmonic time must be finite")
    if order == 1:
        step = elementary_trig(phi, harmonic, u/(2*subdivisions), 1) @ elementary_trig(phi, harmonic, u/(2*subdivisions), -1)
    else:
        plus = elementary_trig(phi, harmonic, u/(4*subdivisions), 1)
        step = plus @ elementary_trig(phi, harmonic, u/(2*subdivisions), -1) @ plus
    return np.exp(-1j*u) * np.linalg.matrix_power(step, subdivisions)


def trig_schedule(n_site, omega, coefficients, h, order, subdivisions):
    """Unmerged physical pulse schedule for the fixed site/harmonic order."""
    if order not in (1, 2) or subdivisions < 1:
        raise ValueError("invalid product formula")
    pulses = []
    def displacement(site, amplitude):
        pulses.extend([dict(gate="Ry", site=-1, value=-np.pi/2), dict(gate="CD_Z", site=site, value=amplitude), dict(gate="Ry", site=-1, value=np.pi/2)])
    def elementary(site, harmonic, v, sign):
        amplitude = harmonic / (2*np.sqrt(2*omega))
        displacement(site, -sign*amplitude)
        pulses.append(dict(gate="Rz", site=-1, value=-2*v))
        displacement(site, sign*amplitude)
    for site in range(n_site):
        for harmonic, coefficient in enumerate(coefficients, 1):
            if coefficient == 0:
                continue
            u = h*coefficient
            for _ in range(subdivisions):
                if order == 1:
                    elementary(site, harmonic, u/(2*subdivisions), -1)
                    elementary(site, harmonic, u/(2*subdivisions), 1)
                else:
                    elementary(site, harmonic, u/(4*subdivisions), 1)
                    elementary(site, harmonic, u/(2*subdivisions), -1)
                    elementary(site, harmonic, u/(4*subdivisions), 1)
    return pulses


def merge_pulses(pulses, cd_cap=1.5, rotation_cap=2*np.pi):
    """Merge adjacent identical generators only within the amplitude cap."""
    merged = []
    for pulse in pulses:
        pulse = dict(pulse)
        if merged and (pulse["gate"], pulse["site"]) == (merged[-1]["gate"], merged[-1]["site"]):
            value = pulse["value"] + merged[-1]["value"]
            cap = cd_cap if pulse["gate"] == "CD_Z" else rotation_cap
            if abs(value) <= cap:
                merged.pop()
                if value != 0:
                    pulse["value"] = value
                    merged.append(pulse)
                continue
        merged.append(pulse)
    return merged


def trig_potential(state, basis, coefficients, h, order, subdivisions):
    output = state
    blocks = {m: trig_block(basis.phi, m, h*c, order, subdivisions) for m, c in enumerate(coefficients, 1) if c != 0}
    for site in range(state.ndim-1):
        for harmonic in (1, 2):
            if harmonic in blocks:
                output = apply_joint_local(blocks[harmonic], output, site)
    return output


def reduced_metrics(joint, reference):
    """Pure-target fidelity from joint amplitudes, with no postselection."""
    flat = joint.reshape(-1, 2)
    ref = reference.reshape(-1)
    if not np.isclose(np.linalg.norm(flat), 1, atol=2e-9, rtol=0) or not np.isclose(np.linalg.norm(ref), 1, atol=2e-9, rtol=0):
        raise ValueError("joint and reference states must be normalized")
    fidelity = float(np.sum(abs(ref.conj() @ flat)**2))
    if fidelity < -2e-9 or fidelity > 1+2e-9:
        raise FloatingPointError("fidelity outside the roundoff envelope")
    fidelity = float(np.clip(fidelity, 0., 1.))
    ancilla = flat.conj().T @ flat
    return dict(infidelity=1-fidelity, bures_distance=float(np.sqrt(2-2*np.sqrt(fidelity))), ancilla_excited_population=float(np.sum(abs(flat[:, 1])**2)), system_purity=float(np.trace(ancilla @ ancilla).real))


def two_site_observables(state, basis, joint=False, vertex=None):
    """C12 and single-site purity. Joint S2 is not an entanglement measure."""
    expected = (basis.n_cut, basis.n_cut, 2) if joint else (basis.n_cut, basis.n_cut)
    if state.shape != expected:
        raise ValueError("expected a two-site state with an optional ancillary axis")
    if vertex is None:
        vertex = (basis.U_phi * np.exp(1j*basis.lam_phi)) @ basis.U_phi.conj().T
    v1 = sg._apply_axis(vertex, state, 0)
    v2 = sg._apply_axis(vertex.conj().T, state, 1)
    pair = sg._apply_axis(vertex, v2, 0)
    connected = np.vdot(state, pair) - np.vdot(state, v1)*np.vdot(state, v2)
    matrix = state.reshape(basis.n_cut, -1)
    rho1 = matrix @ matrix.conj().T
    purity = float(np.vdot(rho1, rho1).real)
    return dict(C12_real=float(connected.real), C12_imag=float(connected.imag), S2=float(-np.log(purity)), purity1=purity)


def prepare_moment_diagnostics(basis,E_J,lambda2,offsets):
    prepared=[]
    for offset in offsets:
        phi=basis.phi+offset*np.eye(basis.n_cut)
        sine1=(basis.U_phi*np.sin(basis.lam_phi+offset))@basis.U_phi.conj().T
        sine2=(basis.U_phi*np.sin(2*(basis.lam_phi+offset)))@basis.U_phi.conj().T
        terms=[]
        for m,c,sine in ((1,E_J,sine1),(2,lambda2,sine2)):
            potential=(basis.U_phi*(c*(1-np.cos(m*(basis.lam_phi+offset)))))@basis.U_phi.conj().T
            terms.append(1j*(potential@basis.pi-basis.pi@potential)+m*c*sine)
        prepared.append((phi,sine1,sine2,*terms))
    return prepared


def moment_diagnostics(state, basis, E_J, kappa, lambda2, hamiltonian, offsets=None, pbc=False, prepared=None):
    """Separate closure residuals from actual finite-commutator remainders."""
    n_site = state.ndim
    offsets = np.zeros(n_site) if offsets is None else np.asarray(offsets)
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm==0:
        raise ValueError("moment diagnostics require a nonzero finite state")
    state = state / norm
    prepared=prepare_moment_diagnostics(basis,E_J,lambda2,offsets) if prepared is None else prepared
    hpsi = hamiltonian(state)
    mean, momentum, variance, third, sine1, sine2, dotphi, dotpi, cut1, cut2 = ([] for _ in range(10))
    for site in range(n_site):
        phi,sin1,sin2,comm1,comm2=prepared[site]
        qpsi = sg._apply_axis(phi, state, site)
        ppsi = sg._apply_axis(basis.pi, state, site)
        q = float(np.vdot(state, qpsi).real)
        centered = qpsi-q*state
        mean.append(q)
        momentum.append(float(np.vdot(state, ppsi).real))
        variance.append(float(np.vdot(centered, centered).real))
        third.append(float(np.vdot(centered, sg._apply_axis(phi-q*np.eye(basis.n_cut), centered, site)).real))
        for sine,result in ((sin1,sine1),(sin2,sine2)):
            result.append(float(sg.expect_mode(state, sine, site).real))
        cut1.append(float(sg.expect_mode(state,comm1,site).real))
        cut2.append(float(sg.expect_mode(state,comm2,site).real))
        dotphi.append(float(2*np.vdot(qpsi, hpsi).imag))
        dotpi.append(float(2*np.vdot(ppsi, hpsi).imag))
    mean, momentum, variance, third, sine1, sine2 = map(np.asarray, (mean, momentum, variance, third, sine1, sine2))
    quantum_force = -E_J*sine1-2*lambda2*sine2
    pairs = [(j, j+1) for j in range(n_site-1)] + ([(n_site-1, 0)] if pbc and n_site > 2 else [])
    for left, right in pairs:
        delta = kappa*(mean[right]-mean[left])
        quantum_force[left] += delta
        quantum_force[right] -= delta
    closure = -E_J*(sine1-np.sin(mean))-2*lambda2*(sine2-np.sin(2*mean))
    return dict(mean_phi=mean,mean_pi=momentum,variance_phi=variance, central_third_phi=third, sin_phi=sine1, sin_2phi=sine2, delta_NG_1=sine1-np.exp(-variance/2)*np.sin(mean), delta_NG_2=sine2-np.exp(-2*variance)*np.sin(2*mean), closure_momentum=closure, cutoff_phi=np.asarray(dotphi)-momentum, cutoff_momentum=np.asarray(dotpi)-quantum_force,cutoff_harmonic1=np.asarray(cut1),cutoff_harmonic2=np.asarray(cut2))


def populations(state,access=16,active=24,edge=4):
    n=state.shape[0]
    probabilities=abs(state)**2
    return dict(outside_joint_access=float(probabilities.sum()-probabilities[:access,:access].sum()),
                outside_local_active_max=float(max(probabilities[active:].sum(),probabilities[:,active:].sum())),
                boundary_band_population=float(probabilities[-edge:].sum()+probabilities[:,-edge:].sum()-probabilities[-edge:,-edge:].sum()))


def make_step(problem, E_J, kappa, lambda2, h, route, order=2, subdivisions=1, local_correction=None,observer=None,local_factors=None):
    """Prepare one two-site step, caching all local pulses outside evolution."""
    if __package__:
        from double_sine_gordon_code.run_snapd_end_to_end import build_two_site_problem
    else:
        from run_snapd_end_to_end import build_two_site_problem
    basis = problem["basis"]
    if problem.get("operator_convention", "spectral") != "spectral":
        raise ValueError("the compiled SG routes are defined for the finite spectral convention")
    n = basis.n_cut
    allowed = {"native", "retuned_native", "exact_split", "H_SNAP", "H_trig", "G_trig", "H_trig_ret"}
    if route not in allowed:
        raise ValueError("unknown SG route")
    coefficients = (E_J, lambda2) if route == "G_trig" else (-4*lambda2, lambda2) if route == "H_trig_ret" else (0., lambda2)
    native_problem = problem
    if route in ("retuned_native", "H_trig_ret"):
        native_problem = build_two_site_problem(n, basis.omega, E_J+4*lambda2, kappa, 0.)
    local=native_problem["native_local_with_bond"]
    coupling=native_problem["bond_strength"]
    def native_hamiltonian_action(columns):
        shape=columns.shape
        state=columns.reshape(n,n,-1)
        result=sg._apply_axis(local,state,0)+sg._apply_axis(local,state,1)
        result-=coupling*sg._apply_axis(basis.phi,sg._apply_axis(basis.phi,state,0),1)
        return result.reshape(shape)
    native = LinearOperator((n*n,n*n),matvec=native_hamiltonian_action,rmatvec=native_hamiltonian_action,matmat=native_hamiltonian_action,rmatmat=native_hamiltonian_action,dtype=np.complex128)
    native_coefficient = h if route in ("native", "retuned_native") else h/2
    native_generator = -1j*native_coefficient*native
    native_trace = complex(-1j*native_coefficient*(2*n*np.trace(local)-coupling*np.trace(basis.phi)**2))
    kinetic_half = expm(-.5j*h*basis.kinetic) if route == "G_trig" else None
    bond_phase = np.exp(-.25j*h*kappa*(basis.lam_phi[:, None]-basis.lam_phi[None, :])**2) if route == "G_trig" else None
    blocks = {m: trig_block(basis.phi, m, h*c, order, subdivisions) for m,c in enumerate(coefficients,1) if c != 0} if "trig" in route else {}
    if route == "exact_split":
        local_correction = expm(-1j*h*problem["correction_local"])
    if route == "H_SNAP" and (local_correction is None or local_correction.shape != (n,n)):
        raise ValueError("H_SNAP requires the local correction in the same padded space")
    def native_action(state):
        return expm_multiply(native_generator, state.reshape(n*n,2), traceA=native_trace).reshape(n,n,2)
    def kinetic(state):
        return sg._apply_axis(kinetic_half, sg._apply_axis(kinetic_half, state, 0), 1)
    def bond(state):
        phase = sg._apply_axis(basis.U_phi.conj().T, sg._apply_axis(basis.U_phi.conj().T,state,0),1)
        phase = bond_phase[:, :, None]*phase
        return sg._apply_axis(basis.U_phi,sg._apply_axis(basis.U_phi,phase,0),1)
    def step(state,record=True):
        if state.shape != (n,n,2):
            raise ValueError("joint state must have two oscillator axes and one qubit axis")
        if route in ("native", "retuned_native"):
            return native_action(state)
        output = bond(kinetic(state)) if route == "G_trig" else native_action(state)
        if observer is not None and record:
            observer(output)
        if "trig" in route:
            for site in range(2):
                for harmonic in (1,2):
                    if harmonic in blocks:
                        output = apply_joint_local(blocks[harmonic], output, site)
                        if observer is not None and record:
                            observer(output)
        else:
            for site in range(2):
                for factor in ([local_correction] if local_factors is None else local_factors):
                    output = sg._apply_axis(factor, output, site)
                    if observer is not None and record:
                        observer(output)
        output=kinetic(bond(output)) if route == "G_trig" else native_action(output)
        if observer is not None and record:
            observer(output)
        return output
    return step
