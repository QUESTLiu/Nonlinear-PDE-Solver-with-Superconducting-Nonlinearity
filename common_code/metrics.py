"""Phase-insensitive distances and finite-input error diagnostics."""

import numpy as np
from scipy.integrate import cumulative_trapezoid


def projective_distance(reference, candidate):
    """Distance between normalized vectors, without discarding output levels."""
    a, b = np.asarray(reference).ravel(), np.asarray(candidate).ravel()
    if a.shape != b.shape or not np.isclose(np.linalg.norm(a), 1, atol=2e-10, rtol=0) or not np.isclose(np.linalg.norm(b), 1, atol=2e-10, rtol=0):
        raise ValueError("states must be normalized in the same space")
    overlap = complex(np.dot(a.real,b.real)+np.dot(a.imag,b.imag),np.dot(a.real,b.imag)-np.dot(a.imag,b.real))
    phase = 1 if abs(overlap) == 0 else np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(a - phase * b))


def projective_distances(reference,candidate):
    """Columnwise distances with an exactly real self-overlap in arithmetic."""
    a,b=np.asarray(reference),np.asarray(candidate)
    if a.ndim!=2 or a.shape!=b.shape:
        raise ValueError("state families must have the same column shape")
    if not np.allclose(np.linalg.norm(a,axis=0),1,atol=2e-10,rtol=0) or not np.allclose(np.linalg.norm(b,axis=0),1,atol=2e-10,rtol=0):
        raise ValueError("state columns must be normalized")
    real=np.sum(a.real*b.real+a.imag*b.imag,axis=0)
    imag=np.sum(a.real*b.imag-a.imag*b.real,axis=0)
    phases=np.exp(-1j*np.angle(real+1j*imag))
    return np.linalg.norm(a-phases[None,:]*b,axis=0)


def centered_operator(operator):
    h = np.asarray(operator, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.allclose(h, h.conj().T, rtol=0, atol=1e-12):
        raise ValueError("centering requires a finite Hermitian matrix")
    e = np.linalg.eigvalsh(h)
    return h - (e[0] + e[-1]) / 2 * np.eye(h.shape[0])


def native_residual_integral(residual, native_states, times):
    """Trapezoidal estimates of T1 and the uncentered Duhamel integral.

    Columns are native reference states. Quadrature and state-solver margins
    must be supplied separately before these estimates support a guarantee.
    """
    states, times = np.asarray(native_states), np.asarray(times)
    residual=np.asarray(residual)
    if times.ndim != 1 or len(times) < 2 or states.ndim != 2 or states.shape[1] != len(times) or np.any(np.diff(times) <= 0) or not np.all(np.isfinite(times)):
        raise ValueError("states and increasing observation times must agree")
    if residual.shape != (states.shape[0],states.shape[0]) or not np.allclose(residual,residual.conj().T,atol=2e-12,rtol=0):
        raise ValueError("residual must be a Hermitian matrix in the state space")
    if not np.allclose(np.sum(abs(states)**2,axis=0),1,atol=2e-10,rtol=0):
        raise ValueError("native reference states must be normalized")
    action = residual @ states
    shift=residual[0,0].real
    shifted_action=(residual-shift*np.eye(residual.shape[0]))@states
    means = np.sum(states.conj()*shifted_action,axis=0).real
    centered = np.linalg.norm(shifted_action-states*means,axis=0)
    raw = np.linalg.norm(action, axis=0)
    return dict(variance=centered**2, centered_integral=cumulative_trapezoid(centered, times, initial=0), uncentered_integral=cumulative_trapezoid(raw, times, initial=0))


def native_verdict(bound_amplitude, threshold, *, amplitude_margin=None, observed_infidelity=None, infidelity_uncertainty=None):
    """Three-way numerical assessment with explicit evidence margins.

    A missing margin is unknown, not zero. The caller identifies whether a
    supplied margin is rigorous or an empirical convergence estimate.
    """
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("infidelity threshold must lie in (0,1)")
    for value in (bound_amplitude, amplitude_margin, infidelity_uncertainty):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("bounds and uncertainties must be finite and nonnegative")
    if observed_infidelity is not None and (not np.isfinite(observed_infidelity) or not 0 <= observed_infidelity <= 1):
        raise ValueError("infidelity must lie in [0,1]")
    upper = None if amplitude_margin is None else min(1., (bound_amplitude + amplitude_margin)**2)
    lower = None if observed_infidelity is None or infidelity_uncertainty is None else max(0., observed_infidelity - infidelity_uncertainty)
    if upper is not None and lower is not None and lower > upper:
        raise ValueError("inconsistent error evidence")
    verdict = "sufficient" if upper is not None and upper <= threshold else "observed_failure" if lower is not None and lower > threshold else "inconclusive"
    return dict(verdict=verdict, infidelity_upper=upper, observed_lower_estimate=lower, threshold=threshold)


def family_action_sum(step, target_step, initial, repetitions, phase=0.):
    """Offline T2 estimate on exact target intermediate states."""
    state = np.asarray(initial).copy()
    errors = []
    for _ in range(repetitions):
        next_target = target_step(state)
        errors.append(float(np.linalg.norm(step(state) - np.exp(1j * phase) * next_target)))
        state = next_target
    return np.asarray(errors)


def first_crossing(times, values, threshold):
    """Return a sampled crossing bracket and linear interpolant, or censoring."""
    t, v = np.asarray(times), np.asarray(values)
    if t.ndim != 1 or t.size < 2 or v.shape != t.shape or np.any(np.diff(t) <= 0) or not np.all(np.isfinite(v)):
        raise ValueError("crossing requires finite values on an increasing grid")
    indices = np.flatnonzero(v > threshold)
    if len(indices) == 0:
        return dict(crossed=False, time=None, bracket=None, right_censored_at=float(t[-1]))
    j = int(indices[0])
    if j == 0:
        return dict(crossed=True, time=float(t[0]), bracket=[float(t[0]), float(t[0])], right_censored_at=None)
    f = (threshold - v[j-1]) / (v[j] - v[j-1])
    return dict(crossed=True, time=float(t[j-1] + f * (t[j]-t[j-1])), bracket=[float(t[j-1]), float(t[j])], right_censored_at=None)
