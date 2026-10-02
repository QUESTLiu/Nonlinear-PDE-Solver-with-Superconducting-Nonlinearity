"""Joint Fourier phase compilation with one coherently reused ancillary qubit.

For G(x)=exp[i(a cos x+b cos 2x)], Jacobi--Anger convolution gives
C_n=sum_l i**(n-l) J_(n-2l)(a) J_l(b). Truncate to [-K,K], and divide
by 1+sum_{|n|>K}|C_n|. Spectral factorization completes F to the SU(2)
Laurent matrix [[F,-H*],[H,F*]], with |H|**2=1-|F|**2 on the circle.
The numerical Bessel sum and root factorization are checked independently,
not treated as a floating-point certificate.

Let C(z)=z**K U(z), of degree d=2K. Its leading column space has rank one.
If P projects onto that space, then ((I-P)+P/z) C(z) has degree d-1:
its coefficients are (I-P) C_k+P C_(k+1). Iterating expresses U as d
balanced queries exp(i x Z/2), conjugated by constant SU(2) matrices.
We recover Z-Y-Z angles, commute Z rotations through Z-controlled signals,
and measure the error of the pulses that are actually counted.

This implements the grouped completion and inverse-factorization construction
of arXiv:2609.40078v1, Sec. III and Appendix C.5, with inside-circle roots.
It is a bounded comparison of this construction, not a QSP optimality test.
No postselection or reset is used. Scalar potential offsets are global phases.
"""

import numpy as np
from scipy.special import jv


def response_coefficients(a, b, degree):
    """Return a contractive truncation and empirical numerical diagnostics.

    Forty Bessel orders suffice here (|a|,|b|<1). The omitted Bessel tails
    are below 1e-60; a 1e-13 slack covers coefficient roundoff, and the
    resulting completion and pulse errors are explicitly measured.
    """
    if max(abs(a), abs(b)) >= 1 or degree < 1:
        raise ValueError("this bounded compiler requires |a|,|b|<1 and K>=1")
    indices = np.arange(-40, 41)
    first = (1j ** indices) * jv(indices, a)
    second = np.zeros(161, complex)
    second[::2] = (1j ** indices) * jv(indices, b)
    coeff = np.convolve(first, second)  # indices -120,...,120
    tail = float(np.sum(np.abs(coeff[np.abs(np.arange(-120, 121)) > degree])))
    f = coeff[120 - degree:121 + degree] / (1 + tail + 1e-13)
    x = 2 * np.pi * np.arange(4096) / 4096
    fft = np.fft.fft(np.exp(1j * (a * np.cos(x) + b * np.cos(2 * x)))) / len(x)
    fft_error = max(abs(coeff[120 + n] - fft[n % len(x)]) for n in range(-20, 21))
    return f, dict(tail=tail, coefficient_fft_error=float(fft_error))


def complete_and_factor(f):
    """Return time-ordered SU(2) rotations around balanced Z queries."""
    d = len(f) - 1
    q = -np.convolve(f, f[::-1].conj())
    q[d] += 1
    roots = np.polynomial.polynomial.polyroots(q)
    inside = roots[np.argsort(np.abs(roots))[:d]]
    h = np.polynomial.polynomial.polyfromroots(inside)
    h *= np.sqrt(q[d].real / np.sum(np.abs(h) ** 2))
    completion_error = np.max(np.abs(np.convolve(h, h[::-1].conj()) - q))
    coeff = np.empty((d + 1, 2, 2), complex)
    coeff[:, 0, 0], coeff[:, 1, 0] = f, h
    coeff[:, 0, 1], coeff[:, 1, 1] = -h[::-1].conj(), f[::-1].conj()
    original = coeff.copy()
    bases, residuals = [], []
    eye = np.eye(2)
    for _ in range(d):
        left, _, _ = np.linalg.svd(coeff[-1])
        v = left[:, 0]
        basis = np.column_stack((v, [-v[1].conjugate(), v[0].conjugate()]))
        p = np.outer(v, v.conj())
        residuals.extend([np.linalg.norm(p @ coeff[0]), np.linalg.norm((eye - p) @ coeff[-1])])
        coeff = (eye - p) @ coeff[:-1] + p @ coeff[1:]
        bases.append(basis)
    left, _, right = np.linalg.svd(coeff[0])
    constant = left @ right
    constant /= np.sqrt(np.linalg.det(constant))
    rotations = [bases[-1].conj().T @ constant]
    rotations += [bases[i].conj().T @ bases[i + 1] for i in range(d - 2, -1, -1)]
    rotations.append(bases[0])
    diagnostics = dict(completion_error=float(completion_error),
                       peeling_residual=float(max(residuals)),
                       constant_projection_error=float(np.linalg.norm(constant - coeff[0])))
    return rotations, original, diagnostics


def euler_pulses(matrix):
    """Time order gamma-Z, beta-Y, alpha-Z for Rz(alpha) Ry(beta) Rz(gamma)."""
    a, minus_b = matrix[0, 0], -matrix[0, 1]
    beta = 2 * np.arctan2(abs(minus_b), abs(a))
    alpha = -np.angle(a) - np.angle(minus_b)
    gamma = -np.angle(a) + np.angle(minus_b)
    return [("Rz", float(gamma)), ("Ry", float(beta)), ("Rz", float(alpha))]


def merge_z(pulses):
    """Commute Rz through CD_Z, then merge, recording any tiny-angle removal.

    Removing a rotation of wrapped angle theta costs at most |theta|/2 in
    operator norm. The summed allowance is recorded, and the final word is
    checked against the completed response. No signal pulse is removed.
    """
    result, pending, omitted = [], 0.0, 0.0

    def flush():
        nonlocal pending, omitted
        # Four-pi wrapping preserves the SU(2) matrix, including its sign.
        angle = float((pending + 2 * np.pi) % (4 * np.pi) - 2 * np.pi)
        if abs(angle) < 1e-12:
            omitted += abs(angle) / 2
        else:
            result.append(("Rz", angle))
        pending = 0.0

    for gate, value in pulses:
        if gate == "Rz":
            pending += value
        elif gate == "CD_Z":
            result.append((gate, value))
        else:
            flush()
            if result and result[-1][0] == "Ry":
                value += result.pop()[1]
            if abs(value) < 1e-12:
                omitted += abs(value) / 2
            else:
                result.append((gate, value))
    flush()
    return result, omitted


def merge_rotations(pulses):
    """Fuse adjacent qubit unitaries before commuting their Euler Z rotations.

    System-only evolution commutes with every qubit rotation, so word boundary
    rotations can be fused even across a native or Gaussian interval. A CD
    separates the groups. Every recovered group is checked by pulse_response
    in the independent whole-circuit check of check_joint_phase.py.
    """
    result = []
    pending = np.eye(2, dtype=complex)

    def flush():
        nonlocal pending
        result.extend(euler_pulses(pending))
        pending = np.eye(2, dtype=complex)

    for gate, angle in pulses:
        if gate == "CD_Z":
            flush()
            result.append((gate, angle))
        elif gate == "Ry":
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            pending = np.array([[c, -s], [s, c]]) @ pending
        else:
            pending = np.diag([np.exp(-.5j * angle), np.exp(.5j * angle)]) @ pending
    flush()
    return merge_z(result)


def pulse_response(pulses, x):
    """Actual pulse product on a scalar signal grid, qubit basis |0>,|1>."""
    x = np.asarray(x)
    result = np.broadcast_to(np.eye(2, dtype=complex), (x.size, 2, 2)).copy()
    for gate, value in pulses:
        if gate == "Ry":
            c, s = np.cos(value / 2), np.sin(value / 2)
            result = np.array([[c, -s], [s, c]]) @ result
        else:
            angle = x * value if gate == "CD_Z" else np.full(x.size, -value / 2)
            result[:, 0, :] *= np.exp(1j * angle)[:, None]
            result[:, 1, :] *= np.exp(-1j * angle)[:, None]
    return result


def compile_phase(a, b, degree):
    """Compile the entire phase and check it on an independent shifted grid."""
    f, diagnostics = response_coefficients(a, b, degree)
    rotations, coeff, factor_info = complete_and_factor(f)
    pulses = []
    for i, rotation in enumerate(rotations):
        if i:
            pulses.append(("CD_Z", 0.5))
        pulses.extend(euler_pulses(rotation))
    pulses, omitted = merge_z(pulses)
    x = 2 * np.pi * (np.arange(8192) + 0.37) / 8192
    word = pulse_response(pulses, x)
    powers = np.exp(1j * x[:, None] * np.arange(-degree, degree + 1))
    completed = np.einsum("xn,nab->xab", powers, coeff)
    target = np.zeros_like(word)
    target[:, 0, 0] = np.exp(1j * (a * np.cos(x) + b * np.cos(2 * x)))
    target[:, 1, 1] = target[:, 0, 0].conj()
    reconstruction = float(np.max(np.linalg.norm(word - completed, ord=2, axis=(1, 2))))
    paired_error = float(np.max(np.linalg.norm(word - target, ord=2, axis=(1, 2))))
    unitarity = float(np.max(np.abs(word.conj().transpose(0, 2, 1) @ word - np.eye(2))))
    if reconstruction > 1e-8 or unitarity > 1e-11:
        raise ArithmeticError(f"failed pulse recovery: {reconstruction=}, {unitarity=}")
    diagnostics.update(factor_info, reconstruction_error=reconstruction,
                       sampled_paired_error=paired_error, unitarity_error=unitarity,
                       omitted_rotation_bound=omitted)
    # Linear interpolation bounds the unsampled matrix-valued error by M2 dx^2/8.
    # Differentiating the 2K unitary queries gives ||W''||<=K^2; the target
    # contributes (|a|+2|b|)^2+|a|+4|b|. This bound excludes floating-point error.
    m2 = degree ** 2 + (abs(a) + 2 * abs(b)) ** 2 + abs(a) + 4 * abs(b)
    diagnostics["interpolation_allowance"] = float(m2 * (2 * np.pi / len(x)) ** 2 / 8)
    return dict(a=a, b=b, degree=degree, pulses=pulses, diagnostics=diagnostics)
