"""Finite-matrix readouts of the canonical Gaussian sequence counted in Appendix J.

Let G(t)=Phi(Q(t)) Pi(P(t)) Phi(Q(t)), with Phi(Q)=exp(-i Q).
The r-step circuit has boundary G(h/2) intervals and r-1 internal G(h)
intervals. Its position phases commute with every local trigonometric block,
so adjacent phases merge by ADDING THEIR GENERATORS, not their time arguments.
This gives r+2 position and r+1 momentum factors. The canonical G group law
does not hold automatically after replacing the quadratures by finite matrices.

Readouts at k*h are complete k-step circuits. Shared intermediate propagation
is closed with the boundary half interval at each readout. Boundary checks
include the potential checkpoints of those complete circuits, including the
different final position phase of a circuit that ends at an earlier readout.
The phase commutation is exact also for the finite spectral representation.
"""


def merged_gaussian_readouts(
    initial,
    steps,
    half,
    full,
    position,
    momentum,
    potential,
    observe,
    *,
    final_only=False,
):
    """Yield complete circuit outputs at 0,h,...,steps*h without storing trajectories.

    half/full are pairs of real diagonal quadratic generators on the position
    and momentum grids. position/momentum apply exp(-i*generator). potential
    applies the trigonometric factors and calls its supplied observer after
    each block. It must commute with all position phases in this construction.
    final_only yields only the initial and final states and monitors only that
    complete circuit, as required for each independent common-time readout.
    """
    if steps < 1:
        raise ValueError("a Gaussian circuit needs at least one step")
    q_half, p_half = half
    q_full, p_full = full
    yield initial
    state = momentum(position(initial, q_half), p_half)
    for k in range(1, steps + 1):
        left = q_half if k == 1 else q_full
        right = q_half if k == steps else q_full
        # All position phases and trigonometric potentials are functions of the
        # same commuting phi_j. Their generators add exactly even at finite N.
        # Q(tau) is nonlinear in tau, so replacing Q(a)+Q(b) by Q(a+b) is invalid.
        state = position(state, left + right)
        closing_shift = q_half - right
        # A prefix ending here needs Q(h/2) on its right, not the interior Q(h).
        # Apply their difference only to the closed readout/checkpoint; retain
        # the interior state so the next step follows the full counted circuit.

        def check(value):
            observe(value)
            if k < steps and not final_only:
                # Check the same potential checkpoint in the circuit ending here.
                observe(position(value, closing_shift))

        check(state)
        state = potential(state, check)
        if k == steps or not final_only:
            closed = state if k == steps else position(state, closing_shift)
            closed = position(momentum(closed, p_half), q_half)
            observe(closed)
            yield closed
        if k < steps:
            state = momentum(state, p_full)
