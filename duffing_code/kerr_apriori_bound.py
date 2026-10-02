"""A priori native-Kerr bound from initial number moments (Duffing, strict sandwich).

Under number-diagonal (Kerr) evolution the norm of each monomial of V_cr applied
to the state is conserved, so
    I_K(t) <= [lambda t A(psi_0)]^2,
    A = |a^4 psi| + |a^dag4 psi| + 4|a^dag3 a psi| + 4|a^dag a^3 psi| + 6|a^2 psi| + 6|a^dag2 psi|,
with all monomials the strict sandwich of the infinite-space operators.
Writes ../data/kerr_apriori_bound.json. Runs in well under a second.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from duffing_code import duffing_padded_hybrid_validation as df  # noqa: E402


def monomials(n, pad=4):
    ext = n + pad
    a = df.ladder(ext)
    ad = a.conj().T
    mats = dict(a4=a @ a @ a @ a, ad4=ad @ ad @ ad @ ad, ad3a=ad @ ad @ ad @ a, ada3=ad @ a @ a @ a, a2=a @ a, ad2=ad @ ad)
    return {k: v[:n, :n] for k, v in mats.items()}


def main():
    n, T = 80, 1.0
    mons = monomials(n)
    weights = dict(a4=1, ad4=1, ad3a=4, ada3=4, a2=6, ad2=6)
    grid = json.loads((REPO / "duffing_data" / "formal" / "E1_retry" / "summary.json").read_text())["points"]
    rows = []
    for p in grid:
        if p["T"] != 1.0 or p["beta"] == 0:
            continue
        lam = p["beta"] / 16.0
        psi, _ = df.coherent_state(n, p["xi"])
        A = sum(w * np.linalg.norm(mons[k] @ psi) for k, w in weights.items())
        bound = min(1.0, (lam * T * A) ** 2)
        rows.append(dict(beta=p["beta"], xi=p["xi"], apriori_infidelity_bound=bound,
                         centered_bound=p["verdicts"][0]["infidelity_upper"], actual=p["max_native_infidelity"]))
    out = dict(N=n, T=T, rows=rows,
               certified_1e3=[(r["beta"], r["xi"]) for r in rows if r["apriori_infidelity_bound"] <= 1e-3],
               certified_1e2=[(r["beta"], r["xi"]) for r in rows if r["apriori_infidelity_bound"] <= 1e-2])
    (Path(__file__).resolve().parents[1] / "duffing_data" / "kerr_apriori_bound.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f'{r["beta"]:.2f} {r["xi"]:.2f}  apriori {r["apriori_infidelity_bound"]:.3e}  centered {r["centered_bound"]:.3e}  actual {r["actual"]:.3e}')
    print("certified 1e-3:", out["certified_1e3"])
    print("certified 1e-2:", out["certified_1e2"])


if __name__ == "__main__":
    main()
