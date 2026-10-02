"""Compare archived reference observables on the circuit's common times."""

import json
import numpy as np
from sine_gordon_code import sglib as sg
from common_code.run_pilot import _sg_observations
from common_code.records import ROOT, provenance, write_json


def main():
    sources = {
        96: ROOT / "double_sine_gordon_data/sg128/reference",
        128: ROOT / "double_sine_gordon_data/sg128/reference",
        160: ROOT / "double_sine_gordon_data/sg160/reference",
    }
    observed = {}
    times = {}
    for n, directory in sources.items():
        with np.load(directory / f"E4_reference_lambda0.3_N{n}.npz") as data:
            states = data["target_step_states"].copy()
            times[n] = data["step_t"].copy()
        observed[n] = _sg_observations(states, sg.build_mode_basis(n, np.sqrt(3)))
    rows = []
    for low, high in ((96, 128), (128, 160)):
        np.testing.assert_allclose(
            times[low],
            times[high],
            atol=4 * np.finfo(float).eps * max(1.0, times[high][-1]),
            rtol=0,
        )
        c = lambda n: observed[n]["C12_real"] + 1j * observed[n]["C12_imag"]
        common = dict(
            C12=float(np.max(abs(c(low) - c(high)))),
            S2=float(np.max(abs(observed[low]["S2"] - observed[high]["S2"]))),
        )
        archived = json.loads((sources[high] / "summary.json").read_text())[
            "empirical_cutoff_estimates"
        ]
        rows.append(
            dict(
                cutoffs=[low, high],
                common_11_times=common,
                archived_257_times={key: archived[key] for key in common},
                common_grid_exceeds_archived={
                    key: common[key] > archived[key] for key in common
                },
            )
        )
    write_json(
        ROOT / "common_data/records/sg_reference_common_times.json",
        dict(
            rows=rows,
            provenance=provenance(),
            scope="Contractions of existing exact step states only. No trajectory or independent-solver rerun. Original solver observable differences remain empirical estimates on their archived 257-point grid.",
        ),
    )
    print(rows)


if __name__ == "__main__":
    main()
