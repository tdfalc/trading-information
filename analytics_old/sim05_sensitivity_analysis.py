import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from tfds.log import create_logger
from tfds.plotting import prettify, use_tex
from tfds.decorators import cache
from matplotlib.axes import Axes

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt
from trading_information.distributions import Uniform, BetaMixture
from trading_information.typing import _Floats


def main():
    logger = create_logger(__name__)
    logger.info("Running sensitivity analysis")

    savedir = Path(__file__).parent / "docs/sim05_sensitivity_analysis"
    os.makedirs(savedir, exist_ok=True)

    num_intervals = 1000
    threshold_indices = np.arange(100, 900, 1).astype(int)
    threshold_indices = np.array([500]).astype(int)

    distributions = {
        "uniform": Uniform(0, 1),
        "bimodal": BetaMixture((8, 60), (30, 30), (0.5, 0.5)),
    }

    dist = Uniform(0, 1)
    # dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))

    taus = np.linspace(0, 2, 20)

    cache_location = savedir / "cache"
    os.makedirs(cache_location, exist_ok=True)

    # We just need to record everything here and run

    @cache(save_dir=cache_location, use_cache=True)
    def run_experiment():
        all_externalities = np.zeros((num_intervals, len(taus)))
        all_transfers = np.zeros((num_intervals, len(taus)))
        all_allocations = np.zeros((num_intervals, len(taus)))
        for i, tau in enumerate(taus):
            prob_state0 = 0
            mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
            hyperopt = HyperOpt(mechanism, threshold_indices=threshold_indices)
            hyperopt.run(tau, prob_state0, desc=f"{i}/{len(taus)}")
            threshold_type = hyperopt.best()
            allocations, transfers, externalities, *_ = mechanism.solve(
                tau, prob_state0, threshold_type
            )

            all_externalities[:, i] = externalities
            all_transfers[:, i] = transfers
            all_allocations[:, i] = allocations
        return all_externalities, all_transfers, all_allocations

    all_externalities, all_transfers, all_allocations = run_experiment()

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    types = mechanism.midpoints
    exp_externalities = np.sum(all_externalities * dist.pdf(types).reshape(-1, 1), axis=0) / 1000

    print(dist.pdf(types), np.sum(dist.pdf(types)))

    fig, ax = plt.subplots()
    ax.plot(taus, exp_externalities)
    ax.set_xlabel("Tau")
    ax.set_ylabel("Expected Externality")
    ax.axhline(y=1)
    prettify(ax=ax)
    fig.savefig(savedir / "avg_externalities.pdf", dpi=300)

    avg_transfers = np.mean(all_transfers, axis=0)
    fig, ax = plt.subplots()
    ax.plot(exp_externalities, avg_transfers)
    ax.set_xlabel("Expected Externality")
    ax.set_ylabel("Expected Transfer")
    prettify(ax=ax)
    fig.savefig(savedir / "avg_transfers.pdf", dpi=300)


if __name__ == "__main__":
    main()
