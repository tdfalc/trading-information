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
    logger.info("Running expected externality analysis")

    savedir = Path(__file__).parent / "docs/sim05_expected_externality"
    os.makedirs(savedir, exist_ok=True)

    num_types = 1000
    threshold_types = np.linspace(0.1, 0.9, 50)

    dist = Uniform(0, 1)
    dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))
    # dist = BetaMixture((8, 60), (30, 30), (0.01, 0.99))
    # dist = Uniform(0.5, 1)
    taus = np.linspace(0, 10, 50)

    cache_location = savedir / "cache"
    os.makedirs(cache_location, exist_ok=True)

    # We just need to record everything here and run

    @cache(save_dir=cache_location, use_cache=True)
    def run_experiment():
        all_externalities = np.zeros((num_types, len(taus)))
        all_transfers = np.zeros((num_types, len(taus)))
        all_allocations = np.zeros((num_types, len(taus)))
        for i, tau in enumerate(taus):
            prob_state0 = 1
            mechanism = Mechanism(num_types=num_types, dist=dist)
            hyperopt = HyperOpt(mechanism, threshold_types=threshold_types, verbose=100)
            hyperopt.run(tau, prob_state0)
            threshold_type = hyperopt.best()
            allocations, transfers, externalities, *_ = mechanism.solve(
                tau, prob_state0, threshold_type
            )

            externalities = mechanism._allocations_to_externalities(
                allocations, tau=tau, prob_state0=prob_state0
            )
            transfers = mechanism._allocations_to_transfers(allocations)

            all_externalities[:, i] = externalities
            all_transfers[:, i] = transfers
            all_allocations[:, i] = allocations
        return all_externalities, all_transfers, all_allocations

    all_externalities, all_transfers, all_allocations = run_experiment()

    types = np.linspace(0, 1, num_types)
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
