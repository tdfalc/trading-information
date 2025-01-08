import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt, HyperOptVirtuals
from trading_information.distributions import Uniform, BetaMixture
from trading_information.virtual_values import VirtualValues

import sys

sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from tfds.plotting import prettify, use_tex

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt, HyperOptVirtuals
from trading_information.distributions import Uniform, BetaMixture
from trading_information.virtual_values import VirtualValues

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.cm as cm


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from trading_information.typing import _Floats
from typing import Optional

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from tfds.log import create_logger
from tfds.plotting import prettify, use_tex
from tfds.decorators import cache, blind_file_cache
from matplotlib.axes import Axes

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt
from trading_information.distributions import Uniform, BetaMixture, Distribution
from trading_information.virtual_values import VirtualValues
from trading_information.typing import _Floats


from analytics.sim03_uniform_types_continuous import (
    _plot_informativeness,
    _plot_transfer,
    _plot_externality,
)


def _plot_virtual_value(
    axs: Axes, midpoints: _Floats, positive: _Floats, negative: _Floats, lag: float, i: int
) -> None:
    ax = axs[0, i]
    ax.sharey(axs[0, 0])
    ax.plot(midpoints, positive, color="k", label=r"$\phi^+$")
    ax.plot(midpoints, negative, color="k", ls="dashed", label=r"$\phi^-$")
    ax.axhline(y=lag, color="red", ls="solid", zorder=0, label=r"$\lambda$")
    where = (negative > lag) & (positive <= lag)
    ax.fill_between(midpoints, negative, positive, where=where, color="yellow", alpha=0.3)
    where = negative <= lag
    ax.fill_between(midpoints, negative, positive, where=where, color="blue", alpha=0.3)
    where = positive > lag
    ax.fill_between(midpoints, negative, positive, where=where, color="blue", alpha=0.3)

    where = (np.abs(negative - lag) <= 1e-3) | (np.abs(positive - lag) <= 1e-3)
    ax.fill_between(midpoints, negative, positive, where=where, color="red", alpha=0.3)

    prettify(ax=ax, legend=(i == 3), legend_loc="lower right")
    if i == 0:
        ax.set_ylabel("Virtual Value")
    else:
        ax.tick_params(labelleft=False)


def main():
    # logger = create_logger(__name__)
    # logger.info("Running optimal allocations analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim04_bimodal_types_continuous"
    os.makedirs(savedir, exist_ok=True)

    num_intervals = 1000
    dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))  # bimodal
    prob_state0 = 0
    taus = [0, 0.3, 0.45, 1]

    fig, axs = plt.subplots(4, len(taus), figsize=(6, 8), sharex=True)

    for i, tau in enumerate(taus):

        mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
        midpoints = mechanism.midpoints

        threshold_indices = [551]  # np.arange(500, 600)
        hyperopt = HyperOpt(mechanism=mechanism, threshold_indices=threshold_indices)

        hyperopt.run(tau, prob_state0, desc="threshold")
        threshold_index = hyperopt.best()
        print(threshold_index)

        allocations, bins, avg_transfer, avg_externality, lag, obj = (
            mechanism.solve_with_increments(
                tau=tau, prob_state0=prob_state0, threshold_index=threshold_index
            )
        )
        lag *= num_intervals
        weighted_pdfs = 1  # mechanism._pdfs / 1  # np.sum(mechanism._pdfs)
        transfers = mechanism._allocations_to_transfers(allocations) * weighted_pdfs
        externalities = (
            mechanism._allocations_to_externalities(allocations, tau=tau, prob_state0=prob_state0)
            * weighted_pdfs
        )
        positive, negative = VirtualValues.get_ironed_values(dist, midpoints, tau, prob_state0)

        _plot_virtual_value(axs, midpoints, positive, negative, lag, i)
        _plot_informativeness(axs, midpoints, allocations, i)
        _plot_transfer(axs, midpoints, transfers, i)
        _plot_externality(axs, midpoints, externalities, i)

        axs[1, i].axvline(x=0.5)

        for j in [1, 2, 3]:
            ax = axs[j, i].twinx()
            ax.plot(midpoints, mechanism._pdfs, alpha=0.1)
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(savedir / "bimodal_types_continuous.pdf", dpi=300)


if __name__ == "__main__":
    main()
