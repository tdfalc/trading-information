import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from trading_information.mechanism import Mechanism
from trading_information.hyperopt import HyperOpt, HyperOptVirtuals
from trading_information.distributions import Uniform, BetaMixture
from trading_information.virtual_values import VirtualValues


def main():
    # logger = create_logger(__name__)
    # logger.info("Running optimal allocations analysis")

    savedir = Path(__file__).parent / "docs/sim01_objective_concavity"
    os.makedirs(savedir, exist_ok=True)

    num_intervals = 1000
    tau = 20
    prob_state0 = 1

    # dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))  # bimodal
    dist = Uniform(low=0, high=1)  # uniform
    dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(midpoints, dist.pdf(midpoints))
    ax.set_ylabel(r"Density")
    ax.set_xlabel("Private Type ($t_i$)")
    fig.savefig(savedir / f"densities.pdf", dpi=300)

    def check_convexity_advanced(x, y, tol=1e-3):
        """
        Advanced check for convexity or concavity using gradients.

        Parameters:
        - x (np.ndarray): 1D array of x-coordinates.
        - y (np.ndarray): 1D array of y-coordinates.
        - tol (float): Tolerance for numerical precision.

        Returns:
        - str: 'convex', 'concave', or 'neither'
        """
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)

        ans = "neither"
        if np.all(d2y >= -tol):
            ans = "convex"
        if np.all(d2y <= tol):
            ans += "concave"
        return ans

    dist = BetaMixture(alphas=(8, 60), betas=(30, 30), weights=(0.5, 0.5))  # bimodal
    # dist = Uniform(low=0, high=1)  # uniform

    mechanism = Mechanism(num_intervals=num_intervals, dist=dist)
    midpoints = mechanism.midpoints

    fig, ax = plt.subplots(figsize=(4, 4))
    allocations = np.linspace(-1, 1, 1000)

    positive, negative = VirtualValues.get_ironed_values(
        dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0
    )
    import matplotlib as mpl
    import matplotlib.cm as cm
    from itertools import cycle

    prob_state0 = 1
    taus = [0.8, 1.2]
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(taus))
    cmap = cm.get_cmap("viridis")
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for tau in taus:
        i = 400

        positive, negative = VirtualValues.get_ironed_values(
            dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0
        )
        virtual_values = (
            np.minimum(allocations, 0) * negative[i]
            + np.maximum(allocations, 0) * positive[i]
            # - 0.6682082082082083 * allocations
        )
        # ax.plot(midpoints, virtual_values, label=i, color=m.to_rgba(tau), zorder=1)

        x = allocations
        y = virtual_values
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)

        ax.plot(allocations, y, label=i, zorder=1)

        # print("check_convexity", check_convexity_advanced(allocations, virtual_values))

    prob_state0 = 0.8
    tau = 2

    # ax.set_xlim(right=1.6)
    # ax.set_ylim(top=1.6)

    fig.savefig(savedir / "concavity.pdf")

    fig, ax = plt.subplots(figsize=(4.5, 3))
    virtual_values = VirtualValues(
        dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=True
    )
    ax.plot(
        midpoints,
        virtual_values.negative,
        color="k",
        ls="solid",
        label=r"$\phi^{-}$",
    )

    ax.plot(
        midpoints,
        virtual_values.positive,
        color="blue",
        ls="solid",
        label=r"$\phi^{+}$",
    )

    # virtual_values = VirtualValues(
    #     dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=False
    # )
    # ax.plot(
    #     midpoints,
    #     virtual_values.negative,
    #     color="k",
    #     ls="dashed",
    #     label=r"$\phi^{-}$",
    # )
    # ax.plot(
    #     midpoints,
    #     virtual_values.positive,
    #     color="blue",
    #     ls="dashed",
    #     label=r"$\phi^{-}$",
    # )

    virtual_values = VirtualValues(
        dist=dist, types=midpoints, tau=tau, prob_state0=prob_state0, iron=True
    )

    i, j = 250, 750
    ax.fill_between(
        midpoints[:i],
        virtual_values.negative[:i],
        virtual_values.positive[:i],
        color="y",
        alpha=0.3,
    )
    ax.fill_between(
        midpoints[i:j],
        virtual_values.negative[i:j],
        virtual_values.positive[i:j],
        color="b",
        alpha=0.3,
    )
    ax.fill_between(
        midpoints[j:],
        virtual_values.negative[j:],
        virtual_values.positive[j:],
        color="y",
        alpha=0.3,
    )

    # ax.fill_between(midpoints[:250], 2, -1, color="y", alpha=0.3)
    # ax.fill_between(midpoints[250:750], 2, -1, color="b", alpha=0.3)
    # ax.fill_between(midpoints[750:], 2, -1, color="y", alpha=0.3)

    ax.axhline(y=0.05, color="red", label="$\lambda$")
    ax.axvline(x=i / 1000, color="k", ls="dashed")
    ax.axvline(x=j / 1000, color="k", ls="dashed")
    ax.legend()
    # ax2 = ax.twinx()
    # ax2.plot(midpoints, allocations, color="red")
    # ax.set_ylim(top=10, bottom=-5)

    ax.set_ylabel("Virtual Value")
    fig.tight_layout()
    fig.savefig(savedir / f"virtuals.pdf", dpi=300)


if __name__ == "__main__":
    main()
