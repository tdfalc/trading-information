import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tfds.plotting import prettify, use_tex


def D(prob_state0):
    return 1 - 3 * prob_state0 + 2 * (prob_state0**2)


def gamma_critical_l(theta_h, theta_l, tau, prob_state0):
    return (1 - theta_l - tau * D(prob_state0)) / (1 - theta_h - tau * D(prob_state0))


if __name__ == "__main__":

    use_tex()

    savedir = Path(__file__).parent / "docs/sim01_binary_types_congruent"
    os.makedirs(savedir, exist_ok=True)

    theta_h = 2 / 3
    theta_l = 5 / 6
    tau = 1
    N = 10000

    prob_state0s = np.linspace(0, 1, N)

    gammas = [
        0.01,
        0.1,
        0.2,
        0.3,
        0.4,
        0.45,
        0.49,
        0.51,
        0.6,
        0.7,
        0.85,
        0.86,
        0.87,
        0.88,
        0.89,
        0.9,
    ]

    for gamma in gammas:
        print(gamma)

        # fig, axs  = plt.subplots(1, 3, figsize=(12, 3 ))

        fig, ax = plt.subplots(figsize=(6.5, 3))
        # ax.plot(prob_state0s, tau_critical_l(theta_l, prob_state0s))

        # with np.errstate(divide='ignore'):
        #     tau_critical_hs = tau_critical_h(theta_h, prob_state0s)
        # tau_critical_hs[tau_critical_hs <= 0] = 1e-5

        # with np.errstate(divide='ignore'):
        #     tau_critical_ls = tau_critical_l(theta_l, prob_state0s)
        # tau_critical_ls[tau_critical_ls <= 0] = 1e-5

        def D(prob_state0):
            return (1 - 2 * prob_state0) * (1 - prob_state0)

        def tau_critical_l(theta_l, theta_h, prob_state0):
            return ((1 - theta_l) - gamma * (1 - theta_h)) / ((1 - gamma) * D(prob_state0))

        def tau_critical_h(theta_h, prob_state0):
            return (1 - theta_h) / D(prob_state0)

        with np.errstate(divide="ignore"):
            tau_critical_hs = tau_critical_h(theta_h, prob_state0s)
        tau_critical_hs[tau_critical_hs <= 0] = 1e5

        with np.errstate(divide="ignore"):
            tau_critical_ls = tau_critical_l(theta_l, theta_h, prob_state0s)
        tau_critical_ls[tau_critical_ls <= 0] = 1e-5

        ax.plot(prob_state0s, tau_critical_ls, color="k", ls="solid")
        ax.plot(prob_state0s, tau_critical_hs, color="k", ls="dashed")

        ax.fill_between(
            prob_state0s[prob_state0s <= 0.5],
            tau_critical_hs[prob_state0s <= 0.5],
            np.nanmax(tau_critical_hs),
            facecolor="red",
            alpha=0.2,
            label=r"$(x_L, x_H) = (0, 0)$",
        )

        ax.fill_between(
            prob_state0s,
            tau_critical_ls,
            tau_critical_hs,
            facecolor="blue",
            alpha=0.2,
            label=r"$(x_L, x_H) = (0, 1)$",
        )

        ax.fill_between(
            prob_state0s,
            0,
            tau_critical_ls,
            facecolor="green",
            alpha=0.2,
            label=r"$(x_L, x_H) = (1, 1)$",
        )

        prettify(ax=ax, legend=True)

        ax.set_ylim(top=1, bottom=0.01)
        ax.set_xlim(left=0, right=1)
        ax.set_xlabel(r"Seller's Information $(\beta)$")
        ax.set_ylabel(r"Threshold $\tau$")
        fig.tight_layout()
        fig.savefig(savedir / f"congruent_gamma{gamma}.pdf", dpi=300)
