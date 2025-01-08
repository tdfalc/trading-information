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

    savedir = Path(__file__).parent / "docs/sim02_binary_types_noncongruent"
    os.makedirs(savedir, exist_ok=True)

    def D(prob_state0):
        return (1 - 2 * prob_state0) * (1 - prob_state0)

    def tau_critical_l(theta_l, prob_state0):
        return theta_l / (prob_state0 * (2 * prob_state0 - 1))

    def tau_critical_h(theta_h, prob_state0):
        return (1 - theta_h) / D(prob_state0)

    gamma = 0.5
    tau = 1
    theta_h = 0.7
    theta_l = 0.2
    tau = 1
    N = 10000

    prob_state0s = np.linspace(0, 1, N)

    fig, ax = plt.subplots(figsize=(6.5, 3))
    # ax.plot(prob_state0s, tau_critical_l(theta_l, prob_state0s))

    with np.errstate(divide="ignore"):
        tau_critical_hs = tau_critical_h(theta_h, prob_state0s)
    tau_critical_hs[tau_critical_hs <= 0] = 1e-5

    with np.errstate(divide="ignore"):
        tau_critical_ls = tau_critical_l(theta_l, prob_state0s)
    tau_critical_ls[tau_critical_ls <= 0] = 1e-5

    with np.errstate(divide="ignore"):
        tau_critical_hs = tau_critical_h(theta_h, prob_state0s)
    tau_critical_hs[tau_critical_hs <= 0] = np.nan

    with np.errstate(divide="ignore"):
        tau_critical_ls = tau_critical_l(theta_l, prob_state0s)
    tau_critical_ls[tau_critical_ls <= 0] = np.nan

    ax.plot(prob_state0s, tau_critical_ls, color="k", ls="solid")
    ax.plot(prob_state0s, tau_critical_hs, color="k", ls="dashed")
    alpha = 0.5  # 0.2
    ax.fill_between(
        prob_state0s[prob_state0s <= 0.5],
        tau_critical_hs[prob_state0s <= 0.5],
        np.nanmax(tau_critical_hs),
        facecolor="#089cfc",
        alpha=alpha,
        label=r"$(x_L, x_H) = (R, 0)$",
        # hatch="//",
    )
    ax.fill_between(
        prob_state0s[prob_state0s >= 0.5],
        tau_critical_ls[prob_state0s >= 0.5],
        np.nanmax(tau_critical_ls),
        facecolor="#e86c44",
        alpha=alpha,
        label=r"$(x_L, x_H) = (0, 1)$",
        # hatch=".",
    )

    ax.fill_between(
        prob_state0s,
        0,
        np.concatenate([tau_critical_hs[prob_state0s <= 0.5], tau_critical_ls[prob_state0s > 0.5]]),
        facecolor="#40a44c",
        alpha=alpha,
        label=r"$(x_L, x_H) = (R, 1)$",
    )

    prettify(ax=ax, legend=True)

    # ax.set_ylim(top=100, bottom=0)
    # ax.set_yscale("log")
    ax.set_ylim(top=1, bottom=0.01)
    ax.set_xlim(left=0, right=1)
    ax.set_xlabel(r"Seller's Information $(\beta)$")
    ax.set_ylabel(r"Threshold $\tau$")

    fig.tight_layout()
    fig.savefig(savedir / "noncongruent.pdf", dpi=300)
