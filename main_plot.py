import numpy as np
from matplotlib import pyplot as plt

import numpy as np
from tfds.log import create_logger
from tfds.plotting import prettify, use_tex

use_tex()


if __name__ == "__main__":
    types = np.linspace(0, 1, 1000)
    allocations1 = np.zeros(len(types))
    allocations1[:250] = -1
    allocations1[250:750] = 0
    allocations1[750:] = 1

    allocations2 = np.zeros(len(types))
    allocations2[:750] = -1 / 3
    allocations2[750:] = 1

    allocations1[np.diff(allocations1, prepend=allocations1[0]) > 0.5] = np.nan
    allocations2[np.diff(allocations2, prepend=allocations2[0]) > 0.5] = np.nan

    transfers1 = np.zeros(len(types))
    transfers1[:250] = 0
    transfers1[250:750] = 0.5
    transfers1[750:] = 0

    externalities1 = np.zeros(len(types))
    externalities1[:250] = 0
    externalities1[250:750] = types.copy()[250:750]
    externalities1[750:] = 1

    externalities2 = np.zeros(len(types))
    externalities2[:750] = types.copy()[:750] * 0.5
    externalities2[750:] = 1

    transfers1[np.diff(transfers1, prepend=transfers1[0]) > 0.3] = np.nan
    transfers1[np.diff(transfers1, prepend=transfers1[0]) < -0.3] = np.nan

    transfers2 = np.zeros(len(types))

    fig, axs = plt.subplots(3, 2, figsize=(9, 9), sharey=False, dpi=300)
    axs = axs.flatten()
    axs[0].sharey(axs[1])
    axs[2].sharey(axs[3])
    axs[4].sharey(axs[5])

    axs[4].plot(types, 1 - externalities1, color="blue")
    axs[5].plot(types, 1 - externalities2, color="darkorange")
    axs[4].axhline(y=np.nanmean(1 - externalities1), color="k", alpha=0.1)
    axs[5].axhline(y=np.nanmean(1 - externalities2), color="k", alpha=0.1)

    axs[0].axhline(y=np.nanmean(allocations1), color="k", alpha=0.1)
    axs[1].axhline(y=np.nanmean(allocations2), color="k", alpha=0.1)

    axs[2].axhline(y=np.nanmean(transfers1), color="k", alpha=0.1)
    axs[3].axhline(y=np.nanmean(transfers2), color="k", alpha=0.1)

    kwargs = dict(
        # markerfacecolor="white",
        ls="solid",
        # lw=1,
    )
    axs[0].plot(
        types,
        allocations1,
        label="Full Info",
        color="blue",
        # markeredgecolor="blue",
        # marker="o",
        # markevery=40,
        zorder=1,
        **kwargs,
    )
    axs[0].plot(
        [0.25, 0.251],
        [-1, 0],
        # label="Full Info",
        color="blue",
        ls="dashed",
        lw=0.5,
        # markeredgecolor="blue",
        # marker="o",
        # markevery=40,
        zorder=1,
        # **kwargs,
    )
    axs[0].plot(
        [0.75, 0.751],
        [0, 1],
        # label="Full Info",
        color="blue",
        ls="dashed",
        lw=0.5,
        # markeredgecolor="blue",
        # marker="o",
        # markevery=40,
        zorder=1,
        # **kwargs,
    )
    axs[0].scatter(
        types[0:1],
        allocations1[0:1],
        color="blue",
        facecolor="white",
        zorder=2,
    )
    axs[0].scatter(
        types[249:250],
        allocations1[249:250],
        color="blue",
        facecolor="white",
        zorder=2,
    )

    axs[0].scatter(
        types[251:252],
        allocations1[251:252],
        color="blue",
        facecolor="white",
        zorder=2,
    )
    axs[0].scatter(
        types[749:750],
        allocations1[749:750],
        color="blue",
        facecolor="white",
        zorder=2,
    )
    axs[0].scatter(
        types[751:752],
        allocations1[751:752],
        color="blue",
        facecolor="white",
        zorder=2,
    )
    axs[0].scatter(
        types[-1:],
        allocations1[-1:],
        color="blue",
        facecolor="white",
        zorder=2,
    )

    axs[1].plot(
        types,
        allocations2,
        label="Half Info",
        color="darkorange",
        # markeredgecolor="darkorange",
        # marker="x",
        # markevery=40,
        **kwargs,
    )
    axs[1].scatter(
        types[0:1],
        allocations2[0:1],
        color="darkorange",
        facecolor="white",
        zorder=2,
    )
    axs[1].scatter(
        types[749:750],
        allocations2[749:750],
        color="darkorange",
        facecolor="white",
        zorder=2,
    )
    axs[1].scatter(
        types[751:752],
        allocations2[751:752],
        color="darkorange",
        facecolor="white",
        zorder=2,
    )
    axs[1].scatter(
        types[-1:],
        allocations2[-1:],
        color="darkorange",
        facecolor="white",
        zorder=2,
    )
    axs[1].plot(
        [0.75, 0.751],
        [-1 / 3, 1],
        # label="Full Info",
        color="darkorange",
        ls="dashed",
        lw=0.5,
        # markeredgecolor="blue",
        # marker="o",
        # markevery=40,
        zorder=1,
        # **kwargs,
    )

    axs[2].plot(
        types,
        transfers1,
        color="blue",
        zorder=1,
        # lw=1,
    )
    axs[2].plot(
        [0.25, 0.251],
        [0, 0.5],
        # label="Full Info",
        color="blue",
        ls="dashed",
        lw=0.5,
        # markeredgecolor="blue",
        # marker="o",
        # markevery=40,
        zorder=1,
        # **kwargs,
    )
    axs[2].plot(
        [0.75, 0.751],
        [0.5, 0],
        # label="Full Info",
        color="blue",
        ls="dashed",
        lw=0.5,
        # markeredgecolor="blue",
        # marker="o",
        # markevery=40,
        zorder=1,
        # **kwargs,
    )
    axs[2].scatter(
        types[0:1],
        transfers1[0:1],
        color="blue",
        facecolor="white",
        zorder=2,
    )
    axs[2].scatter(
        types[249:250],
        transfers1[249:250],
        color="blue",
        facecolor="white",
        zorder=2,
    )

    axs[2].scatter(
        types[251:252],
        transfers1[251:252],
        color="blue",
        facecolor="white",
        zorder=2,
    )
    axs[2].scatter(
        types[749:750],
        transfers1[749:750],
        color="blue",
        facecolor="white",
        zorder=2,
    )
    axs[2].scatter(
        types[751:752],
        transfers1[751:752],
        color="blue",
        facecolor="white",
        zorder=2,
    )
    axs[2].scatter(
        types[-1:],
        transfers1[-1:],
        color="blue",
        facecolor="white",
        zorder=2,
    )

    axs[3].plot(
        types,
        transfers2,
        color="darkorange",
        zorder=1,
        # lw=1,
    )
    axs[3].scatter(
        types[0:1],
        transfers2[0:1],
        color="darkorange",
        facecolor="white",
        zorder=2,
    )
    axs[3].scatter(
        types[-1:],
        transfers2[-1:],
        color="darkorange",
        facecolor="white",
        zorder=2,
    )

    axs[1].set_ylabel("Transfer ($\\pi_j$)")
    axs[3].set_ylabel("Transfer ($\\pi_j$)")
    axs[1].set_xlabel("Private Type ($t_i$)")
    axs[3].set_xlabel("Private Type ($t_i$)")

    axs[0].set_xlabel("Private Type ($t_i$)")
    axs[2].set_xlabel("Private Type ($t_i$)")
    axs[0].set_ylabel("Allocation ($\\xi_j$)")
    axs[2].set_ylabel("Allocation ($\\xi_j$)")

    axs[4].set_xlabel("Private Type ($t_i$)")
    axs[5].set_xlabel("Private Type ($t_i$)")

    axs[4].set_ylabel("Profit")
    axs[5].set_ylabel("Profit")
    # axs[0].legend()

    # axs[0].fill_between(
    #     [0.25, 0.75],
    #     [-1, -1],
    #     [1, 1],
    #     alpha=0.1,
    #     color="gray",
    # )
    # axs[0].fill_between([0.0, 0.25], [-1, -1], [1, 1], alpha=0.1, color="gray")
    # axs[0].fill_between([0.75, 1.0], [-1, -1], [1, 1], alpha=0.1, color="gray")
    # axs[1].fill_between([0.0, 0.25], [-1, -1], [1, 1], alpha=0.1, color="gray")
    # axs[1].fill_between([0.75, 1.0], [-1, -1], [1, 1], alpha=0.1, color="gray")
    # axs[2].fill_between([0.75, 1.0], [-1, -1], [1, 1], alpha=0.1, color="gray")

    prettify(ax=axs[0], legend=False)
    prettify(ax=axs[1], legend=False)
    prettify(ax=axs[2], legend=False)
    prettify(ax=axs[3], legend=False)
    prettify(ax=axs[4], legend=False)
    prettify(ax=axs[5], legend=False)

    fig.tight_layout()

    fig.savefig("./alls_plot.pdf", dpi=300)
