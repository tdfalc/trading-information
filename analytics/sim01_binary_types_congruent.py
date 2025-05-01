# import os
# from typing import Tuple
# from pathlib import Path
# import io

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from tfds.plotting import prettify, use_tex
# from tfds.log import create_logger

# from analytics.sim03_uniform_types_continuous import add_discontinuities

# logger = create_logger(__name__)


# class Colors:
#     black = "#000000"
#     orange = "#e69f00"
#     skyblue = "#56b4e9"
#     green = "#009e73"
#     yellow = "#f0e442"
#     blue = "#0072b2"
#     vermilion = "#d55e00"
#     purple = "#cc79a7"


# def D(prob_state0):
#     return (1 - 2 * prob_state0) * (1 - prob_state0)


# def tau_critical_low(type_low: float, type_high: float, prob_state0: float, phi: float) -> float:
#     return ((1 - type_low) - phi * (1 - type_high)) / ((1 - phi) * D(prob_state0))


# def tau_critical_high(type_high: float, prob_state0: float) -> float:
#     return (1 - type_high) / D(prob_state0)


# def compute_critical_taus(
#     type_low: float, type_high: float, prob_state0s: np.ndarray, phi: float
# ) -> Tuple:

#     with np.errstate(divide="ignore", invalid="ignore"):
#         # Compute critical taus for the high type
#         taus_critical_high = tau_critical_high(type_high, prob_state0s)
#         # taus_critical_high[taus_critical_high <= 0] = np.nan  # 1e5

#         # Compute critical taus for the low type
#         taus_critical_low = tau_critical_low(type_low, type_high, prob_state0s, phi)
#         # taus_critical_low[taus_critical_low <= 0] = np.nan  # 1e-5

#     return taus_critical_low, taus_critical_high


# if __name__ == "__main__":

#     logger.info("Running binary types (congruent) analysis")

#     use_tex()

#     savedir = Path(__file__).parent / "docs/sim01_binary_types_congruent"
#     os.makedirs(savedir, exist_ok=True)

#     type_high = 2 / 3
#     type_low = 5 / 6
#     num_types = 1000
#     prob_state0s = np.linspace(0, 1, num_types)

#     fig, axs = plt.subplots(3, 1, figsize=(4, 2.5))

#     def _plot_results(phi, ax, xlabel=True, legend=False):

#         taus_critical_low, taus_critical_high = compute_critical_taus(
#             type_low, type_high, prob_state0s, phi
#         )

#         ax.plot(
#             prob_state0s,
#             add_discontinuities(taus_critical_low, threshold=0.1),
#             color="k",
#             ls="solid",
#         )
#         ax.plot(
#             prob_state0s,
#             add_discontinuities(taus_critical_high, threshold=0.1),
#             color="k",
#             ls="dashed",
#         )

#         alpha = 0.7

#         ax.fill_between(
#             prob_state0s,
#             np.where(prob_state0s <= 0.5, taus_critical_high, taus_critical_high),
#             np.where(
#                 prob_state0s <= 0.5, np.nanmax(taus_critical_high), np.nanmin(taus_critical_high)
#             ),
#             facecolor=Colors.orange,
#             alpha=alpha,
#             label=r"$(I^l, I^h) = (1, 1)$",
#         )

#         ax.fill_between(
#             prob_state0s,
#             np.where(prob_state0s <= 0.5, taus_critical_low, taus_critical_high),
#             np.where(prob_state0s <= 0.5, taus_critical_high, taus_critical_low),
#             facecolor=Colors.blue,
#             alpha=alpha,
#             label=r"$(I^l, I^h) = (1, 0)$",
#         )

#         ax.fill_between(
#             prob_state0s,
#             np.where(prob_state0s <= 0.5, 0, 1e5),
#             np.where(prob_state0s <= 0.5, taus_critical_low, taus_critical_low),
#             facecolor=Colors.green,
#             alpha=alpha,
#             label=r"$(I^l, I^h) = (0, 0)$",
#         )

#         ax.set_ylim(top=1, bottom=0.01)
#         ax.set_xlim(left=0, right=1)
#         ax.set_ylabel(r"Threshold $\tau$")
#         if xlabel:
#             ax.set_xlabel(r"Seller's Information ($v_s$)")
#         prettify(ax=ax, legend=True if legend else False)

#     # Save static figure for paper
#     phis = [0.36, 0.5, 0.56]
#     for i, phi in enumerate(phis):
#         ax = axs[i]
#         if i == 1:
#             ax.axvline(x=0.5, color="k")
#         ax.set_title(f"$\\phi = {phi:.1f}$")
#         _plot_results(
#             phi,
#             ax,
#             xlabel=True if i == len(phis) - 1 else False,
#             legend=True if i == 0 else False,
#         )

#     fig.tight_layout()
#     fig.savefig(savedir / f"binary_types_congruent.pdf", dpi=300)

#     # Make gif with pillow
#     frames = []
#     phis = np.concatenate((np.linspace(0.01, 0.49, 49), np.linspace(0.51, 0.65, 15)))

#     for phi in phis:
#         fig, ax = plt.subplots(figsize=(6, 3))
#         _plot_results(phi, ax, legend=True)
#         ax.set_title(f"$\\phi = {phi:.2f}$")
#         fig.tight_layout()

#         # Save as BytesIO object temporarily to avoid saving every frame to disc
#         buf = io.BytesIO()
#         fig.savefig(buf, format="png", dpi=300)
#         plt.close(fig)  # Close the figure to free memory

#         # Load the image from the buffer
#         buf.seek(0)
#         img = Image.open(buf)
#         frames.append(img)

#     frames[0].save(
#         savedir / "binary_types_congruent.gif",
#         save_all=True,
#         append_images=frames[1:],
#         optimize=False,
#         duration=100,  # Duration per frame in milliseconds
#         loop=0,  # Loop forever
#     )


import os
from typing import Tuple
from pathlib import Path
import io

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tfds.plotting import prettify, use_tex
from tfds.log import create_logger

from analytics.sim03_uniform_types_continuous import add_discontinuities

logger = create_logger(__name__)


class Colors:
    black = "#000000"
    orange = "#e69f00"
    skyblue = "#56b4e9"
    green = "#009e73"
    yellow = "#f0e442"
    blue = "#0072b2"
    vermilion = "#d55e00"
    purple = "#cc79a7"


def D(prob_state0):
    return (1 - 2 * prob_state0) * (1 - prob_state0)


def tau_critical_low(type_low: float, type_high: float, prob_state0: float, phi: float) -> float:
    return ((1 - type_low) - phi * (1 - type_high)) / ((1 - phi) * D(prob_state0))


def tau_critical_high(type_high: float, prob_state0: float) -> float:
    return (1 - type_high) / D(prob_state0)


def compute_critical_taus(
    type_low: float, type_high: float, prob_state0s: np.ndarray, phi: float
) -> Tuple:

    with np.errstate(divide="ignore", invalid="ignore"):
        # Compute critical taus for the high type
        taus_critical_high = tau_critical_high(type_high, prob_state0s)
        # taus_critical_high[taus_critical_high <= 0] = np.nan  # 1e5

        # Compute critical taus for the low type
        taus_critical_low = tau_critical_low(type_low, type_high, prob_state0s, phi)
        # taus_critical_low[taus_critical_low <= 0] = np.nan  # 1e-5

    return taus_critical_low, taus_critical_high


if __name__ == "__main__":

    logger.info("Running binary types (congruent) analysis")

    use_tex()

    savedir = Path(__file__).parent / "docs/sim01_binary_types_congruent"
    os.makedirs(savedir, exist_ok=True)

    type_high = 2 / 3
    type_low = 5 / 6
    num_types = 1000
    prob_state0s = np.linspace(0, 1, num_types)

    def _plot_results(phi, ax, xlabel=True, legend=False):

        taus_critical_low, taus_critical_high = compute_critical_taus(
            type_low, type_high, prob_state0s, phi
        )

        ax.plot(
            prob_state0s,
            add_discontinuities(taus_critical_low, threshold=0.1),
            color="k",
            ls="solid",
            lw=1,
        )
        ax.plot(
            prob_state0s,
            add_discontinuities(taus_critical_high, threshold=0.1),
            color="k",
            ls="dashed",
            lw=1,
        )

        ax.annotate(
            "",
            xy=(2 / 3, -0.05),
            xytext=(2 / 3, 0.05),  # Adds another annotation for the text that you want
            arrowprops=dict(arrowstyle="-"),  # sets style of arrow and colour
            annotation_clip=False,
            color="k",
        )

        ax.annotate(
            r"$v_b^h$",
            xy=(0.63, 0),
            xytext=(0.63, 0.08),  # Adds another annotation for the text that you want
            annotation_clip=False,
            color="k",
        )

        ax.annotate(
            "",
            xy=(5 / 6, -0.05),
            xytext=(5 / 6, 0.05),  # Adds another annotation for the text that you want
            arrowprops=dict(arrowstyle="-"),  # sets style of arrow and colour
            annotation_clip=False,
            color="k",
        )

        ax.annotate(
            r"$v_b^l$",
            xy=(0.8, 0),
            xytext=(0.8, 0.08),  # Adds another annotation for the text that you want
            annotation_clip=False,
            color="k",
        )

        # ax.fill_between(
        #     prob_state0s,
        #     np.where(prob_state0s <= 0.5, taus_critical_high, taus_critical_high),
        #     np.where(
        #         prob_state0s <= 0.5, np.nanmax(taus_critical_high), np.nanmin(taus_critical_high)
        #     ),
        #     facecolor="k",
        #     alpha=0.3,
        #     label=r"$(1, 1)$",
        # )

        # ax.fill_between(
        #     prob_state0s,
        #     np.where(prob_state0s <= 0.5, taus_critical_low, taus_critical_high),
        #     np.where(prob_state0s <= 0.5, taus_critical_high, taus_critical_low),
        #     facecolor="k",
        #     alpha=0.1,
        #     label=r"$(1, 0)$",
        # )

        # ax.fill_between(
        #     prob_state0s,
        #     np.where(prob_state0s <= 0.5, 0, 1e5),
        #     np.where(prob_state0s <= 0.5, taus_critical_low, taus_critical_low),
        #     facecolor="w",
        #     alpha=1,
        #     label=r"$(0, 0)$",
        # )

        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(top=1, bottom=0)
        ax.set_xlim(left=0, right=1)
        ax.set_ylabel(r"$\tau$")
        if xlabel:
            ax.set_xlabel(r"$v_s$")
        prettify(ax=ax, legend=False if legend else False, legend_loc="lower right")

    # Save static figure for paper
    phis = [0.36, 0.5, 0.56]
    for i, phi in enumerate(phis):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        if i == 1:
            ax.axvline(x=0.5, color="k", lw=1)
        # ax.set_title(f"$\\phi = {phi:.1f}$")
        _plot_results(
            phi,
            ax,
            xlabel=True,
            legend=True if i == 0 else False,
        )
        j = ["a", "b", "c"][i]

        # if i == 0:
        # ax.text(0.125, 0.9, r"$(1, 0)$", ha="center", size=10)
        # ax.text(0.15, 0.25, r"$(1, 1)$", ha="center", size=10)
        # ax.text(0.75, 0.5, r"$(0, 0)$", ha="center", size=10)

        fig.tight_layout()
        fig.savefig(savedir / f"binary_types_congruent_{j}.pdf", dpi=300)
