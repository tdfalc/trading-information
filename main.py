from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import brentq, root

FIGURE_WIDTH = 7.2  # Inches.
BLACK = "#262626"
GREY = "#90959B"
BLUE = "C0"
ORANGE = "C1"
PURPLE = "C4"
GREEN = "C2"

STYLE = {
    "font.size": 12,
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "axes.linewidth": 1,
    "legend.facecolor": "#eeeeee",
    "legend.edgecolor": "#ffffff",
    "legend.framealpha": 0.85,
    "legend.labelspacing": 0.25,
    "figure.facecolor": "white",
    "figure.autolayout": True,
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "lines.linewidth": 1.5,
}


def full_information_value(theta):
    """Return v(theta) = min(theta, 1 - theta)."""
    return np.minimum(theta, 1 - theta)


def information_value(q0, q1, theta):
    """Return the value of E(q0, q1), net of the uninformed outside option."""
    outside_option = np.maximum(theta, 1 - theta)
    informed_value = (1 - theta) * q0 + theta * q1
    return np.maximum(informed_value - outside_option, 0)


def uniform_cutoff(tau):
    """Return the uniform-case cutoff and price 1 / [2(2 - tau)]."""
    return 1 / (2 * (2 - tau))


def uniform_profit(tau):
    """Return uniform-case incremental profit (1 - tau)^2 / [4(2 - tau)]."""
    return (1 - tau) ** 2 / (4 * (2 - tau))


def density(theta):
    """Irregular-example density f(theta) = 1 + 0.6 sin(4 pi theta)."""
    return 1 + 0.6 * np.sin(4 * np.pi * theta)


def cdf(theta):
    """Return the CDF obtained by integrating density from zero to theta."""
    frequency = 4 * np.pi
    return theta + 0.6 / frequency * (1 - np.cos(frequency * theta))


def phi_minus(theta, tau=0.0):
    """Return Phi_-(theta; tau) = F(theta) + (1 - tau) theta f(theta)."""
    return cdf(theta) + (1 - tau) * theta * density(theta)


def phi_plus(theta, tau=0.0):
    """Return Phi_+(theta; tau) = F(theta) - (1 - tau)(1 - theta)f(theta)."""
    return cdf(theta) - (1 - tau) * (1 - theta) * density(theta)


def integrated_phi_minus(theta):
    """Return H_-(theta) = theta F(theta), a primitive of Phi_-(theta; 0)."""
    return theta * cdf(theta)


@dataclass(frozen=True)
class IrregularSolution:
    """The ironing endpoints, allocation, and multipliers used in Figure 5."""

    a: float
    b: float
    d: float
    partial_distortion: float
    lambda_tau0: float
    cutoff_tau095: float

    @property
    def partial_q1(self) -> float:
        """Second coordinate of the partial experiment E(1, 1 + I)."""
        return 1 + self.partial_distortion


def solve_irregular_example() -> IrregularSolution:
    """Compute the common tangent and the cutoffs needed to draw Figure 5.

    The tangent to H_- has the same slope at a and b and along their chord.
    The starting guess and scalar brackets target the paper's example.
    """

    def tangent_residuals(endpoints):
        a, b = endpoints
        chord_slope = (integrated_phi_minus(b) - integrated_phi_minus(a)) / (b - a)
        return [phi_minus(a) - phi_minus(b), phi_minus(a) - chord_slope]

    solution = root(tangent_residuals, [0.22, 0.38], tol=1e-11)
    if not solution.success:
        raise RuntimeError(f"The ironing solver failed: {solution.message}")
    a, b = map(float, solution.x)
    multiplier = float(phi_minus(a))
    d = brentq(lambda x: phi_plus(x) - multiplier, 0.65, 0.9, xtol=1e-14)

    # Allocation balance determines the level on the ironing interval [a, b].
    partial_distortion = (a + d - 1) / (b - a)
    cutoff = brentq(
        lambda x: phi_minus(x, 0.95) - phi_plus(1 - x, 0.95),
        0,
        0.5,
        xtol=1e-14,
    )
    return IrregularSolution(
        a, b, float(d), float(partial_distortion), multiplier, float(cutoff)
    )



def new_figure(columns: int, height: float, **kwargs) -> tuple[Figure, np.ndarray]:
    """Create a row of panels at the shared figure width."""
    fig, axes = plt.subplots(
        1, columns, figsize=(FIGURE_WIDTH, height), squeeze=False, **kwargs
    )
    # A little extra padding lets 12 pt labels breathe between panels.
    fig.set_layout_engine("tight", pad=1.0, w_pad=1.15)
    return fig, axes[0]


def label_panel(ax: Axes, title: str, **kwargs) -> None:
    """Place a bold, centered title above a panel."""
    options = {"loc": "center", "fontsize": 12, "pad": 15}
    options.update(kwargs)
    if mpl.rcParams["text.usetex"]:
        title = "\n".join(rf"\textbf{{{line}}}" for line in title.splitlines())
    else:
        options.setdefault("fontweight", "bold")
    ax.set_title(title, **options)


def styled_legend(ax: Axes, **kwargs):
    """Apply the same soft grey legend box to every figure."""
    options = {"frameon": True, "handlelength": 1.7, "fontsize": 10}
    options.update(kwargs)
    legend = ax.legend(**options)
    legend.get_frame().set_linewidth(0)
    return legend


def add_panel_gap(
    fig: Figure,
    ax1: Axes,
    ax2: Axes,
    extra_gap: float = 0.25,
) -> None:
    """Widen a two-panel figure by extra_gap inches, preserving panel sizes.

    This optional helper can be called before saving a figure.
    """
    fig.draw_without_rendering()
    fig.set_layout_engine("none")
    old_width, height = fig.get_size_inches()
    positions = [ax.get_position().frozen() for ax in (ax1, ax2)]
    new_width = old_width + extra_gap
    fig.set_size_inches(new_width, height)
    for ax, pos, shift in zip((ax1, ax2), positions, (0, extra_gap)):
        ax.set_position([
            (pos.x0 * old_width + shift) / new_width,
            pos.y0,
            pos.width * old_width / new_width,
            pos.height,
        ])


def style_axes(ax: Axes, xlabel: str, ylabel: str, *, grid: bool = True) -> None:
    """Apply shared axis labels, outward ticks, and optional light grid lines."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="out", top=False, right=False, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, color="#E2E5E8", linewidth=0.55, linestyle="--", dashes=(2, 2))
    else:
        ax.grid(False)


def draw_allocation(ax: Axes, cutoffs, levels, colors) -> None:
    """Draw constant allocations with dotted guides at the jumps."""
    for left, right, level, color in zip(cutoffs[:-1], cutoffs[1:], levels, colors):
        ax.plot(
            [left, right], [level, level], color=color, lw=1.8, solid_capstyle="butt"
        )
    for cutoff, before, after in zip(cutoffs[1:-1], levels[:-1], levels[1:]):
        ax.plot([cutoff, cutoff], [before, after], color=GREY, lw=0.8, ls=":")
    ax.set(xlim=(0, 1), ylim=(-1.13, 1.13), yticks=[-1, 0, 1])
    style_axes(ax, r"Posterior belief $\theta$", r"Distortion $I^*$", grid=False)


# Figure 1: your revised experiment geometry and information values.


def figure_experiments() -> Figure:
    fig, (geometry, values) = new_figure(2, 3.1, width_ratios=[1, 2])

    geometry.fill([0, 1, 1], [1, 0, 1], color=GREY, alpha=0.2, zorder=0)
    geometry.plot([0, 1], [1, 1], color=BLACK, lw=2.5)
    geometry.plot([1, 1], [0, 1], color=BLACK, lw=2.5)
    geometry.scatter([0, 1], [1, 0], s=28, facecolor="white",
                edgecolor=[BLACK, BLACK], lw=1.1, zorder=4)
    geometry.annotate(r"$I=1$", (0, 1), xytext=(1, 7), textcoords="offset points",
                 color=BLACK, fontsize=10)
    geometry.annotate(r"$I=-1$", (1, 0), xytext=(-6, -2), textcoords="offset points",
                 ha="right", color=BLACK, fontsize=10)
    geometry.scatter([1], [1], s=31, color=BLACK, zorder=4)
    geometry.annotate(r"$I=0$", (1, 1), xytext=(0, 7), textcoords="offset points",
                 ha="right", fontsize=10)

    # Adding epsilon to both coordinates preserves signed distortion.
    geometry.annotate(
        "", xy=(1, 1 / 2), xytext=(4 / 5, 3 / 10),
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.7,
                    "shrinkA": 2, "shrinkB": 3},
    )
    geometry.scatter([4 / 5, 1], [3 / 10, 1 / 2], s=24, color=GREEN, zorder=5)
    geometry.annotate(r"$(4/5, 3/10)$", (0.78, 0.3), xytext=(-4, -14),
                 textcoords="offset points", ha="right", fontsize=10, color=GREEN)
    geometry.annotate(r"$E(1,1/2)$", (1.02, 0.45), xytext=(-8, 10),
                 textcoords="offset points", ha="right", fontsize=10, color=GREEN)
    geometry.text(0.56, 0.40, r"$\varepsilon=1/5$", ha="center", fontsize=10,
             color=GREEN, zorder=5)
    geometry.set_xticks([0, 0.5, 1], ["0", r"$1/2$", "1"])
    geometry.set_yticks([0, 0.5, 1], ["0", r"$1/2$", "1"])
    style_axes(geometry, r"$q_0$", r"$q_1$", grid=False)

    theta = np.unique(np.r_[np.linspace(0, 1, 1201), 0.5, 2 / 3])
    values.plot(theta, information_value(1, 1, theta), lw=1.5, color=BLUE,
             label=r"$E(1,1)$")
    values.plot(theta, information_value(1, 0.5, theta), lw=1.5, color=ORANGE,
             ls="--", label=r"$E(1,1/2)$")
   
    values.scatter([0.5, 2 / 3], [0.25, 0], s=22, color=ORANGE, zorder=4)
    values.plot([0.5, 0.5], [0, 0.25], color=GREY, ls=":", lw=0.8)
    values.plot([0, 0.5], [0.25, 0.25], color=GREY, ls=":", lw=0.8)
    values.plot([0, 0.5], [0.5, 0.5], color=GREY, ls=":", lw=0.8)
    styled_legend(values, loc="upper right", fontsize=12)
    values.set(xlim=(0, 1), ylim=(-0.015, 0.58))
    values.set_xticks([0, 0.5, 2 / 3, 1], ["0", r"$1/2$", r"$2/3$", "1"])
    values.set_yticks([0, 0.125, 0.25, 0.5], ["0", r"$1/8$", r"$1/4$", r"$1/2$"])
    style_axes(values, r"Posterior belief $\theta$", r"Buyer value $V(E,\theta)$", grid=False)
    label_panel(geometry, "(a) Geometry")
    label_panel(values, "(b) Value")
    # add_panel_gap(fig, geometry, values, extra_gap=0.6)
    return fig


def figure_two_types() -> Figure:
    """Draw required rents and the two parameter-region boundaries."""
    fig, (rents, same_default, opposite_defaults) = new_figure(3, 2.65)
   
    alpha = np.linspace(0, 1, 501)
    rents.plot(alpha, 0.2 * alpha, color=BLUE, label="Same")
    rents.plot(
        alpha,
        np.maximum(0.4 * alpha - 0.2, 0),
        color=ORANGE,
        ls="--",
        label="Opposite",
    )
    rents.axvline(0.5, color="#B5B5B5", ls=":", lw=0.8)
    rents.text(0.59, 0.017, r"$\bar\alpha=1/2$", fontsize=10, color=ORANGE)
    rents.set(
        xlim=(0, 1), ylim=(-0.008, 0.26), xticks=[0, 0.5, 1], yticks=[0, 0.1, 0.2]
    )
    styled_legend(rents,
        loc="upper left",
        borderaxespad=0.15,
        handlelength=1.7,
        handletextpad=0.5,
        labelspacing=0.4,
        fontsize=10,
    )
    style_axes(rents, "Information frac. " + r"$\alpha$", r"Required rent $r_h$", grid=False)

    rents.set_xticks([0, 0.5, 1], ["0", r"$1/2$", "1"])
    rents.set_yticks([0, 0.1, 0.2], ["0", r"$1/10$", r"$1/5$"])

    tau = np.linspace(0, 1, 1001)
    for ax, denominator, upper_region in (
        (same_default, 2, "Exclusion"),
        (opposite_defaults, 3, "Separation"),
    ):
        boundary = (1 - tau) / (denominator - tau)
        
        upper_color = GREY if denominator == 2 else ORANGE
        boundary_color = BLUE if denominator == 2 else ORANGE
        switch = 2 / 3 if denominator == 2 else 1 / 3
        upper_allocation = r"$\alpha^*=0$" if denominator == 2 else r"$\alpha^*=1/2$"
        ax.fill_between(tau, 0, boundary, color=BLUE, alpha=0.13, zorder=0)
        ax.fill_between(tau, boundary, 1, color=upper_color, alpha=0.11, zorder=0)
        ax.plot(tau, boundary, color=boundary_color, lw=1.8)
        ax.axhline(0.25, color="#797F85", ls=":", lw=0.8)
        ax.plot([switch, switch], [0, 0.25], color="#797F85", ls=":", lw=0.8)
        ax.scatter([switch], [0.25], s=16, color=BLACK, zorder=4)
        ax.text(
            0.56,
            0.73,
            upper_region + "\n" + upper_allocation,
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.5,
        )
        ax.text(
            0.23 if denominator == 2 else 0.17,
            0.16 if denominator == 2 else 0.12,
            "Pooling\n" + r"$\alpha^*=1$",
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.5,
            color=BLUE,
        )
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_yticks([0, 0.25, 0.5, 1], ["0", r"$1/4$", r"$1/2$", "1"])
        ax.set_xticks(
            [0, switch, 1], ["0", r"$2/3$" if denominator == 2 else r"$1/3$", "1"]
        )
        # ylabel = r"High-type share $\gamma$" if denominator == 2 else ""
        ylabel = r"High-type share $\gamma$"
        style_axes(ax, r"Competition $\tau$", ylabel, grid=False)

    label_panel(rents, "(a) Rent")
    label_panel(same_default, "(b) Same")
    label_panel(opposite_defaults, "(c) Opposite")
    return fig


def figure_posted_price() -> Figure:
    """Draw a balanced allocation and its posted-price implementation."""
    fig, (allocation, utility) = new_figure(2, 3.35)
    price = 1 / 3

    allocation.fill_between([0, price], -1, 0, color=ORANGE, alpha=0.11, zorder=0)
    allocation.fill_between([1 - price, 1], 0, 1, color=ORANGE, alpha=0.11, zorder=0)
    allocation.axhline(0, color=GREY, lw=0.65, zorder=0)
    draw_allocation(
        allocation, [0, price, 1 - price, 1], [-1, 0, 1], [GREY, BLUE, GREY]
    )
    allocation.text(price / 2, -0.43, r"$-c$", ha="center", fontsize=11, color=ORANGE)
    allocation.text(
        1 - price / 2, 0.43, r"$+c$", ha="center", fontsize=11, color=ORANGE
    )
    allocation.text(
        price / 2, -0.82, "Default 0", ha="center", fontsize=10, color="#676D73"
    )
    allocation.text(
        1 - price / 2, 0.82, "Default 1", ha="center", fontsize=10, color="#676D73"
    )
    allocation.text(0.5, 0.16, "Full information", ha="center", fontsize=10, color=BLUE)
    allocation.set_xticks(
        [0, price, 1 - price, 1], ["0", r"$1/3$", r"$2/3$", "1"]
    )
    allocation.set_ylabel(r"Signed distortion $I(\theta)$")

    theta = np.unique(np.r_[np.linspace(0, 1, 1201), price, 0.5, 1 - price])
    value = full_information_value(theta)
    rent = np.maximum(value - price, 0)
    utility.axvspan(price, 1 - price, color=BLUE, alpha=0.065, zorder=0)
    utility.plot(theta, value, color=BLUE, label=r"Value $v(\theta)$")
    utility.axhline(price, color=ORANGE, ls="--", lw=1.8, label=r"Price $c=1/3$")
    utility.plot(theta, rent, color=PURPLE, ls="-.", label=r"Rent $U(\theta)$")
    utility.text(0.50, 0.22, "Purchase", ha="center", fontsize=10, color=BLUE)
    styled_legend(utility,
        loc="upper right",
        fontsize=10,
        handlelength=1.6,
        borderaxespad=0.15,
        labelspacing=0.3,
    )
    utility.set(xlim=(0, 1), ylim=(-0.015, 0.72))
    utility.set_xticks([0, price, 1 - price, 1], ["0", r"$1/3$", r"$2/3$", "1"])
    utility.set_yticks([0, 1 / 6, 1 / 3, 1 / 2], ["0", r"$1/6$", r"$1/3$", r"$1/2$"])
    style_axes(utility, r"Posterior belief $\theta$", "Value, price and rent", grid=False)
    label_panel(allocation, "(a) Balanced allocation")
    label_panel(utility, "(b) Value, price and rent")
    return fig


def figure_uniform_competition() -> Figure:
    """Draw the served beliefs and incremental profit under uniform types."""
    fig, (allocation, profit) = new_figure(2, 3.1)
    tau = np.linspace(0, 1, 1001)
    cutoff = uniform_cutoff(tau)

    allocation.fill_between(tau, cutoff, 1 - cutoff, color=BLUE, alpha=0.13, zorder=0)
    allocation.plot(tau, cutoff, color=BLUE)
    allocation.plot(tau, 1 - cutoff, color=BLUE)
    allocation.plot([0.5, 0.5], [1 / 3, 2 / 3], color=BLUE, lw=1, ls=":")
    allocation.scatter([0.5, 0.5], [1 / 3, 2 / 3], s=19, color=BLUE, zorder=4)
    allocation.scatter(
        [1],
        [0.5],
        s=24,
        facecolor="white",
        edgecolor=BLUE,
        lw=1,
        zorder=4,
        clip_on=False,
    )
    allocation.text(
        0.30, 0.5, "Full information", ha="center", va="center", fontsize=10, color=BLUE
    )
    allocation.text(
        0.40, 0.085, "No information", ha="center", fontsize=10, color="#60656A"
    )
    allocation.text(
        0.40, 0.90, "No information", ha="center", fontsize=10, color="#60656A"
    )
    allocation.text(
        0.40, 0.19, r"Cutoff = price $t^*(\tau)$", ha="center", fontsize=10, color=BLUE
    )
    allocation.text(0.40, 0.77, r"$1-t^*(\tau)$", ha="center", fontsize=10, color=BLUE)
    allocation.set(
        xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.25, 0.5, 0.75, 1]
    )
    style_axes(
        allocation, r"Competition $\tau$", r"Posterior belief $\theta$", grid=False
    )

    allocation.set_xticks([0, 0.5, 1], ["0", r"$1/2$", "1"])
    allocation.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", r"$1/4$", r"$1/2$", r"$3/4$", "1"])

    profit.plot(tau, uniform_profit(tau), color=BLUE, lw=1.5)
    profit.scatter([0, 0.5], [1 / 8, 1 / 24], s=20, color=BLUE, zorder=4, clip_on=False)
    profit.plot([0, 0.5, 0.5], [1 / 24, 1 / 24, 0], color=GREY, ls=":", lw=0.8)
    profit.scatter(
        [1], [0], s=24, facecolor="white", edgecolor=BLUE, lw=1, zorder=4, clip_on=False
    )
    profit.set(xlim=(0, 1), ylim=(-0.004, 0.132), xticks=[0, 0.25, 0.5, 0.75, 1])
    profit.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", r"$1/4$", r"$1/2$", r"$3/4$", "1"])
    profit.set_yticks([0, 1 / 24, 1 / 8], ["0", r"$1/24$", r"$1/8$"])
    style_axes(profit, r"Competition $\tau$", r"Incremental profit $\Pi^*$", grid=False)
    label_panel(allocation, "(a) Served beliefs")
    label_panel(profit, "(b) Incremental profit")
    return fig


def figure_ironing(solution: IrregularSolution) -> Figure:
    """Draw raw and ironed virtual values and allocations at two tau values."""
    fig, (ironing, allocation_zero, allocation_high) = new_figure(3, 2.65)
    a, b, d = solution.a, solution.b, solution.d
    multiplier = solution.lambda_tau0

    theta = np.linspace(0.12, 0.46, 1501)
    raw = phi_minus(theta)
    ironed = np.where((theta >= a) & (theta <= b), multiplier, raw)
    ironing.axhline(multiplier, color="#AAAAAA", ls=":", lw=0.8)
    ironing.plot(theta, raw, color=GREY, ls="--", lw=1.6, label="Raw")
    ironing.plot(theta, ironed, color=PURPLE, lw=1.5, label="Ironed")
    ironing.text(0.128, multiplier + 0.018, r"$\lambda^*$", fontsize=10, color="#777777")
    for cutoff, label in ((a, "a"), (b, "b")):
        ironing.plot([cutoff, cutoff], [0.34, multiplier], color=PURPLE, ls=":", lw=0.8)
        ironing.text(cutoff + 0.008, 0.37, f"${label}$", fontsize=10, color=PURPLE)
    styled_legend(ironing,
        loc="upper left",
        handlelength=1.25,
        handletextpad=0.45,
        labelspacing=0.3,
        borderaxespad=0.35,
        fontsize=10,
    )
    ironing.set(
        xlim=(0.12, 0.46),
        ylim=(0.34, 0.84),
        xticks=[0.2, 0.3, 0.4],
        yticks=[0.4, 0.6, 0.8],
    )
    style_axes(ironing, r"Posterior belief $\theta$", r"Virtual value $\Phi_-$", grid=False)

    allocation_zero.axvspan(a, b, color=ORANGE, alpha=0.08, zorder=0)
    draw_allocation(
        allocation_zero,
        [0, a, b, d, 1],
        [-1, solution.partial_distortion, 0, 1],
        [GREY, ORANGE, BLUE, GREY],
    )
    allocation_zero.annotate(
        rf"$E(1,{solution.partial_q1:.3f})$",
        xy=((a + b) / 2, solution.partial_distortion),
        xytext=(0.65, -0.75),
        ha="center",
        fontsize=10,
        color=ORANGE,
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": ORANGE},
    )
    allocation_zero.set_xticks([0, 0.5, 1], ["0", r"$1/2$", "1"])
    allocation_zero.text(
        0.58, 0.12, r"$I=0$", ha="center", fontsize=10, color=BLUE
    )

    cutoff = solution.cutoff_tau095
    allocation_high.axvspan(cutoff, 1 - cutoff, color=BLUE, alpha=0.10, zorder=0)
    draw_allocation(
        allocation_high, [0, cutoff, 1 - cutoff, 1], [-1, 0, 1], [GREY, BLUE, GREY]
    )
    allocation_high.set_xticks([0, 0.5, 1], ["0", r"$1/2$", "1"])
    allocation_high.set_ylabel("")
    for ax in (ironing, allocation_zero, allocation_high):
        ax.set_xlabel("Posterior belief " + r"$\theta$")
    label_panel(ironing, "(a) Virtual values")
    label_panel(allocation_zero, r"(b) $\tau=0$")
    label_panel(allocation_high, r"(c) $\tau=0.95$")
    return fig




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-tex", action="store_true",
        help="Use Matplotlib text and Computer Modern math without a TeX installation.",
    )
    args = parser.parse_args()
    style = dict(STYLE)
    if args.no_tex:
        style["text.usetex"] = False

    directory = Path(__file__).resolve().parent / "figures"
    directory.mkdir(exist_ok=True)
    solution = solve_irregular_example()
    figures = (
        ("fig01_experiments.pdf", figure_experiments),
        ("fig02_two_types.pdf", figure_two_types),
        ("fig03_posted_price.pdf", figure_posted_price),
        ("fig04_uniform_competition.pdf", figure_uniform_competition),
        ("fig05_ironing.pdf", lambda: figure_ironing(solution)),
    )
    with mpl.rc_context(style):
        for filename, draw_figure in figures:
            fig = draw_figure()
            try:
                fig.savefig(directory / filename)
            finally:
                plt.close(fig)
            print(f"Saved {directory / filename}")


if __name__ == "__main__":
    main()