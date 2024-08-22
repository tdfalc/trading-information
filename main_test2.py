if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    from tfds.log import create_logger
    from tfds.plotting import prettify, use_tex

    use_tex()

    from trading_information.distributions import BetaMixture, Uniform
    from trading_information.virtual_values import VirtualValues
    from trading_information.mechanism import Mechanism

    from scipy.integrate import quad
    from scipy.interpolate import interp1d

    tau = 1
    prob_state0 = 1
    tj = prob_state0
    alpha = tau - tj * (2 * tau)

    print("alpha", alpha)

    dist = Uniform(0, 1)
    from scipy import stats

    # dist = stats.norm(0, 1)
    # dist = BetaMixture((20, 60), (30, 30), (0.99, 0.01))

    num_types = 1000
    types = np.linspace(0, 1, num_types)

    fig, ax = plt.subplots()
    ax.plot(types, dist.pdf(types))
    fig.savefig("./dist.pdf")

    q25 = int(num_types / 4)
    q50 = int(num_types / 2)
    q75 = int(3 * num_types / 4)

    allocations_full = np.zeros(num_types)
    allocations_full[:q25] = -1
    allocations_full[q25:q75] = 0
    allocations_full[q75:] = 1

    # allocations_full = np.zeros(num_types)
    # allocations_full[:150] = -1
    # allocations_full[150:300] = -1 / 3
    # allocations_full[300:600] = 0
    # allocations_full[600:] = 0.2

    if prob_state0 >= 0.5:

        allocation_half = np.zeros(num_types)
        allocation_half[:q50] = -0.5
        allocation_half[q50:q75] = 0
        allocation_half[q75:] = 1

        allocations_info = np.zeros(num_types)
        allocations_info[:q75] = -1 / 3
        allocations_info[q75:] = 1
        # allocations_info[:q75] = -1
        # allocations_info[q25:q50] = -1 / 2
        # allocations_info[q50:q75] = -1 / 3
        # allocations_info[q75:] = 1

    else:
        allocation_half = np.zeros(num_types)
        allocation_half[:q25] = -1
        allocation_half[q25:q50] = 0
        allocation_half[q50:] = 0.5

        allocations_info = np.zeros(num_types)
        allocations_info[:q25] = -1
        allocations_info[q25:] = 1 / 3

    q_full = interp1d(types, allocations_full)
    q_half = interp1d(types, allocation_half)
    q_info = interp1d(types, allocations_info)

    pdf = dist.pdf
    cdf = dist.cdf

    fig, ax = plt.subplots(figsize=(6, 4))
    kwargs = dict(marker="o", markevery=10, markerfacecolor="white", ls="dashed", lw=0.5)
    ax.plot(
        types,
        q_full(types),
        label="Full Info",
        color="blue",
        markeredgecolor="blue",
        **kwargs,
    )
    ax.plot(
        types,
        q_half(types),
        label="Half Info",
        color="darkorange",
        markeredgecolor="darkorange",
        **kwargs,
    )
    ax.plot(
        types,
        q_info(types),
        label="No Transfer",
        color="green",
        markeredgecolor="green",
        **kwargs,
    )
    ax.set_xlabel("Private Type ($t_i$)")
    ax.set_ylabel("Allocation ($\\xi_j$)")
    ax.legend()
    prettify(ax=ax, legend_loc="upper left")
    fig.savefig("./allocations_test.png", dpi=300)

    def transfer_for_type(type):

        transfer = q(type) * (type * pdf(type) + cdf(type)) + pdf(type) * np.minimum(0, -q(type))
        return transfer

    def externality_for_type(type):

        ps1 = 1 - type - type * q(type) + (1 - 2 * type) * np.minimum(0, -q(type))

        externality = tj + (1 - 2 * tj) * ps1

        externality *= tau
        externality *= pdf(type)

        # type, ps1, externality)

        return externality

    def utility_for_type(type):
        utility = 1
        return utility

    def objective():
        exp_transfer = quad(transfer_for_type, 0, 1)[0]
        exp_ex = quad(externality_for_type, 0, 1)[0]
        print("results", exp_transfer, exp_ex)
        print("profit", 1 - exp_ex)
        print("profit overall", exp_transfer + 1 - exp_ex)
        return exp_transfer - exp_ex

    fig, ax = plt.subplots()

    print(" ")

    q = q_full
    print("full information", objective())
    ax.plot(types[:], externality_for_type(types)[:])
    print("mu", np.mean(externality_for_type(types)[:]))
    print("avg all", np.mean(q(types)))

    print(" ")

    q = q_half
    print("half information", objective())
    ax.plot(types[:], externality_for_type(types)[:])
    print("mu", np.mean(externality_for_type(types)[:]))
    print("avg all", np.mean(q(types)))

    print(" ")

    q = q_info
    print("no transfer", objective())
    ax.plot(types[:], externality_for_type(types)[:])
    print("mu", np.mean(externality_for_type(types)[:]))
    print("avg all", np.mean(q(types)))

    fig.savefig("./ex.png", dpi=300)

    def _allocations_to_transfers(allocation):
        return types * allocation + np.minimum(-allocation, 0) - np.cumsum(allocation) / num_types

    fig, ax = plt.subplots(figsize=(6, 4))
    kwargs = dict(marker="o", markevery=10, markerfacecolor="white", ls="dashed", lw=0.5)
    q = q_full(types)
    q = q_full
    ts = _allocations_to_transfers(q(types))
    ax.plot(
        types,
        # _allocations_to_transfers(q),
        ts,
        label="Full Info",
        color="blue",
        markeredgecolor="blue",
        **kwargs,
    )
    q = q_half(types)
    q = q_half
    ts = _allocations_to_transfers(q(types))
    ax.plot(
        types,
        # _allocations_to_transfers(q),
        ts,
        label="Half Info",
        color="darkorange",
        markeredgecolor="darkorange",
        **kwargs,
    )
    q = q_info
    ts = _allocations_to_transfers(q(types))
    ax.plot(
        types,
        ts,
        label="No Transfer",
        color="green",
        markeredgecolor="green",
        **kwargs,
    )
    ax.set_xlabel("Private Type ($t_i$)")
    ax.set_ylabel("Transfer ($\pi_j$)")
    ax.legend()
    prettify(ax=ax, legend_loc="upper left")
    fig.savefig("./transfers.png", dpi=300)

    ## new experiment

    def profit():
        exp_transfer = quad(transfer_for_type, 0, 1)[0]
        exp_ex = quad(externality_for_type, 0, 1)[0]

        return exp_transfer + 1 - exp_ex

    def externality():
        exp_ex = quad(externality_for_type, 0, 1)[0]

        return exp_ex

    def transfer():
        exp_transfer = quad(transfer_for_type, 0, 1)[0]

        return exp_transfer

    taus = np.linspace(0, 3, 50)
    profits1 = np.zeros(len(taus))
    externalities1 = np.zeros(len(taus))
    profits2 = np.zeros(len(taus))
    externalities2 = np.zeros(len(taus))
    from tqdm import tqdm

    transfers1 = np.zeros(len(taus))
    transfers2 = np.zeros(len(taus))

    for i, tau in tqdm(enumerate(taus)):
        # tau = 1
        prob_state0 = 1
        tj = prob_state0
        alpha = tau - tj * (2 * tau)

        q = q_info
        profits1[i] = profit()
        externalities1[i] = externality()
        transfers1[i] = transfer()

        q = q_full
        profits2[i] = profit()
        externalities2[i] = externality()
        transfers2[i] = transfer()

    # ax.plot(taus, profits)
    fig, ax = plt.subplots(dpi=300)
    ax.plot(taus, profits1)
    ax.plot(taus, profits2)
    prettify(ax=ax)
    fig.savefig("profits.pdf")

    fig, ax = plt.subplots(dpi=300)
    ax.plot(taus, externalities1)
    ax.plot(taus, externalities2)
    ax.plot(taus, tau / 2)
    prettify(ax=ax)
    fig.savefig("ex.pdf")

    fig, ax = plt.subplots(dpi=300)
    ax.plot(taus, transfers1)
    ax.plot(taus, transfers2)
    prettify(ax=ax)
    fig.savefig("transfers.pdf")
