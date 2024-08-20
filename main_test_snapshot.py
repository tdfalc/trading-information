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

    tau = 5
    prob_state0 = 1
    tj = prob_state0
    alpha = tau - tj * (2 * tau)

    print("alpha", alpha)

    dist = Uniform(0.5, 1)
    dist = BetaMixture((20, 60), (30, 30), (0.99, 0.01))

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

    if prob_state0 == 1:

        allocation_half = np.zeros(num_types)
        allocation_half[:q50] = -0.5
        allocation_half[q50:q75] = 0
        allocation_half[q75:] = 1

        allocations_info = np.zeros(num_types)
        allocations_info[:q75] = -1 / 3
        allocations_info[q75:] = 1

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
        label="(1) $\\tau=0$",
        color="blue",
        markeredgecolor="blue",
        **kwargs,
    )
    ax.plot(
        types,
        q_half(types),
        label="(2) $\\tau=1$",
        color="darkorange",
        markeredgecolor="darkorange",
        **kwargs,
    )
    ax.plot(
        types,
        q_info(types),
        label="(2) $\\tau=3$",
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

        # def _transfer(type):
        #     tr = type * q(type) + np.minimum(0, -q(type))
        #     tr -= quad(q, 0, type)[0]
        #     tr *= pdf(type)
        #     return tr

        # transfer = _transfer(type)

        transfer = q(type) * (type * pdf(type) + cdf(type)) + pdf(type) * np.minimum(0, -q(type))
        return transfer

    def externality_for_type(type):

        ps1 = 1 - type - type * q(type) + (1 - 2 * type) * np.minimum(0, -q(type))

        externality = tj + (1 - 2 * tj) * ps1

        externality *= tau
        externality *= pdf(type)

        # type, ps1, externality)

        return externality

    def objective():
        exp_transfer = quad(transfer_for_type, 0, 1)[0]
        exp_ex = quad(externality_for_type, 0, 1)[0]
        print("results", exp_transfer, exp_ex)
        return exp_transfer - exp_ex

    fig, ax = plt.subplots()

    print(" ")

    q = q_full
    print("full information", objective())
    ax.plot(types[:], externality_for_type(types)[:])
    print("mu", np.mean(externality_for_type(types)[:q50]))

    print(" ")

    q = q_half
    print("half information", objective())
    ax.plot(types[:], externality_for_type(types)[:])
    print("mu", np.mean(externality_for_type(types)[:q50]))

    print(" ")

    q = q_info
    print("no transfer", objective())
    ax.plot(types[:], externality_for_type(types)[:])
    print("mu", np.mean(externality_for_type(types)[:q50]))

    fig.savefig("./ex.png", dpi=300)

    def _allocations_to_transfers(allocation):
        return types * allocation + np.minimum(-allocation, 0) - np.cumsum(allocation) / num_types

    fig, ax = plt.subplots(figsize=(6, 4))
    kwargs = dict(marker="o", markevery=10, markerfacecolor="white", ls="dashed", lw=0.5)
    q = q_full(types)
    ax.plot(
        types,
        _allocations_to_transfers(q),
        label="(1) $\\tau=0$",
        color="blue",
        markeredgecolor="blue",
        **kwargs,
    )
    q = q_half(types)
    ax.plot(
        types,
        _allocations_to_transfers(q),
        label="(2) $\\tau=1$",
        color="darkorange",
        markeredgecolor="darkorange",
        **kwargs,
    )
    q = q_info(types)
    ax.plot(
        types,
        _allocations_to_transfers(q),
        label="(2) $\\tau=3$",
        color="green",
        markeredgecolor="green",
        **kwargs,
    )
    ax.set_xlabel("Private Type ($t_i$)")
    ax.set_ylabel("Transfer ($\pi_j$)")
    ax.legend()
    prettify(ax=ax, legend_loc="upper left")
    fig.savefig("./transfers.png", dpi=300)
