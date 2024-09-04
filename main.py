# I found a case study where its possble to distory only the bottom half!!!

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    from tfds.log import create_logger
    from tfds.plotting import prettify, use_tex

    use_tex()

    from trading_information.distributions import BetaMixture, Uniform
    from trading_information.virtual_values import VirtualValues
    from trading_information.mechanism import Mechanism
    from trading_information.hyperopt import HyperOpt

    logger = create_logger(__name__)

    logger.info("Testing")

    from scipy import stats

    # dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))
    dist = Uniform(0, 1)
    dist = stats.norm(0, 0.3)
    # dist = BetaMixture((20,), (20,), (1,))
    # dist = BetaMixture([2], [4], [1])

    # dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))
    # from scipy import stats

    # # dist = stats.lognorm(s=0.3)
    # # dist = BetaMixture((1, 40, 200), (50, 50, 50), (0.3, 0.3, 0.4))
    # xs = np.linspace(0, 1, 1000)
    # ys = dist.pdf(xs) * 2.5

    # fig, ax = plt.subplots()
    # ax.plot(xs, dist.pdf(xs))
    # fig.savefig("./dist.pdf")

    mechanism = Mechanism(num_types=1000, dist=dist)
    types = mechanism.types

    tau = 4

    tau0 = tau1 = tau
    prob_state0 = 1
    tj = prob_state0

    threshold_types = np.linspace(0.1, 0.9, 30)
    hyperopt = HyperOpt(mechanism, threshold_types=threshold_types)
    hyperopt.run(tau, prob_state0, desc=f"Hyperopt")
    threshold_type = hyperopt.best()
    print("threshold_type", threshold_type)

    print(hyperopt.objectives)
    # threshold_type = 0.75

    alpha = tau1 - tj * (tau0 + tau1)

    # print(alpha)

    output = mechanism.solve(tau, prob_state0, threshold_type, return_dict=True)
    allocations = output["allocations"]
    transfers1 = output["transfers"]
    externalities1 = output["externalities"]

    # These do not add up for non uniform distribution
    # No worries, now it does, i forogt to multuply by pdf.
    print("--")
    print(output["avg_transfer"], output["avg_externality"])
    print(np.mean(transfers1[1:]), np.mean(externalities1[1:]))
    N = 1000
    print(
        np.sum(transfers1 * dist.pdf(types), axis=0) / N,
        np.sum(externalities1 * dist.pdf(types), axis=0) / N,
    )
    print("--")

    # allocations75, transfers, multiplier, obj = mechanism.solve(tau, prob_state0, 0.75)
    # print("OBJECTIVE 0.75", obj)

    threshold_types = np.linspace(0.1, 0.9, 50)
    from tqdm import tqdm

    # virtual_values = VirtualValues(dist, types, alpha=alpha, iron=True)
    # pos_iron = virtual_values.positive
    # neg_iron = virtual_values.negative

    # taus = [0, 1, 2, 3, 4, 50, 100]
    # for tau in taus:
    #     print("TAU", tau)

    #     hyperopt = HyperOpt(mechanism, threshold_types, verbose=10)
    #     hyperopt.run(tau, prob_state0)

    #     fig, ax = plt.subplots()
    #     ax.plot(threshold_types, hyperopt.objectives)
    #     ax.axvline(x=0.25)
    #     ax.axvline(x=0.75)
    #     fig.savefig(f"./objectives_{tau}.pdf")

    # # allocations = np.zeros(len(types))
    # # allocations = np.ones(len(types))
    # # allocations[: int(len(types) / 2)] = -1
    # # transfers = mechanism._allocations_to_transfers(allocations)

    # virtual_values = VirtualValues(dist, types, alpha=alpha, iron=False)
    # pos = virtual_values.positive
    # neg = virtual_values.negative

    # virtual_values = VirtualValues(dist, types, alpha=alpha, iron=True)

    # pos_iron = virtual_values.positive
    # neg_iron = virtual_values.negative

    # # pos_iron[500:] = np.linspace(-200000, 100000, 500)

    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.plot(types, neg, color="blue", label="Neg")
    # ax.plot(types, neg_iron, color="blue", ls="dashed")
    # ax.plot(types, pos, color="darkorange", label="Pos")
    # ax.plot(types, pos_iron, color="darkorange", ls="dashed")
    # ax.legend()
    # ax.axhline(y=0, c="k")
    # ax.ticklabel_format(useOffset=False)

    # prettify(ax=ax)
    # ax.set_xlabel("Virtual Values")
    # fig.tight_layout()
    # fig.savefig("./virtuals.pdf", dpi=300)

    # # transfers = mechanism._allocations_to_transfers(allocations)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(types, dist.pdf(types))
    ax.set_ylabel(r"Density")
    ax.set_xlabel("Private Type ($t_i$)")
    prettify(ax=ax)
    fig.savefig("./densities.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(types, allocations)
    # ax.plot(types, externalities1 - transfers1)
    prettify(ax=ax)
    ax.set_xlabel("Allocations")
    fig.tight_layout()
    fig.savefig("./allocations.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(types, externalities1)
    prettify(ax=ax)
    ax.set_xlabel("externalities")
    fig.tight_layout()
    fig.savefig("./revenues.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(types, transfers1)
    prettify(ax=ax)
    ax.set_xlabel("transfers")
    fig.tight_layout()
    fig.savefig("./transfers.pdf", dpi=300)

    # def vv(q, i):
    #     v = neg_iron[i] if q <= 0 else pos_iron[i]
    #     # if v < 0:
    #     #     v = 0

    #     return v * q

    # fig, ax = plt.subplots(figsize=(5, 3))

    # objs50 = [vv(q, i) for i, q in enumerate(allocations50)]
    # objs75 = [vv(q, i) for i, q in enumerate(allocations75)]
    # print("50", np.mean(objs50))
    # print("75", np.mean(objs75))

    # print(np.mean(objs50[:500]))
    # print(np.mean(objs75[:750]))

    # ax.plot(types, objs50, label=0.5)
    # ax.plot(types, objs75, label=0.75)
    # prettify(ax=ax)
    # fig.tight_layout()
    # fig.savefig("./objs.pdf", dpi=300)

    # tau = 0

    # tau0 = tau1 = tau
    # prob_state0 = 1
    # tj = prob_state0
    # threshold_type = 0.75

    # alpha = tau1 - tj * (tau0 + tau1)

    theta = 0.001
    alls = np.linspace(-1, 1, 100)  # why is this externality not linear?

    #  because of this, externality cost is not convex
    # it is convex for some types but concave for others, so we cannot use toikka!!

    def externality_for_type(type, x):
        ps1 = 1 - prob_state0 - prob_state0 * x + (1 - 2 * prob_state0) * np.minimum(0, -x)
        externality = ps1
        externality *= 1 - 2 * prob_state0
        # externality *= prob_state0
        externality *= tau

        return externality + tau * prob_state0

    exs = externality_for_type(theta, alls)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(alls, exs)
    prettify(ax=ax)
    ax.set_xlabel("Allocations")
    fig.tight_layout()
    fig.savefig("./check.png", dpi=200)

    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.plot(types, allocations)
    # ax.plot(types, transfers)
    # prettify(ax=ax)
    # ax.set_xlabel("Allocations")
    # fig.tight_layout()
    # fig.savefig("./allocations.pdf", dpi=300)

    exp_transfers, exp_externality = 0, 0
    externalities = np.zeros(len(allocations))
    transfers = np.zeros(len(allocations))

    from tqdm import tqdm

    for i, type in tqdm(enumerate(mechanism.types)):
        if i > 0:
            step = mechanism.types[1] - mechanism.types[0]
            transfer = (
                allocations[i] * (mechanism.types[i] * mechanism._pdfs[i] + mechanism._cdfs[i])
                + np.minimum(-allocations[i], 0) * mechanism._pdfs[i]
            )
            transfers[i] = transfer
            exp_transfers += transfer * step

            alpha = tau1 - tj * tau0 - tj * tau1

            externality = tau0 * tj + alpha * (
                1 - tj - tj * allocations[i] + (1 - 2 * tj) * np.minimum(-allocations[i], 0)
            )
            externalities[i] = externality
            exp_externality += step * externality * mechanism._pdfs[i]

    exp_transfers, exp_externality

    print(exp_transfers, exp_externality, exp_transfers - exp_externality)

    fig, ax = plt.subplots()
    ax.plot(externalities)
    ax.plot(externalities1)
    fig.savefig("./externalities1")

    fig, ax = plt.subplots()
    ax.plot(transfers)
    ax.plot(transfers1)
    fig.savefig("./transfers1")

    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.plot(types, externalities)
    # ax.axhline(y=np.mean(externalities))
    # prettify(ax=ax)
    # ax.set_xlabel("externalities")
    # fig.tight_layout()
    # fig.savefig("./externalities.pdf", dpi=300)

    # def calc_value_before(type: float) -> float:
    #     # return type * (type > 0.5)
    #     return np.maximum(type, 1 - type)

    # def calc_value_after(q: float, type: float) -> float:
    #     return type * q + 1 + np.minimum(-q, 0)

    # def calculate_gain(q: float, type: float) -> float:
    #     return np.maximum(0, calc_value_after(q, type) - calc_value_before(type))

    # gains = np.zeros(len(types))
    # values_before = np.zeros(len(types))
    # values_after = np.zeros(len(types))
    # for i, type in tqdm(enumerate(mechanism.types)):
    #     allocation = allocations[i]
    #     gains[i] = calculate_gain(allocation, type)
    #     values_after[i] = calc_value_after(allocation, type)
    #     values_before[i] = calc_value_before(type)

    # def _allocations_to_transfers(allocation):
    #     ts = types * allocation
    #     ts += np.minimum(-allocation, 0)
    #     ts -= np.cumsum(allocation) / len(types)
    #     return ts

    # transfers = _allocations_to_transfers(allocations)

    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.plot(types, gains, label="Gain")

    # ax.plot(types, transfers, label="Transfers")
    # print(np.mean(transfers), np.mean(gains))
    # # ax.plot(types, values_before, label="Before")
    # # ax.plot(types, values_after, label="After")
    # ax.legend()
    # prettify(ax=ax)
    # ax.set_xlabel("gains")
    # fig.tight_layout()
    # fig.savefig("./gains.pdf", dpi=300)
