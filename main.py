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

    logger = create_logger(__name__)

    logger.info("Testing")

    # dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))
    dist = Uniform(0, 1)
    dist = BetaMixture((20, 60), (30, 30), (0.99, 0.01))
    xs = np.linspace(0, 1, 1000)
    ys = dist.pdf(xs) * 2.5

    mechanism = Mechanism(num_types=1000, dist=dist)

    tau = 5

    tau0 = tau1 = tau
    prob_state0 = 1
    tj = prob_state0
    threshold_type = 0.25

    alpha = tau1 - tj * (tau0 + tau1)

    print(alpha)

    allocations, transfers, multiplier = mechanism.solve(alpha, threshold_type)

    types = mechanism.types

    # virtual_values = VirtualValues(dist, types, alpha=alpha, iron=False)
    # pos = virtual_values.positive
    # neg = virtual_values.negative

    # virtual_values = VirtualValues(dist, types, alpha=alpha, iron=True)
    # pos_iron = virtual_values.positive
    # neg_iron = virtual_values.negative

    # fig, ax = plt.subplots(figsize=(5, 3))
    # # ax.plot(types, neg, color="blue", label="Neg")
    # ax.plot(types, neg_iron, color="blue", ls="dashed")
    # # ax.plot(types, pos, color="darkorange", label="Pos")
    # ax.plot(types, pos_iron, color="darkorange", ls="dashed")
    # ax.legend()
    # ax.axhline(y=0, c="k")
    # ax.ticklabel_format(useOffset=False)

    # prettify(ax=ax)
    # ax.set_xlabel("Virtual Values")
    # fig.tight_layout()
    # fig.savefig("./virtuals.pdf", dpi=300)

    transfers = mechanism._allocations_to_transfers(allocations)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(types, allocations)
    ax.plot(types, transfers)
    prettify(ax=ax)
    ax.set_xlabel("Allocations")
    fig.tight_layout()
    fig.savefig("./allocations.pdf", dpi=300)

    exp_transfers, exp_externality = 0, 0
    externalities = np.zeros(len(allocations))

    from tqdm import tqdm

    for i, type in tqdm(enumerate(mechanism.types)):
        if i > 0:
            step = mechanism.types[1] - mechanism.types[0]
            exp_transfers += (
                allocations[i] * (mechanism.types[i] * mechanism._pdfs[i] + mechanism._cdfs[i])
                + np.minimum(-allocations[i], 0) * mechanism._pdfs[i]
            ) * step

            alpha = tau1 - tj * tau0 - tj * tau1

            externality = mechanism._pdfs[i] * (
                tau0 * tj
                + alpha
                * (
                    1
                    - mechanism.types[i]
                    - mechanism.types[i] * allocations[i]
                    + (1 - 2 * mechanism.types[i]) * np.minimum(-allocations[i], 0)
                )
            )
            externalities[i] = externality
            exp_externality += step * externality

    exp_transfers, exp_externality

    print(exp_transfers, exp_externality, exp_transfers - exp_externality)

    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.plot(types, externalities)
    # ax.axhline(y=np.mean(externalities))
    # prettify(ax=ax)
    # ax.set_xlabel("externalities")
    # fig.tight_layout()
    # fig.savefig("./externalities.pdf", dpi=300)
