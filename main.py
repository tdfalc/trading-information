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

    dist = BetaMixture((8, 60), (30, 30), (0.5, 0.5))
    dist = Uniform(0, 1)
    xs = np.linspace(0, 1, 1000)
    ys = dist.pdf(xs) * 2.5

    mechanism = Mechanism(num_types=1000, dist=dist)

    tau0 = 1
    tau1 = 1
    prob_state0 = 0.5
    threshold_type = 0.75

    alpha = 0

    allocations, transfers, multiplier = mechanism.solve(alpha, threshold_type)
    types = mechanism.types

    fig, ax = plt.subplots(figsize=(5, 3))

    # # use_tex()
    ax.plot(types, allocations)
    # ax.plot(xs, transfers)

    prettify(ax=ax)

    ax.set_xlabel("Total Quantity ($a_i$ + $a_j$)")

    fig.tight_layout()
    fig.savefig("./test.pdf", dpi=300)
