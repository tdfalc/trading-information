# Markets for Information
 
Python implemention of [Selling Information in Games with Externalities](https://arxiv.org/abs/2505.00405). 

In this work, we:

* Model a competitive market as game of incomplete information between two players.
* Consider one player observes some payoff-relevant state and can sell (possibly noisy)
messages thereof to the other.
* Frame the decision of what information to sell, and at what price, as a joint problem of [mechanism design](https://en.wikipedia.org/wiki/Mechanism_design) and [information design](https://en.wikipedia.org/wiki/Bayesian_persuasion).
* Characetize the mechanism that maximizes profit, which is  the payment minus the externality induced by selling information to a competitor, that is, the cost of refining a competitor’s beliefs.

As an example, in the case of congruent binary types (Section 4.1), information is sold only to the low type when the intensity of competition ($\tau$) is below the solid line, and only to the high type when $\tau$ is below the dashed line (see Figure 6).

<p align="center" width="100%">
    <img  src="./analytics/docs/sim01_binary_types_congruent/binary_types_congruent.gif">
</p>

If you find this useful in your work, we kindly request that you cite the following publication(s):

```
@misc{falconer2025sellinginformation,
      title={Selling Information in Games with Externalities}, 
      author={Thomas Falconer and Anubhav Ratha and Jalal Kazempour and Pierre Pinson and Maryam Kamgarpour},
      year={2025},
      eprint={2505.00405},
      archivePrefix={arXiv},
      primaryClass={cs.GT},
      url={https://arxiv.org/abs/2505.00405}, 
}
```