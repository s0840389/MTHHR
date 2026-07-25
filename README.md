# Monetary Transmission in a HANK Model with Housing and Rental Sectors

Replication code for Daniel Albuquerque, Thomas Lazarowicz and Jamie Lenney,
*Monetary Transmission in a HANK Model with Housing and Rental Sectors*.

The paper is included in this repository as [`MTHANKHOUSE.pdf`](MTHANKHOUSE.pdf).

This repository reproduces the **baseline model** of the paper: its steady
state, its response to a monetary policy shock, and the decompositions built on
that response. Everything runs from a single notebook,
[`Main.ipynb`](Main.ipynb).

---

## What is reproduced

| Paper output | Notebook section |
| --- | --- |
| Tables 2 and 4 — calibration targets and untargeted moments | *Solve the steady state*, *Income process and untargeted moments* |
| Figure 4 — IRF matching, model against SVAR | *IRF matching* |
| Figure 5 — housing market response to a higher interest rate | *Housing market response* |
| Figure 6 — consumption decomposition, iMPCs and heterogeneity | *Jacobians*, *The housing channel of monetary policy* |
| Figure 7 — housing market elasticities | *Elasticities* |
| Figures C.1 and C.2 — asset distribution and transition probabilities | *Transitions and wealth distributions* |
| Table 5 — IRF-matched parameters | Applied as `x0` in *IRF matching* |
| Figure D.1 — sales volume and rental share | *IRF matching* |

**Not reproduced [For now]:** the empirical SVAR of Section 2 (its estimated IRFs enter
here as the data files in `data/`), and the counterfactuals and extensions of
Sections 4.4 and 4.5 (Figures 8, 9 and 10). The model blocks those extensions
need are kept in the source under a clearly marked `NOT USED BY Main.ipynb`
banner, so they can be swapped back in.

---

## Getting started

### Requirements

Python 3.9 or later, and:

```bash
pip install sequence-jacobian numpy scipy numba pandas matplotlib jupyter
```

Developed against `sequence-jacobian` 1.0.0.

### Running

```bash
jupyter lab Main.ipynb     # or: jupyter notebook Main.ipynb
```

Run the cells in order.

Two steps are expensive: solving the steady state, and computing the household
Jacobians over a 400-quarter horizon. Both are cached as pickles in `data/`.
The first cell detects whether those caches exist and sets `solvess` and
`solveJK` accordingly, so:

- **on a fresh clone** the caches are absent, both steps run, and the results
  are saved;
- **on later runs** they can be loaded in seconds.

Set either flag to `1` by hand to force a recompute — you must do this after
changing anything in the calibration, since the cache will otherwise be stale.
The caches are gitignored and never committed.

### Repository layout

```
Main.ipynb              the replication notebook; run this
MTHANKHOUSE.pdf         the paper
data/
  reirf_match_BMT.csv   empirical IRFs from the Section 2 SVAR (the matching targets)
  wmatrix_match_BMT.csv weighting matrix for the minimum-distance objective
  *.pickle              cached steady state and Jacobians (generated, gitignored)
HANK_Model/
  household.py          household problem: grids, income, tenure choice, DC-EGM
  steadystate.py        steady-state blocks
  dynamics.py           aggregate blocks of the dynamic model
  fullgridclass.py      *** state-dependent grid stages for sequence_jacobian ***
  behave_algo.py        *** behavioural expectations in the sequence space ***
  aux_fns.py            grid mapping, steady-state moments, IRF matching
```

The model is built on [`sequence_jacobian`](https://github.com/shade-econ/sequence-jacobian)
(Auclert, Bardóczy, Rognlie and Straub, 2021). Reading its documentation first
will make everything here much easier to follow.

---

## The two code innovations

Most of this repository is a standard application of `sequence_jacobian`. Two
pieces are not, and both are needed to solve this model at all. Each is
documented in full at the top of its module; this is the summary.

### 1. State-dependent grids — `HANK_Model/fullgridclass.py`

**The problem.** A `sequence_jacobian` `Continuous1D` or `Continuous2D` stage
assumes each endogenous policy is chosen from a *single* grid, shared by every
point of the state space: the stage looks up `inputs['a_grid']`, a 1-D vector.
That assumption fails here on both endogenous dimensions.

- **Assets.** The asset grid has to depend on the housing transition. A renter
  cannot borrow at all, so their grid starts at zero; an owner-occupier may
  borrow against a house up to the LTV/LTI limit, so theirs starts at `-p_h H_2`;
  a landlord may borrow against both their home and their rental flat. The tops
  of the grids differ too, being a multiple of the property owned (Table C.1).

  Forcing all tenures onto one common grid would mean either spending most of
  the 250 gridpoints on regions a given tenure can never reach, or resolving the
  borrowing-constrained region far too coarsely — and that region is exactly
  where the policy functions kink and where the high MPCs driving the results
  live.

- **Rental prices.** Renters and landlords carry a household-specific nominal
  rent as a state, because rents reset only with probability `theta_r` each
  period. Owner-occupiers have no rent to carry. The grid is therefore
  state-dependent by construction, not by choice.

**The fix.** `Continuous2DFullGrid` reads `<policy>_grid_full` — an array with
the *same shape as the state space*, `(n_h, n_z, n_a, n_pr)` — instead of the
1-D `<policy>_grid`. Interpolation of the policy onto the grid, the resulting
policy lottery, and the grid spacing used to differentiate that lottery are all
done grid-by-grid. In this model the grids vary only along the housing-transition
dimension, which the interpolation exploits, so the cost of the extra generality
is one Python-level loop over 17 states.

**It also fixes an upstream bug.** `sequence_jacobian`'s `ShockedPolicyLottery2D`
inherits the constructor of `PolicyLottery2D` (which stores `i1, pi1, i2, pi2`)
but its `__matmul__` calls `forward_policy_shock_2d(X, self.i, self.pi)` —
attributes that exist only on the *one*-dimensional lottery, and passes three
arguments to a function that takes seven. `Continuous2D` compounds this by
passing the *derivatives* of the lottery weights in the slots meant for the
steady-state weights, discarding the latter entirely.
`ShockedPolicyLottery2DFullGrid` carries both sets and passes all seven.

### 2. Behavioural expectations — `HANK_Model/behave_algo.py`

`redform_jacob` takes the rational-expectations Jacobian of a block and returns
the generalised Jacobian implied by the reduced-form expectations process of
Kohlhas and Walther (2021), in which agents simultaneously under-react to news
about the future and extrapolate from current conditions:

```
f_t x_{t+k} = x_ss + 1/(1+delta) * ( delta * f_{t-1} x_{t+k} + E*_t[dx_{t+k}] - gamma * dx_t )
```

Three cases are nested — rational (`delta = gamma = 0`), sticky
(`delta > 0, gamma = 0`), and sticky-and-extrapolative (`delta > 0, gamma != 0`,
the baseline). Section 3.3 and appendix C.2 of the paper give the derivation;
Table 5 gives the estimated parameters.

**A separate write-up of this method, with worked examples, is at
<https://s0840389.github.io/jamielenneyecon/>. That is the place to look for the
details.**

The implementation point worth flagging here: following Bardóczy and Guerreiro
(2025), the generalised Jacobian can be written as
`Jhat = J·Λ₀ + Σ_h R_h(Λ_h − Λ_{h−1})`, which taken literally means forming and
multiplying `T` matrices of size `T × T` — expensive at `T = 400`. The code
instead works with the **fake news matrix** `F`, the object `sequence_jacobian`
builds on the way to `J` and then discards. Because `F[t, s]` isolates the *new
information* arriving at each date, the news and extrapolation channels act on
it separately and additively, and the whole calculation collapses to a single
pass over `F`.

Getting hold of `F` needs no changes to `sequence_jacobian` at all. The
Jacobian is built by accumulating the fake news matrix along its diagonals,
`J[t, s] = J[t-1, s-1] + F[t, s]`, so `F` is recovered by differencing:
`JtoF` inverts the recursion exactly (to machine precision), and `JtoFdict`
applies it to a whole `JacobianDict`. The model is therefore solved with the
standard, unmodified `sequence_jacobian` API — `hh.jacobian(...)` — and `F` is
derived afterwards.

---

## Solution method

Beyond the two items above, the model follows Section 3.5 of the paper:

- Steady state and dynamics are solved with the sequence-space Jacobian method
  of Auclert et al. (2021), to first order.
- The household problem is split into four stages: exogenous productivity, the
  rental-contract reset, the discrete tenure choice, and consumption-saving.
- The discrete tenure choice is smoothed by Gumbel taste shocks, giving the
  closed-form logit transition probabilities of equation (2).
- The consumption-saving stage uses the endogenous grid point method of Carroll
  (2006) augmented with the upper envelope step of Iskhakov et al. (2017),
  which is necessary because the discrete choice makes the value function
  non-concave and the Euler equation can have several solutions.
- The model tracks the housing **transition** rather than the housing tenure —
  17 states — because budget and borrowing constraints (Table 1) are specific to
  the move being made. The enumeration is documented at the top of
  `household.py` and is the key to reading that file.

---

## How to cite

```bibtex
@techreport{albuquerque_lazarowicz_lenney_mthhr,
  author = {Albuquerque, Daniel and Lazarowicz, Thomas and Lenney, Jamie},
  title  = {Monetary Transmission in a {HANK} Model with Housing and Rental Sectors},
  year   = {2026}
}
```

A previous version of the paper circulated as *Monetary Transmission Through the
Housing Sector*.

---

## Disclaimer

The views expressed in the paper and in this code are those of the authors and
do not necessarily represent those of the Bank of England or any of its
committees.
