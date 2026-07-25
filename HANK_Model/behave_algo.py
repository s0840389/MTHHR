"""Behavioural (sticky and extrapolative) expectations in the sequence space.

This module is the second of the two code innovations in this replication
package (the other is `fullgridclass.py`).  `redform_jacob` takes the
*rational-expectations* Jacobian of a block and returns the generalised
Jacobian that would obtain if agents formed expectations according to the
reduced-form process of Kohlhas and Walther (2021), in which agents
simultaneously under-react to news about the future and extrapolate from
current conditions.

A standalone write-up of the method, with worked examples and a small test
suite, lives at

    https://s0840389.github.io/jamielenneyecon/

and that is the place to look for the derivations.  What follows is the short
version needed to read the code.


The expectations process
------------------------
Households forecast an input `x` of their decision problem `k` periods ahead as
(equation 5 of the paper; equation C.2 of the appendix)

    f_t x_{t+k} = x_ss + 1 / (1 + delta) * ( delta * f_{t-1} x_{t+k}
                                             + E*_t[dx_{t+k}] - gamma * dx_t )

where `E*` is the perfect-foresight (rational) forecast, `delta` governs
under-reaction to news about the future, and `gamma` governs extrapolation from
today's realisation.  Three cases are nested:

    delta = 0, gamma = 0   rational expectations
    delta > 0, gamma = 0   sticky expectations (Auclert et al., 2020)
    delta > 0, gamma != 0  sticky and extrapolative expectations (the baseline)

Iterating forward from the steady state gives the forecast at date `t` of a
date-`k` outcome as a weighted sum of the realised path (appendix eq. C.3): a
*news* term whose weight `sum_{l=0}^{k} delta^l / (1 + delta)^{l+1}` rises
towards one as the date approaches, and an *extrapolation* term that loads on
the history of realisations with geometrically declining weight.


Mapping into the sequence space
-------------------------------
Following Bardoczy and Guerreiro (2025), a generalised Jacobian can be written
in terms of the rational-expectations Jacobian `J` and forecast matrices
`Lambda_k` that map the realised path into the sequence of forecasts
(appendix eq. C.5):

    Jhat = J Lambda_0 + sum_h R_h ( Lambda_h - Lambda_{h-1} )

where `R_h` shifts `J` diagonally by horizon `h`.  Implemented literally this
means forming and multiplying `T` matrices of size `T x T`, which at `T = 400`
is expensive.

The implementation here is algebraically equivalent but works with the *fake
news* matrix `F` instead of `J`.  `F` is the object Auclert et al. (2021)
construct on the way to `J`: `F[t, s]` is the response at date `t` to news at
date 0 about a shock at date `s`, with the accumulated-anticipation effects
stripped out (`J[t, s] = J[t-1, s-1] + F[t, s]`).  Because `F` isolates the
*new information* arriving at each date, the two channels of the expectations
process act on it separately and additively, and the whole calculation
collapses into the single triple loop in `compute_Jbehave`.

`sequence_jacobian` builds `F` internally but returns only `J`.  Rather than
reaching into its Jacobian machinery to intercept `F`, we simply invert the
accumulation: `JtoF` below recovers `F` from `J` exactly, so the model can be
solved with the standard, unmodified `sequence_jacobian` API.
"""

import copy

import numpy as np
from numba import njit, prange


############################################################
## Recovering the fake news matrix from the Jacobian
############################################################


@njit
def JtoF(Jyp):
    """Recover a fake news matrix from a Jacobian.

    The exact inverse of `sequence_jacobian`'s `HetBlock.J_from_F`, which builds
    the Jacobian by accumulating the fake news matrix along its diagonals:

        J[t, s] = J[t-1, s-1] + F[t, s]

    with the first row and first column of `J` left equal to those of `F`.
    Undoing that recursion is a single differencing pass, so `F` never has to be
    intercepted inside `sequence_jacobian`; the block's Jacobian can be computed
    with the standard `.jacobian()` method and `F` reconstructed here.

    The round trip is exact to machine precision.
    """
    T = Jyp.shape[0]

    Fyp = np.zeros((T, T))

    Fyp[0, :] = Jyp[0, :]
    Fyp[:, 0] = Jyp[:, 0]

    for i in range(1, T):
        Fyp[i, 1:] = Jyp[i, 1:] - Jyp[i - 1, 0:T - 1]

    return Fyp


def JtoFdict(Jk):
    """Apply `JtoF` to every matrix of a `JacobianDict`.

    Returns a `JacobianDict` of fake news matrices with the same outputs and
    inputs as `Jk`, in the form `redform_jacob` expects.
    """
    Fk = copy.deepcopy(Jk)

    for i in Jk.outputs:
        for j in Jk.inputs:
            # not every output responds to every input
            if j in Fk[i]:
                Fk[i][j] = JtoF(Jk[i][j])

    return Fk


@njit
def calc_news_eff(T, delta):
    """Weight on news that has been available for `k` periods.

    `fts0[k] = sum_{l=0}^{k} delta^l / (1 + delta)^{l+1}` is the share of a
    date-0 revelation about the future that has been incorporated into
    expectations after `k` periods.  It starts at `1 / (1 + delta)` and rises to
    one, so agents converge on the rational forecast as the date approaches;
    with `delta = 0` it equals one immediately and the news channel is rational.

    Returns
    -------
    fts0     : array (T,), cumulative news weight after `k` periods
    deltaseq : array (T-1,), the per-period increments being cumulated
    """
    vdelta = 1 / (1 + delta)

    deltaseq = vdelta * np.power(delta * vdelta, np.arange(T - 1))

    fts0 = np.zeros(T)
    for t in range(0, T):
        fts0[t] = np.sum(deltaseq[0:t + 1])

    return fts0, deltaseq


@njit
def calc_extrap_eff(T, gamma, delta, Trunc=20, TruncDecay=0.86):
    """Weights describing extrapolation from realisations into forecasts.

    `forcwgt_y0[t, s]` is the contribution of the realisation of `x` observed
    `t` periods ago to the forecast of `x` at horizon `s`.  Row 0 is the impact
    effect: seeing `dx_0` today shifts the forecast of every future date by
    `-gamma / (1 + delta)`.  Subsequent rows apply the recursion
    `delta / (1 + delta)`, since a past observation keeps influencing forecasts
    through the previous period's forecast that the agent anchors on.

    `Trunc` and `TruncDecay` implement assumption (2) of appendix C.2.1: beyond
    some long horizon agents expect a return to the steady state.  Rather than
    imposing that as a hard cut-off, the extrapolation weight is held flat out
    to `Trunc` periods and then decays geometrically at rate `TruncDecay`, so
    that very-long-horizon forecasts revert smoothly to `x_ss`.  Without this
    the extrapolation term would place undamped weight on forecasts arbitrarily
    far into the future, where the model must in any case converge back to the
    steady state.  The baseline calls use `Trunc = 350` with `T = 400`.
    """
    vdelta = 1 / (1 + delta)

    # matrix describing the contribution of y0 to the forecast of p_s at time t
    forcwgt_y0 = np.zeros((T, T))
    forcwgt_y0[0, 0] = 1
    forcwgt_y0[0, 1:Trunc] = -gamma * vdelta
    forcwgt_y0[0, Trunc:T - 1] = -gamma * vdelta * np.power(TruncDecay, np.arange(T - Trunc - 1))

    # after period 0, y0 contributes via lagged forecasts of yk and yk-h
    for t in range(1, T):
        forcwgt_y0[t, t + 1:T] = delta * vdelta * forcwgt_y0[t - 1, t + 1:T]
        forcwgt_y0[t, 1:T - t] = forcwgt_y0[t, t + 1:T]

    return forcwgt_y0


@njit(parallel=True)
def compute_Jbehave(Fkin, fts0, extrap_effg):
    """Assemble one behavioural Jacobian from one fake news matrix.

    Walks the same recursion that turns a fake news matrix into a Jacobian
    (`J[t, s] = J[t-1, s-1] + F[t, s]`), but weights each contribution by how
    much of the relevant information agents have actually acted upon:

    * `k < s` -- the date-`s` shock is still in the future at date `t - k`, so
      it reaches behaviour only as *news*, discounted by `fts0[k]`, the share of
      news absorbed after `k` periods.
    * `k >= s` -- the shock has been realised, so it reaches behaviour through
      *extrapolation* from the observed path, `extrap_effg[k - s, t - k]`.

    Parameters
    ----------
    Fkin        : array (T, T), rational-expectations fake news matrix
    fts0        : array (T,), news weights from `calc_news_eff`
    extrap_effg : array (T, T), extrapolation weights already contracted with
                  the fake news matrix, i.e. `calc_extrap_eff(...) @ Fkin.T`

    Returns
    -------
    Jkg : array (T, T), the generalised (behavioural) Jacobian
    """
    T = Fkin.shape[0]
    Jkg = np.zeros(Fkin.shape)

    for t in prange(T):
        for s in prange(0, T):
            for k in range(0, t + 1):
                if k < s:
                    Jkg[t, s] += Fkin[t - k, s - k] * fts0[k]
                else:
                    Jkg[t, s] += extrap_effg[k - s, t - k]

    return Jkg


def redform_jacob(Jk, Fk, p, gamma, delta, Trunc=20, TruncDecay=0.86):
    """Convert a rational-expectations Jacobian into a behavioural one.

    Parameters
    ----------
    Jk    : JacobianDict, the rational-expectations Jacobian of the block, as
            returned by `sequence_jacobian`'s standard `.jacobian()` method
    Fk    : JacobianDict, the matching fake news matrices, i.e. `JtoFdict(Jk)`
    p     : list of str, the inputs whose expectations are behavioural.  Inputs
            not in this list keep their rational-expectations Jacobian, which
            is what makes it possible to apply different `(gamma, delta)` pairs
            to different prices by calling this function more than once -- the
            baseline does exactly that, giving house prices their own
            parameters (`gammaWKph`, `deltaWKph`).
    gamma : float, extrapolation parameter (negative implies over-extrapolation)
    delta : float, stickiness parameter; the per-period probability of updating
            expectations is `1 / (1 + delta)`
    Trunc, TruncDecay : long-horizon reversion to steady state, see
            `calc_extrap_eff`

    Returns
    -------
    Jkout : JacobianDict with the behavioural Jacobian substituted in for every
            output with respect to each input in `p`
    """
    # start from a deep copy so inputs outside `p` retain the FIRE Jacobian
    # (this matters for e.g. the discount factor shock)
    Jkout = copy.deepcopy(Jk)

    T = Fk[Jk.outputs[0]][Jk.inputs[0]].shape[0]

    fts0, deltaseq = calc_news_eff(T, delta)
    extrap_eff = calc_extrap_eff(T, gamma, delta, Trunc, TruncDecay)

    for i in Fk.outputs:
        for j in p:
            # not every output responds to every input; skip the pairs the
            # block does not define rather than requiring the caller to know
            # which combinations exist
            if j in Fk[i]:
                Jkout[i][j] = compute_Jbehave(Fk[i][j], fts0, extrap_eff @ Fk[i][j].T)

    return Jkout