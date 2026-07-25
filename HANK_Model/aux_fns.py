"""Auxiliary routines: grid construction, fake news matrices, moments, IRF matching.

Contents
--------
Grid mapping        `mapx`, `mapgrid`, `mapgrid_inv` -- build the unevenly
                    spaced, tenure-specific asset grids of Table C.1.
Moments             `income_variance_n_quarters`, `weighted_percentile` -- the
                    income-process and untargeted moments of Tables 2 and 4.
IRF matching        `dynIRFs`, `irfloss` -- solve the model for a candidate
                    parameter vector and score it against the empirical IRFs
                    (Section 3.6, Figure 4, Table 5).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy

import numpy as np
from numba import njit

from behave_algo import redform_jacob


############################################################
## Asset grid construction
############################################################


@njit
def mapx(a, a0, a1, b0, b1):
    """Map `a` from the interval [a0, a1] onto [b0, b1], linearly."""
    return (a - a0) / (a1 - a0) * (b1 - b0) + b0


def mapgrid(ugrid, cutvalues, Gbreaks):
    """Stretch a uniform grid into an unevenly spaced asset grid.

    `ugrid` is a uniform grid on [0, 1].  `cutvalues` are the points of that
    grid at which the spacing changes (the 0th, 60th, 95th and 100th
    percentiles), and `Gbreaks` are the asset values those points should map
    to.  Each segment is stretched linearly between consecutive breakpoints, so
    a fixed share of gridpoints is allocated to each segment: 60 per cent of
    points below the 60th percentile of the asset range, 35 per cent between
    the 60th and 95th, and the top 5 per cent spread over the long thin upper
    tail.  See appendix C.1 and Table C.1.
    """
    G = ugrid * 0

    for i in range(cutvalues.shape[0] - 1):
        seg = (ugrid >= cutvalues[i]) & (ugrid <= cutvalues[i + 1])
        G[seg] = mapx(ugrid[seg], cutvalues[i], cutvalues[i + 1], Gbreaks[i], Gbreaks[i + 1])

    return G


def mapgrid_inv(G, cutvalues, Gbreaks):
    """Inverse of `mapgrid`: from the stretched asset grid back to the uniform grid."""
    u = G * 0

    for i in range(cutvalues.shape[0] - 1):
        seg = (G >= Gbreaks[i]) & (G <= Gbreaks[i + 1])
        u[seg] = mapx(G[seg], Gbreaks[i], Gbreaks[i + 1], cutvalues[i], cutvalues[i + 1])

    u[G > Gbreaks[-1]] = 1

    return u


######################################################################################################
## Steady-state moments
######################################################################################################


@njit
def income_variance_n_quarters(z, pi_z, z_markov, n):
    """Variance of the n-quarter change in log income.

    Used for the one-year and five-year earnings-change standard deviations
    reported in Table 2.  Note that the mean n-quarter change is zero under the
    stationary distribution, so the second moment computed here is the variance.

    Parameters
    ----------
    z        : array, income levels on the productivity grid
    pi_z     : array, stationary distribution over that grid
    z_markov : array, one-quarter transition matrix
    n        : int, horizon in quarters
    """
    nz = len(z)
    log_z = np.log(z)

    # n-step transition matrix
    trans_n = z_markov.copy()
    trans_n = trans_n.T
    for i in range(n - 1):
        trans_n = trans_n @ z_markov
    trans_n = trans_n.T

    # E[log(z_t+n) - log(z_t)]^2
    variance = 0.0
    for i in range(nz):
        for j in range(nz):
            log_diff = log_z[j] - log_z[i]
            variance += pi_z[i] * trans_n[i, j] * log_diff ** 2

    return variance


def weighted_percentile(a, d, percentile):
    """Percentile of values `a` under the distribution `d`, both over the state space.

    Used for the 90-10 income ratio (Table 2) and the top-10 per cent wealth
    share (Table 4).
    """
    a_flat = a.flatten()
    d_flat = d.flatten()

    # sort by value
    sorted_indices = np.argsort(a_flat)
    a_sorted = a_flat[sorted_indices]
    d_sorted = d_flat[sorted_indices]

    # cumulative distribution function, normalised
    cdf = np.cumsum(d_sorted)
    cdf /= cdf[-1]

    return np.interp(percentile / 100.0, cdf, a_sorted)


########################################################################################################
## IRF matching
########################################################################################################


def dynIRFs(x, modssin, hhJk, hhFk, dynmod, irftarget, irfW,
            unknowns, targets, exogenous, T, Etype):
    """Solve the model at parameter vector `x` and score it against the data.

    Implements the minimum-distance objective of Section 3.6, equation (14).
    The model is solved for a 1 p.p. annualised monetary policy shock and the
    resulting IRFs for the Bank Rate, output, the price level excluding rent,
    house prices and rents are compared with the empirical SVAR responses of
    Section 2, weighted by the inverse variances of those responses.

    Parameters
    ----------
    x         : array, candidate parameters.  The first six are always
                (kappa, kappaw, phi, gammatax, rhom, phiy); the remainder are
                expectations parameters whose interpretation depends on `Etype`.
                Note that `x[6]` and `x[7]` are *update probabilities*
                `1 / (1 + delta)`, converted to `delta` below.
    modssin   : SteadyStateDict, the model steady state
    hhJk      : JacobianDict, household rational-expectations Jacobian
    hhFk      : JacobianDict, household fake news matrices
    dynmod    : the `create_model` object for the dynamic model
    irftarget : array (12, 5), empirical IRFs to match
    irfW      : array (60, 60), weighting matrix (inverse variances)
    Etype     : int, expectations process:
                1 -- rational expectations
                2 -- sticky expectations only (gamma = 0)
                5 -- sticky and extrapolative, common parameters for all prices
                6 -- sticky and extrapolative, separate parameters for house
                     prices (the baseline; parameters reported in Table 5)

    Returns
    -------
    irfloss  : the scalar minimum-distance objective
    modirfs  : array (12, 5), model IRFs
    modssout : the steady state with the candidate parameters substituted in
    Gsolve   : the general equilibrium Jacobian of the solved model
    hhJk     : the household Jacobian actually used (behavioural, unless Etype 1)
    """
    modssout = copy.deepcopy(modssin)

    # dynamic parameters
    modssout['kappa'] = x[0] * 1        # slope of the price Phillips curve
    modssout['kappaw'] = x[1] * 1       # slope of the wage Phillips curve
    modssout['phi'] = x[2] * 1          # Taylor rule weight on inflation
    modssout['gammatax'] = x[3] * 1     # debt stabilisation in the fiscal rule
    modssout['rhom'] = x[4] * 1         # monetary policy smoothing
    modssout['phiy'] = x[5] * 1         # Taylor rule weight on the output gap

    # prices whose expectations are behavioural
    behave_prices = ['r', 'pr', 'w', 'hours', 'Tax', 'Div', 'ph', 'pi']

    # replace the rational-expectations household Jacobian with the behavioural one
    if Etype == 2:      # sticky expectations
        modssout['deltaWK'] = 1 / (1 - x[6]) - 1
        modssout['deltaWKph'] = 1 / (1 - x[7]) - 1

        hhJk = redform_jacob(hhJk, hhFk, behave_prices, 0.0, modssout['deltaWK'])

    if Etype == 5:      # sticky and extrapolative, common parameters
        modssout['deltaWK'] = 1 / (1 - x[6]) - 1
        modssout['gammaWK'] = x[7] * 1

        hhJk = redform_jacob(hhJk, hhFk, behave_prices,
                             modssout['gammaWK'], modssout['deltaWK'], Trunc=350)

    if Etype == 6:      # sticky and extrapolative, separate house price parameters
        modssout['deltaWK'] = 1 / (1 - x[6]) - 1
        modssout['deltaWKph'] = 1 / (1 - x[7]) - 1
        modssout['gammaWK'] = x[8] * 1
        modssout['gammaWKph'] = x[9] * 1

        # first pass sets common parameters on all prices, second pass overwrites
        # the house price column with its own parameters
        hhJk = redform_jacob(hhJk, hhFk, behave_prices,
                             modssout['gammaWK'], modssout['deltaWK'], Trunc=350)
        hhJk = redform_jacob(hhJk, hhFk, ['ph'],
                             modssout['gammaWKph'], modssout['deltaWKph'], Trunc=350)

    # solve model dynamics
    Gsolve = dynmod.solve_jacobian(modssout, unknowns, targets, exogenous, T=T, Js={'hh': hhJk})

    # monetary policy shock: 25bp on the quarterly nominal rate on impact
    epsr = np.arange(T) * 0.0
    epsr[0] = 0.0025

    IRFi = (Gsolve['i']['epsr'] @ epsr)[0:20]

    # rescale so the impact response is exactly 1 p.p. annualised, matching the
    # normalisation of the empirical IRFs
    scal = 0.25 / (IRFi[0])

    model_BR = scal * IRFi[0:12] * 4
    model_cpi = np.cumsum(Gsolve['pi']['epsr'] @ epsr * scal)[0:12]
    model_gdp = (Gsolve['Y']['epsr'] @ epsr * scal)[0:12]
    model_hpi = (Gsolve['ph']['epsr'] @ epsr * scal)[0:12] + model_cpi
    model_rent = (Gsolve['rentindex']['epsr'] @ epsr * scal)[0:12] / modssout['rentindex'] + model_cpi

    # house prices and rents are matched in nominal terms, hence the addition of
    # the cumulated price level to both
    modirfs = np.array([model_BR, model_gdp, model_cpi, model_hpi, model_rent]).transpose()

    # weighted sum of squared deviations, stacked variable by variable
    irflossC = (modirfs - irftarget).flatten(order='F')
    irfloss = irflossC[np.newaxis, :] @ irfW @ irflossC[:, np.newaxis]

    return irfloss, modirfs, modssout, Gsolve, hhJk


def irfloss(x, modssin, hhJk, hhFk, dynmod, irftarget, irfW,
            unknowns, targets, exogenous, T, Etype):
    """Scalar objective for the minimum-distance estimator.

    Thin wrapper around `dynIRFs` returning only the loss, in the form
    `scipy.optimize.minimize` expects.  The parameters reported in Table 5 are
    the minimiser of this function under `Etype = 6`; re-running the estimation
    takes several hours, so Main.ipynb uses the stored solution.
    """
    irflossV, modirfs, modssout, Gsolve, hhJk = dynIRFs(x, modssin, hhJk, hhFk, dynmod, irftarget,
                                                        irfW, unknowns, targets, exogenous, T, Etype)

    return irflossV
