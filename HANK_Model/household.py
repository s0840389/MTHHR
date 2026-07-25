"""The household block: grids, income, tenure choice and consumption-saving.

This module supplies every piece of the household problem described in Section
3.1 of the paper.  It is assembled into a `sequence_jacobian` `StageBlock` in
Main.ipynb from four stages:

    1. `prod`         exogenous idiosyncratic productivity `z` (Markov)
    2. `rent_calvo`   exogenous reset of the household-specific rent (Markov)
    3. `house_choice` discrete housing tenure choice (logit / taste shocks)
    4. `consav`       continuous consumption-saving choice (DC-EGM)

with `make_grids`, `prices_baseline` and `labor_income` as hetinputs and
`labor_supply`, `housingDemand`, `assetDemand` as hetoutputs.

State space
-----------
Every array in this module is on `(n_h, n_z, n_a, n_pr)`:

    n_h  = 17   housing *transition* (see below)
    n_z  = 15   idiosyncratic labour productivity (5 persistent x 3 transitory)
    n_a  = 250  financial assets, on a grid that varies with `n_h`
    n_pr = 3    the household-specific rental price carried into the period

Housing transitions
-------------------
We track the housing *transition* `h` rather than the housing tenure, because
budget and borrowing constraints (Table 1 of the paper) are specific to the
move being made, not to the destination.  A household moving 'own -> rent'
faces different costs and a different asset grid from one already in
'rent -> rent'.  The 17 transitions are:

     0.  own       -> own            9.  ownF      -> ownF
     1.  own       -> rent          10.  ownF      -> rent
     2.  rent      -> ownF          11.  landlord  -> landlord2
     3.  rent      -> rent          12.  landlord2 -> landlord
     4.  own       -> landlord      13.  landlord2 -> landlord2
     5.  landlord  -> own           14.  rent      -> rent*
     6.  landlord  -> landlord      15.  landlord  -> landlord*
     7.  own       -> ownF          16.  landlord2 -> landlord2*
     8.  ownF      -> own

where 'own' is an owner-occupier of a house (size `H2`), 'ownF' an
owner-occupier of a flat (size `H1 = Fsize`), 'landlord' an owner-occupier of a
house who also lets one flat, and 'landlord2' one who lets two.  States 14-16
are the starred duplicates of 3, 6 and 13: they are the same tenure, but
distinguish households whose rent has just been reset at the market spot price
from those still carrying last period's rent.  This is what makes the
individual rental price a household-level state (Section 3.1, Stage 1).

Two grids per transition
------------------------
Because a transition starts in one tenure and ends in another, each transition
needs *two* asset grids: `a_grid_exp_t0`, the beginning-of-period grid of the
tenure being left, and `a_grid_exp_t1`, the end-of-period grid of the tenure
being entered.  A single common grid would resolve the borrowing-constrained
region badly for some tenures and waste points for others; see `fullgridclass.py`
for the machinery that lets `sequence_jacobian` handle state-dependent grids,
and Table C.1 of the paper for the four grids.
"""

import numpy as np
from numba import njit, prange

from sequence_jacobian import grids, interpolate
from sequence_jacobian.utilities.discretize import nonlinspace

from aux_fns import mapgrid


############################################################################
## Utility functions
############################################################################


@njit
def util_c_KMV(c, h, eis, phihs):
    """Period utility over non-durable consumption `c` and housing services `h`.

    Cobb-Douglas aggregator with housing share `phihs`, inside a CRRA shell
    with `crra = 1 / eis` (equation in Section 3.1).  Non-positive consumption
    is assigned a large negative utility so that the upper envelope never
    selects it.
    """
    crra = 1 / eis

    if crra == 1:
        u = np.log(c ** (1 - phihs) * h ** phihs)
    else:
        u = 1 / (1 - crra) * (c ** (1 - phihs) * h ** phihs) ** (1 - crra)

    u = u * (c > 0) - 100000 * (c <= 0)

    return u


@njit
def marg_util_c_KMV(c, h, eis, phihs):
    """Marginal utility of non-durable consumption."""
    crra = 1 / eis

    uc = (1 - phihs) * c ** (-phihs) * h ** phihs * (c ** (1 - phihs) * h ** phihs) ** (-crra)

    uc = uc * (c > 0) + 100000 * (c <= 0)

    return uc


@njit
def inv_marg_util_c_KMV(uc, h, eis, phihs):
    """Consumption implied by a level of marginal utility.

    The inverse of `marg_util_c_KMV` in `c`, holding housing services fixed.
    This is the step that makes the endogenous grid point method work: given
    end-of-period marginal value, it returns consumption directly rather than
    by root-finding.
    """
    crra = 1 / eis

    c = (uc / ((1 - phihs) * h ** (phihs * (1 - crra)))) ** (1 / (-crra - phihs + crra * phihs))

    return c


#####################################################################################################
## Idiosyncratic income process
#####################################################################################################


def mixedAR1(rho1, sig1, n1, rho2, sig2, n2, minz):
    """Discretise the sum of a persistent and a transitory AR(1) process.

    Section 3.1: `z = z1 + z2` with `z1` persistent and `z2` transitory.  Each
    component is discretised separately by Rouwenhorst (1995) and combined by
    Kronecker product, giving `n1 * n2` states.  The stationary distribution is
    recovered as the unit eigenvector of the transposed transition matrix
    rather than by iteration.

    The grid is floored at `minz` to keep very low income realisations away
    from zero (calibrated to the 90-10 ratio, Table 2), and then rescaled so
    that mean productivity is one.
    """
    z1, z1pi, z1markov = grids.markov_rouwenhorst(rho1, (sig1 ** 2 / (1 - rho1 ** 2)) ** 0.5, n1)
    z2, z2pi, z2markov = grids.markov_rouwenhorst(rho2, (sig2 ** 2 / (1 - rho2 ** 2)) ** 0.5, n2)

    z = np.kron(z1, z2)
    z_markov = np.kron(z1markov, z2markov)

    # stationary distribution: eigenvector of z_markov' with unit eigenvalue
    ez, vz = np.linalg.eig(z_markov.T)
    ezi = np.argmin(np.abs(ez - 1))
    z_pi = np.abs(vz[:, ezi]) / np.sum(np.abs(vz[:, ezi]))

    z = np.maximum(z, np.ones_like(z) * minz)   # avoid very low values of z

    zmu = np.sum(z * z_pi)
    z = z / zmu                                 # normalise mean productivity to one

    return z, z_pi, z_markov


################################################################################
## Hetinput: grids
################################################################################


def make_grids(rho_z1, sd_z1, n_z1, rho_z2, sd_z2, n_z2, n_h, min_z, min_a, max_a, n_a,
               borlim, phss, Fsize, omegah, pr_calvo, kappah, kappahLL, prss):
    """Build the productivity, asset and rental-price grids, and the Markov chains.

    The asset grids follow Table C.1.  A uniform grid on [0, 1] is stretched by
    `mapgrid` so that 60 per cent of points sit below the 60th percentile of the
    range, 35 per cent between the 60th and 95th, and 5 per cent cover the long
    upper tail out to `max_a`.  The breakpoints differ by tenure because
    borrowing capacity does:

        renters       cannot borrow at all, so the grid starts at `borlim`
        flat owners   may borrow against a flat, so it starts at `-H1 * ph`
        owners        may borrow against a house, so it starts at `-ph`
        landlords     may borrow against home and rental flat, so it starts at
                      `-ph * (kappah + kappahLL * Fsize)`

    Returns both `a_grid_exp_t0` (grid of the tenure being left) and
    `a_grid_exp_t1` (grid of the tenure being entered) for each of the 17
    transitions, plus the full state-space versions `a_grid_full_t0` and
    `a_grid_full` that `Continuous2DFullGrid` consumes.
    """
    z_grid, z_dist, z_markov = mixedAR1(rho_z1, sd_z1, n_z1, rho_z2, sd_z2, n_z2, min_z)
    n_z = n_z1 * n_z2

    # normalised grid between zero and one with n_a points, evenly spaced
    a_grid = nonlinspace(1.0, n_a, 1.0, amin=0.0)

    # points of the uniform grid at which the spacing changes; taking quantiles
    # of a_grid itself guarantees the cut values land exactly on gridpoints
    cutvaluesR = np.quantile(a_grid, [0, 0.6, 0.95, 1.0], interpolation='higher')
    cutvaluesO = np.quantile(a_grid, [0, 0.6, 0.95, 1.0], interpolation='higher')
    cutvaluesLL = np.quantile(a_grid, [0, 0.6, 0.95, 1.0], interpolation='higher')
    cutvaluesOF = np.quantile(a_grid, [0, 0.6, 0.95, 1.0], interpolation='higher')

    # asset values those cut points map to: [low point, 60th pct, 95th pct, max]
    GbreakR = np.array([borlim, 4, phss, max_a])
    GbreakO = np.array([-phss, 0, phss, max_a])
    GbreakLL = np.array([-1.0 * phss * (kappah + kappahLL * Fsize), 0, phss, max_a])
    GbreakLL2 = np.array([-1.0 * phss * (kappah + kappahLL * Fsize), 0, phss, max_a])
    GbreakOF = np.array([-phss * Fsize, 0, phss, max_a])

    a_grid_r = mapgrid(a_grid, cutvaluesR, GbreakR)     # renters grid
    a_grid_o = mapgrid(a_grid, cutvaluesO, GbreakO)     # owners grid
    a_grid_ll = mapgrid(a_grid, cutvaluesLL, GbreakLL)  # landlords grid
    a_grid_of = mapgrid(a_grid, cutvaluesOF, GbreakOF)  # flat owners grid
    a_grid_ll2 = mapgrid(a_grid, cutvaluesLL, GbreakLL2)  # two-flat landlords grid

    # end-of-period assets: the grid of the tenure being entered
    a_grid_exp_t1 = np.zeros((n_h, n_a))

    a_grid_exp_t1[0] = a_grid_o    # own to own
    a_grid_exp_t1[1] = a_grid_r    # own to rent
    a_grid_exp_t1[2] = a_grid_of   # rent to ownF
    a_grid_exp_t1[3] = a_grid_r    # rent to rent

    a_grid_exp_t1[4] = a_grid_ll   # own to landlord
    a_grid_exp_t1[5] = a_grid_o    # landlord to own
    a_grid_exp_t1[6] = a_grid_ll   # landlord to landlord

    a_grid_exp_t1[7] = a_grid_of   # own to ownF

    a_grid_exp_t1[8] = a_grid_o    # ownF to own
    a_grid_exp_t1[9] = a_grid_of   # ownF to ownF
    a_grid_exp_t1[10] = a_grid_r   # ownF to rent

    a_grid_exp_t1[11] = a_grid_ll  # landlord to landlord2
    a_grid_exp_t1[12] = a_grid_ll2  # landlord2 to landlord
    a_grid_exp_t1[13] = a_grid_ll2  # landlord2 to landlord2

    a_grid_exp_t1[14] = a_grid_r   # rent to rent*
    a_grid_exp_t1[15] = a_grid_ll  # landlord to landlord*
    a_grid_exp_t1[16] = a_grid_ll2  # landlord2 to landlord2*

    # beginning-of-period assets: the grid of the tenure being left
    a_grid_exp_t0 = np.zeros((n_h, n_a))

    a_grid_exp_t0[0] = a_grid_o    # own to own
    a_grid_exp_t0[1] = a_grid_o    # own to rent
    a_grid_exp_t0[2] = a_grid_r    # rent to ownF
    a_grid_exp_t0[3] = a_grid_r    # rent to rent

    a_grid_exp_t0[4] = a_grid_o    # own to landlord
    a_grid_exp_t0[5] = a_grid_ll   # landlord to own
    a_grid_exp_t0[6] = a_grid_ll   # landlord to landlord

    a_grid_exp_t0[7] = a_grid_o    # own to ownF

    a_grid_exp_t0[8] = a_grid_of   # ownF to own
    a_grid_exp_t0[9] = a_grid_of   # ownF to ownF
    a_grid_exp_t0[10] = a_grid_of  # ownF to rent

    a_grid_exp_t0[11] = a_grid_ll2  # landlord to landlord2
    a_grid_exp_t0[12] = a_grid_ll  # landlord2 to landlord
    a_grid_exp_t0[13] = a_grid_ll2  # landlord2 to landlord2

    a_grid_exp_t0[14] = a_grid_r   # rent to rent*
    a_grid_exp_t0[15] = a_grid_ll  # landlord to landlord*
    a_grid_exp_t0[16] = a_grid_ll2  # landlord2 to landlord2*

    # housing services consumed in each transition: a house by default, a flat
    # for renters and flat owners (landlords live in a house and let flats)
    hsizes = np.ones([n_h]) * omegah
    hsizes[[1, 3, 10, 14]] = Fsize
    hsizes[[2, 7, 9]] = Fsize * omegah

    # Rent reset (stage 2): with probability pr_calvo a renter or landlord moves
    # to the market spot price, entering a starred state (14, 15, 16); otherwise
    # they carry last period's nominal rent, staying in 3, 6 or 13.
    h_markov = np.eye(n_h)

    # renters in previous period
    h_markov[3, 14] = pr_calvo
    h_markov[3, 3] = 1 - pr_calvo

    h_markov[14, 14] = pr_calvo
    h_markov[14, 3] = 1 - pr_calvo

    h_markov[10, 14] = pr_calvo
    h_markov[10, 3] = 1 - pr_calvo
    h_markov[10, 10] = 0

    h_markov[1, 14] = pr_calvo
    h_markov[1, 3] = 1 - pr_calvo
    h_markov[1, 1] = 0

    # landlord in previous period
    h_markov[6, 15] = pr_calvo
    h_markov[6, 6] = 1 - pr_calvo

    h_markov[15, 15] = pr_calvo
    h_markov[15, 6] = 1 - pr_calvo

    h_markov[4, 15] = pr_calvo
    h_markov[4, 6] = 1 - pr_calvo
    h_markov[4, 4] = 0

    h_markov[12, 15] = pr_calvo
    h_markov[12, 6] = 1 - pr_calvo
    h_markov[12, 12] = 0

    # landlord2 in previous period
    h_markov[13, 16] = pr_calvo
    h_markov[13, 13] = 1 - pr_calvo

    h_markov[16, 16] = pr_calvo
    h_markov[16, 13] = 1 - pr_calvo

    h_markov[11, 16] = pr_calvo
    h_markov[11, 13] = 1 - pr_calvo
    h_markov[11, 11] = 0

    # Rental price grid: three points, tightly spaced around the steady-state
    # rent.  Only the local slope of policies in the individual rent matters for
    # the first-order solution, so three points suffice; `pri_grid2` is the
    # degenerate counterpart for tenures that pay no rent.
    pri_grid = np.array([np.log(np.exp(prss) / (1 + 1E-4)), prss, prss + 1E-4])
    pri_grid2 = np.array([-1E-4, 0.0, 1E-4])

    # Full state-space grids, (n_h, n_z, n_a, n_pr), for Continuous2DFullGrid.
    # Broadcasting with `0 *` keeps the grid constant along dimensions it does
    # not vary over while giving it the full shape.
    a_grid_full = (np.zeros(n_z)[np.newaxis, :, np.newaxis, np.newaxis]
                   + a_grid_exp_t1[:, np.newaxis, :, np.newaxis]
                   + 0 * pri_grid[np.newaxis, np.newaxis, np.newaxis, :])

    pri_grid_full = np.zeros_like(a_grid_full)
    # renters and landlords carry an individual rent
    pri_grid_full[[1, 3, 4, 6, 10, 11, 12, 13, 14, 15, 16], :, :, :] = (
        pri_grid[np.newaxis, np.newaxis, np.newaxis, :] + np.zeros((n_z, n_a, 1)))
    # owner-occupiers do not
    pri_grid_full[[0, 2, 5, 7, 8, 9], :, :, :] = (
        pri_grid2[np.newaxis, np.newaxis, np.newaxis, :] + np.zeros((n_z, n_a, 1)))

    # beginning-of-period counterpart
    a_grid_full_t0 = (np.zeros(n_z)[np.newaxis, :, np.newaxis, np.newaxis]
                      + a_grid_exp_t0[:, np.newaxis, :, np.newaxis]
                      + 0 * pri_grid[np.newaxis, np.newaxis, np.newaxis, :])

    return z_grid, z_dist, z_markov, a_grid, a_grid_exp_t0, a_grid_exp_t1, a_grid_r, a_grid_o, a_grid_ll, a_grid_of, a_grid_full, hsizes, h_markov, n_z, pri_grid, pri_grid_full, a_grid_full_t0


################################################################################
## Hetinput: prices and income
################################################################################


def prices_baseline(r, ph, Fsize):
    """Mortgage rate and rental flat price in the baseline model.

    Mortgages are variable rate, so `rmort` tracks the safe rate one for one;
    the borrowing wedge is added in `labor_income`.
    """
    rmort = r * 1
    phF = np.log(np.exp(ph) * Fsize)     # price of a rental flat

    return rmort, phF


def labor_income(z_grid, z_dist, r, rmort, w, ph, phF, transac, pr, hours, Tax, Div, MPC,
                 a_grid_exp_t0, borwedge, Fsize, n_h, kappah, kappahLL, kappay, borlim,
                 hmcost, TransfAgg, pri_grid, pri_grid_full, pi):
    """Assemble cash on hand and the borrowing constraint for every transition.

    Implements the budget and borrowing constraints of Table 1: for each of the
    17 transitions, the housing costs incurred (`housecost`), the rent paid or
    received (`rentcost`), the interest rate faced (`rexpand`, which is the
    deposit rate on positive balances and the mortgage rate plus the borrowing
    wedge on negative ones), and the borrowing limit (`bchh`).

    The borrowing limit is the tighter of the LTV limit (`kappah` against the
    home, `kappahLL` against rental flats) and the LTI limit (`kappay` times
    disposable income).  Crucially, both apply only *at origination*: for
    transitions that do not change the housing position (own -> own, landlord ->
    landlord, ...) the constraint is `min(current assets, the limit)`, so a
    household that is already more indebted than the limit may stay there.  This
    is what makes mortgages long-term in the model (Section 3.1).

    Returns
    -------
    disinc      : disposable income, excluding housing costs
    coh         : cash on hand, including assets and housing costs
    y           : the non-asset part of the budget constraint (used by EGM)
    rexpand     : the state-contingent interest rate
    bchh        : the borrowing limit
    agridexpand : beginning-of-period assets, on the full state space
    rentcost    : rent paid (negative) or received (positive)
    """
    nz = z_grid.shape[0]
    na = a_grid_exp_t0.shape[1]

    # beginning-of-period assets on (n_h, n_z, n_a, n_pr)
    agridexpand = (np.zeros([nz, ])[np.newaxis, :, np.newaxis, np.newaxis]
                   + a_grid_exp_t0[:, np.newaxis, :, np.newaxis]
                   + 0 * pri_grid[np.newaxis, np.newaxis, np.newaxis, :])

    # state-contingent interest rate: borrowers pay the mortgage rate plus wedge
    rexpand = r * np.ones_like(agridexpand)
    rexpand[agridexpand < 0] = rmort + borwedge

    divi = np.exp(Div) * z_grid / (np.sum(z_grid * z_dist))   # dividends paid out in proportion to productivity

    labinc = z_grid * np.exp(w) * np.exp(hours) * (1 - Tax)   # post-tax labour income

    MPC = MPC * np.ones([n_h, ])    # lump-sum transfer used to compute iMPCs; zero in steady state

    # disposable income
    disinc = (labinc[np.newaxis, :, np.newaxis, np.newaxis]
              + divi[np.newaxis, :, np.newaxis, np.newaxis]
              + 0 * a_grid_exp_t0[:, np.newaxis, :, np.newaxis]
              + 0 * pri_grid[np.newaxis, np.newaxis, np.newaxis, :]
              + MPC[:, np.newaxis, np.newaxis, np.newaxis] + TransfAgg)

    # housing costs by transition: purchases, sales, transaction costs `transac`
    # and maintenance `hmcost` (column -C_h of Table 1)
    housecost = np.array([0 - hmcost, np.exp(ph) - transac - 0,                                      # o/o, o/r
                          -Fsize * np.exp(ph) - transac - Fsize * hmcost, -0,                        # r/oF, r/r
                          0 - np.exp(phF) - transac - hmcost * (1 + Fsize),
                          np.exp(phF) - transac - hmcost, 0 - hmcost * (1 + Fsize),                  # o/l, l/o, l/l
                          np.exp(ph) - Fsize * np.exp(ph) - 2 * transac - hmcost * Fsize,            # o/oF
                          -np.exp(ph) + Fsize * np.exp(ph) - 2 * transac - hmcost,
                          0 - hmcost * Fsize, Fsize * np.exp(ph) - transac - 0,                      # oF/o, oF/oF, oF/r
                          2 * 0 - np.exp(phF) - transac - hmcost * (1 + 2 * Fsize),
                          0 + np.exp(phF) - transac - hmcost * (1 + Fsize),
                          2 * 0 - hmcost * (1 + 2 * Fsize),                                          # l/l2, l2/l, l2/l2
                          -0, 0 - hmcost * (1 + Fsize), 2 * 0 - hmcost * (1 + 2 * Fsize)])           # r/r*, ll/ll*, ll2/ll2*

    # rent paid or received.  Households in a non-reset state carry last
    # period's nominal rent, deflated by current inflation; those in a starred
    # (reset) state or changing tenure transact at the spot price `pr`.
    rentcost = pri_grid_full * 0
    rentcost[[3], :, :, :] = -1 * np.exp(pri_grid_full[[3], :, :, :]) / (1 + pi)     # renters pay rent
    rentcost[[6], :, :, :] = np.exp(pri_grid_full[[6], :, :, :]) / (1 + pi)          # landlords receive rent
    rentcost[[13], :, :, :] = 2 * np.exp(pri_grid_full[[13], :, :, :]) / (1 + pi)    # two-flat landlords receive twice

    rentcost[[1, 10, 14], :, :, :] = -np.exp(pr)     # transact at the spot price
    rentcost[[4, 12, 15], :, :, :] = np.exp(pr)
    rentcost[[11, 16], :, :, :] = 2 * np.exp(pr)

    # non-asset part of the budget constraint
    y = (labinc[np.newaxis, :, np.newaxis, np.newaxis] + divi[np.newaxis, :, np.newaxis, np.newaxis]
         + housecost[:, np.newaxis, np.newaxis, np.newaxis]
         + MPC[:, np.newaxis, np.newaxis, np.newaxis] + rentcost + TransfAgg)

    # cash on hand today
    coh = (disinc + (1 + rexpand) * agridexpand
           + housecost[:, np.newaxis, np.newaxis, np.newaxis] + rentcost)

    # Borrowing constraint.  `np.maximum(-LTV, -LTI)` picks whichever of the two
    # limits binds first; the outer `np.minimum(agridexpand, .)` applies only to
    # transitions that do not re-originate the mortgage, letting existing
    # borrowers stay above their current limit.
    bchh = np.zeros_like(coh)
    bchh[0] = np.minimum(agridexpand[0], np.maximum(-np.exp(ph) * kappah, -disinc[0] * kappay))     # own to own
    bchh[1] = borlim                                                                                # own to rent
    bchh[2] = np.maximum(-Fsize * np.exp(ph) * kappah, -disinc[2] * kappay)                         # rent to ownF
    bchh[3] = borlim                                                                                # rent to rent

    bchh[4] = np.maximum(-np.exp(ph) * kappah - np.exp(phF) * kappahLL,
                         -np.exp(phF) * kappahLL - disinc[4] * kappay)                              # own to landlord
    bchh[5] = np.minimum(agridexpand[5] + np.exp(phF) - transac,
                         np.maximum(-np.exp(ph) * kappah, -disinc[5] * kappay))                     # landlord to own
    bchh[6] = np.minimum(agridexpand[6],
                         np.maximum(-np.exp(ph) * kappah - np.exp(phF) * kappahLL,
                                    -np.exp(phF) * kappahLL - disinc[6] * kappay))                  # landlord to landlord

    bchh[7] = np.maximum(-Fsize * np.exp(ph) * kappah, -disinc[7] * kappay)                         # own to ownF

    bchh[8] = np.maximum(-np.exp(ph) * kappah, -disinc[8] * kappay)                                 # ownF to own
    bchh[9] = np.minimum(agridexpand[9],
                         np.maximum(-Fsize * np.exp(ph) * kappah, -disinc[9] * kappay))             # ownF to ownF
    bchh[10] = borlim                                                                               # ownF to rent

    bchh[11] = np.maximum(-np.exp(ph) * kappah - np.exp(phF) * kappahLL,
                          -np.exp(phF) * kappahLL - disinc[11] * kappay)                            # landlord to landlord2
    bchh[12] = np.minimum(agridexpand[12] + np.exp(phF) - transac,
                          np.maximum(-np.exp(ph) * kappah - np.exp(phF) * kappahLL,
                                     -np.exp(phF) * kappahLL - disinc[12] * kappay))                # landlord2 to landlord
    bchh[13] = np.minimum(agridexpand[13],
                          np.maximum(-np.exp(ph) * kappah - np.exp(phF) * kappahLL,
                                     -np.exp(phF) * kappahLL - disinc[13] * kappay))                # landlord2 to landlord2

    bchh[14] = borlim                                                                               # rent to rent*
    bchh[15] = np.minimum(agridexpand[15],
                          np.maximum(-np.exp(ph) * kappah - np.exp(phF) * kappahLL,
                                     -np.exp(phF) * kappahLL - disinc[15] * kappay))                # landlord to landlord*
    bchh[16] = np.minimum(agridexpand[16],
                          np.maximum(-np.exp(ph) * kappah - np.exp(phF) * kappahLL,
                                     -np.exp(phF) * kappahLL - disinc[16] * kappay))                # landlord2 to landlord2*

    return disinc, coh, y, rexpand, bchh, agridexpand, rentcost


def hh_init(disinc, eis, beta, phihs, hsizes):
    """Initial guess for backward iteration: the value of consuming income forever."""
    V = util_c_KMV(disinc, hsizes[:, np.newaxis, np.newaxis, np.newaxis] * np.ones_like(disinc),
                   eis, phihs) / (1 - beta)

    Va = np.empty_like(V)
    Va = marg_util_c_KMV(disinc, hsizes[:, np.newaxis, np.newaxis, np.newaxis] * np.ones_like(disinc),
                         eis, phihs)

    return V, Va


################################################################################
## Stage 2 - Tenure choice
################################################################################


def util_l(V, coh, movecost, movecostLL, defaultcost, llcost, borlim, n_h, bchh):
    """Flow payoff of each housing transition, on `(n_h | n_h_prev, n_z, n_a, n_pr)`.

    Feeds the `LogitChoice` stage: adding this to the end-of-stage-2 value
    function and applying the Gumbel taste shock gives the transition
    probabilities of equation (2).  The fixed costs are the `eta(h)` of the
    paper: `movecost` for transactions in owner-occupied property, `movecostLL`
    for transactions in rental property, and `llcost` a flow utility cost of
    being a landlord.

    Entries left at `-inf` are transitions that do not exist (you cannot go from
    'rent -> rent' to 'own -> own' without first buying).  Setting the cost to
    infinity, rather than dropping the state, keeps the array rectangular.

    Three further groups of entries are overwritten:

    * transitions whose cash on hand cannot cover the borrowing constraint are
      ruled out, since they would imply negative consumption;
    * except that households with nowhere else to go may *default*, which is
      always feasible but carries the large utility cost `defaultcost`.  Default
      routes an owner or landlord into renting or into a smaller portfolio, at
      the borrowing constraint of the state they land in (Section 3.1).
    """
    flow_u = np.ones([n_h, n_h]) * -np.inf

    # Valid transitions
    flow_u[[0, 1, 4, 7], 0] = [0, -movecost, -movecostLL - llcost, -movecost]     # (x | O/O)
    flow_u[[2, 3], 1] = [-movecost, 0]                                            # (x | O/R)
    flow_u[[8, 9, 10], 2] = [-movecost, 0, -movecost]                             # (x | R/OF)
    flow_u[[3], 3] = [0]                                                          # (x | R/R)

    flow_u[[5, 6, 11], 4] = [-movecostLL, -llcost, -movecostLL - llcost]          # (x | O/LL)
    flow_u[[0, 1, 4, 7], 5] = [0, -movecost, -movecostLL - llcost, -movecost]     # (x | LL/O)
    flow_u[[6], 6] = [-llcost]                                                    # (x | LL/LL)

    flow_u[[8, 9, 10], 7] = [-movecost, 0, -movecost]                             # (x | O/OF)
    flow_u[[0, 1, 4, 7], 8] = [0, -movecost, -movecostLL - llcost, -movecost]     # (x | OF/O)
    flow_u[[8, 9, 10], 9] = [-movecost, 0, -movecost]                             # (x | OF/OF)
    flow_u[[2, 3], 10] = [-movecost, 0]                                           # (x | OF/R)

    flow_u[[12, 13], 11] = [-movecostLL - llcost, -llcost]                        # (x | LL/LL2)
    flow_u[[5, 6, 11], 12] = [-movecostLL, -llcost, -movecostLL - llcost]         # (x | LL2/LL)
    flow_u[[13], 13] = [-llcost]                                                  # (x | LL2/LL2)

    flow_u[[2, 14], 14] = [-movecost, 0]                                          # (x | R/R*)
    flow_u[[5, 15, 11], 15] = [-movecostLL, -llcost, -movecostLL - llcost]        # (x | LL/LL*)
    flow_u[[12, 16], 16] = [-movecostLL - movecost, -llcost]                      # (x | LL2/LL2*)

    # broadcast to (n_h | n_h_prev, n_z, n_a, n_pr)
    shape = np.zeros((n_h, n_h,) + V.shape[1:])
    flow_u = flow_u[..., np.newaxis, np.newaxis, np.newaxis] + shape

    # rule out transitions that cannot satisfy the borrowing constraint
    for i in range(n_h):
        flow_u[i, :, coh[i] - bchh[i] <= 0.0] = -np.inf

    # Default: a household with insufficient resources can default on the
    # property, incurring `defaultcost`, consuming its disposable income and
    # restarting at the borrowing constraint of the state it moves to.
    flow_u[1, 0, (coh[1] <= borlim)] = -defaultcost      # own/own to own/rent
    flow_u[1, 5, (coh[1] <= borlim)] = -defaultcost      # landlord/own to own/rent
    flow_u[1, 8, (coh[1] <= borlim)] = -defaultcost      # ownF/own to own/rent

    flow_u[5, 6, (coh[5] - bchh[5] <= 0.0)] = -defaultcost      # landlord/landlord to landlord/own
    flow_u[5, 15, (coh[5] - bchh[5] <= 0.0)] = -defaultcost     # landlord/landlord* to landlord/own
    flow_u[5, 4, (coh[5] - bchh[5] <= 0.0)] = -defaultcost      # own/landlord to landlord/own
    flow_u[5, 12, (coh[5] - bchh[5] <= 0.0)] = -defaultcost     # landlord2/landlord to landlord/own

    flow_u[10, 9, (coh[10] <= 0.0)] = -defaultcost      # ownF/ownF to ownF/rent
    flow_u[10, 2, (coh[10] <= 0.0)] = -defaultcost      # rent/ownF to ownF/rent
    flow_u[10, 7, (coh[10] <= 0.0)] = -defaultcost      # own/ownF to ownF/rent

    flow_u[12, 13, (coh[12] - bchh[12] <= 0.0)] = -defaultcost    # landlord2/landlord2 to landlord2/landlord
    flow_u[12, 11, (coh[12] - bchh[12] <= 0.0)] = -defaultcost    # landlord/landlord2 to landlord2/landlord
    flow_u[12, 16, (coh[12] - bchh[12] <= 0.0)] = -defaultcost    # landlord2/landlord2* to landlord2/landlord

    return flow_u


################################################################################
## Stage 3 - Consumption and saving
################################################################################


def dcegm(V, Va, coh, y, rexpand, beta, eis, a_grid_exp_t0, a_grid_exp_t1, disinc, bchh,
          hsizes, phihs, pr, pri_grid_full, pi):
    """Consumption-saving choice conditional on the housing decision, by DC-EGM.

    The endogenous grid point method of Carroll (2006), augmented as in Iskhakov
    et al. (2017) with an upper envelope step.  The extra step is needed because
    the discrete tenure choice makes the value function non-concave, so the
    Euler equation can have multiple solutions and the raw endogenous grid can
    be non-monotone; taking the upper envelope over the candidate solutions
    picks the optimum (Section 3.5).

    Steps: invert the Euler equation on the end-of-period grid to get
    consumption, back out the implied beginning-of-period assets, take the upper
    envelope onto the exogenous grid, impose the borrowing constraint, and
    update the marginal value.

    Also returns `pri`, the rent carried into next period: households in a
    non-reset state carry last period's nominal rent deflated by inflation,
    while those that reset or move take the spot price `pr`.
    """
    W = beta * V           # end-of-stage value function
    uc_endo = beta * Va    # envelope condition

    # Euler equation, inverted on the end-of-period grid
    c_endo = inv_marg_util_c_KMV(uc_endo,
                                 hsizes[:, np.newaxis, np.newaxis, np.newaxis] + np.zeros_like(uc_endo),
                                 eis, phihs)

    # beginning-of-period assets implied by that consumption
    a_endo = (c_endo + a_grid_exp_t1[:, np.newaxis, :, np.newaxis] - y) / (1 + rexpand)

    # interpolate with upper envelope, giving unconstrained values and policies
    V_uc, c_uc, a_uc = upperenv_vec(W, a_endo, coh, a_grid_exp_t0, a_grid_exp_t1, eis, hsizes, phihs)

    # enforce the borrowing constraint
    V, c, a = constrain_policies(V_uc, c_uc, a_uc, coh, eis, W, a_grid_exp_t1, bchh, disinc,
                                 hsizes, phihs)

    # update marginal value on the exogenous grid
    uc = marg_util_c_KMV(c, hsizes[:, np.newaxis, np.newaxis, np.newaxis] + np.zeros_like(c), eis, phihs)
    Va = (1 + rexpand) * uc

    # rent carried into next period
    pri = pri_grid_full * 0
    pri[[3, 6, 13], :, :, :] = np.log(np.exp(pri_grid_full[[3, 6, 13], :, :, :]) / (1 + pi))   # keep last period's rent
    pri[[1, 10, 14], :, :, :] = pr                                                             # move to the spot price
    pri[[4, 12, 15, 11, 16], :, :, :] = pr                                                     # move to the spot price

    return V, Va, a, c, pri


@njit(parallel=True)
def upperenv_vec(W, a_endo, coh, a_grid_exp_t0, a_grid_exp_t1, eis, hsizes, phihs):
    """Interpolate value and consumption from the endogenous grid onto the exogenous one.

    The upper envelope step of Iskhakov et al. (2017).  For each segment
    `[a_endo[ja], a_endo[ja+1]]` of the endogenous grid we interpolate onto every
    exogenous gridpoint the segment covers, and keep the candidate that delivers
    the highest value.  Where the endogenous grid folds back on itself -- which
    is exactly where the discrete tenure choice creates a non-concavity -- more
    than one segment covers the same exogenous point, and this comparison is
    what selects the optimal one.

    Note the two grids play different roles: `a_grid_exp_t0` is the
    beginning-of-period grid we evaluate today's policies on, while
    `a_grid_exp_t1` is the end-of-period grid that yesterday's saving decision
    landed on.
    """
    n_n, n_z, n_a, n_pr = W.shape
    a = np.zeros((n_n, n_z, n_a, n_pr))               # asset policy
    c = np.zeros((n_n, n_z, n_a, n_pr))               # consumption policy
    V = -np.inf * np.ones((n_n, n_z, n_a, n_pr))      # value function

    for i_n in prange(n_n):             # housing transition
        for i_z in prange(n_z):         # productivity
            for i_pr in prange(n_pr):   # rental price

                for ja in prange(n_a - 1):    # segments of the endogenous grid

                    a_low, a_high = a_endo[i_n, i_z, ja, i_pr], a_endo[i_n, i_z, ja + 1, i_pr]
                    W_low, W_high = W[i_n, i_z, ja, i_pr], W[i_n, i_z, ja + 1, i_pr]
                    ap_low, ap_high = a_grid_exp_t1[i_n, ja], a_grid_exp_t1[i_n, ja + 1]

                    for ia in range(n_a):    # exogenous asset grid, increasing

                        acur = a_grid_exp_t0[i_n, ia]
                        coh_cur = coh[i_n, i_z, ia, i_pr]

                        interp = (a_low <= acur <= a_high)
                        extrap = (ja == n_a - 2) and (acur > a_endo[i_n, i_z, n_a - 1, i_pr])
                        extrap_low = (ja == 0) and (acur < a_endo[i_n, i_z, 0, i_pr])

                        # exploit that a_grid is increasing: once past this
                        # segment there is nothing left for it to cover
                        if (a_high < acur < a_endo[i_n, i_z, n_a - 1, i_pr]):
                            break

                        if interp or extrap or extrap_low:
                            # a0, c0, W0: the end-of-period assets, consumption and
                            # continuation value chosen by a household holding
                            # `acur` today, according to this segment
                            W0 = interpolate.interpolate_point(acur, a_low, a_high, W_low, W_high)
                            a0 = interpolate.interpolate_point(acur, a_low, a_high, ap_low, ap_high)
                            c0 = coh_cur - a0

                            V0 = util_c_KMV(c0, hsizes[i_n], eis, phihs) + W0

                            # upper envelope: keep this candidate if it beats
                            # whatever an earlier segment proposed
                            if V0 > V[i_n, i_z, ia, i_pr]:
                                a[i_n, i_z, ia, i_pr] = a0
                                c[i_n, i_z, ia, i_pr] = c0
                                V[i_n, i_z, ia, i_pr] = V0

    return V, c, a


@njit
def constrain_policies(V, c, a_x, coh, eis, W, a_grid_exp_t1, bchh, y, hsizes, phihs):
    """Impose the borrowing constraint on the unconstrained DC-EGM policies.

    Households whose unconstrained saving choice lies below their limit are put
    on the limit and consume the remainder of cash on hand.  If that remainder
    is still non-positive the household defaults: it consumes the disposable
    income of the 'rent to rent' state (index 3), the outside option that
    default routes it to.  The corresponding utility cost is applied in
    `util_l`, not here.
    """
    for i_n in range(len(V[:, 0, 0, 0])):           # housing transition
        for i_z in range(len(V[0, :, 0, 0])):       # productivity
            for i_pr in range(len(V[0, 0, 0, :])):  # rental price

                W0 = np.interp(bchh[i_n, i_z, :, i_pr], a_grid_exp_t1[i_n], W[i_n, i_z, :, i_pr])

                tbc = (a_x[i_n, i_z, :, i_pr] < bchh[i_n, i_z, :, i_pr])   # to become constrained

                W[i_n, i_z, :, i_pr][tbc] = W0[tbc]

                a_x[i_n, i_z, :, i_pr][tbc] = bchh[i_n, i_z, :, i_pr][tbc]

                c[i_n, i_z, :, i_pr][tbc] = coh[i_n, i_z, :, i_pr][tbc] - a_x[i_n, i_z, :, i_pr][tbc]

                default = (c[i_n, i_z, :, i_pr] <= 0)

                c[i_n, i_z, :, i_pr][default] = y[3, i_z, i_pr][default]

                V[i_n, i_z, :, i_pr][tbc] = (util_c_KMV(c[i_n, i_z, :, i_pr][tbc], hsizes[i_n], eis, phihs)
                                             + W[i_n, i_z, :, i_pr][tbc])

                V[i_n, i_z, :, i_pr][default] = (util_c_KMV(c[i_n, i_z, :, i_pr][default], hsizes[i_n], eis, phihs)
                                                 + W[i_n, i_z, :, i_pr][default])

    return V, c, a_x


################################################################################
## Hetoutputs: aggregation
################################################################################


def labor_supply(c, eis, z_grid, hours, w, Tax, frisch, hsizes, phihs):
    """Household-level term in the wage Phillips curve, equation (8).

    The union sets wages off the average marginal rate of substitution across
    households, so heterogeneity feeds into labour supply.
    """
    swn = np.exp(hours) ** (1 / frisch)
    swd = (marg_util_c_KMV(c, hsizes[:, np.newaxis, np.newaxis, np.newaxis] + np.zeros_like(c), eis, phihs)
           * z_grid[np.newaxis, :, np.newaxis, np.newaxis] * np.exp(w) * (1 - Tax))
    sw = swn / swd

    return sw


def housingDemand(c, a, Fsize, pr, phss, agridexpand, coh, pri):
    """Indicator arrays that aggregate into housing quantities and tenure shares.

    Each array marks the transitions counting towards a given aggregate, so that
    integrating it against the distribution gives the aggregate.  These supply
    the housing and rental market clearing conditions (equations 3 and 4), the
    activity measures compared with the SVAR, and the consumption series split
    by tenure that appear in Figure 6.

    The consumption series are additionally filtered by productivity band and by
    the sign of assets, which controls for composition: `cmort` picks indebted
    owner-occupiers, `coo` outright owners, `crent` renters and `cll` landlords,
    each within a comparable income range.  The matching `*wgt` arrays count the
    households included, so a population average is `c<type> / c<type>wgt`.
    """
    hoo = np.zeros_like(a)     # owner-occupier of a house
    hr = np.zeros_like(a)      # renter
    hll = np.zeros_like(a)     # landlord with one flat
    hooF = np.zeros_like(a)    # owner-occupier of a flat
    hll2 = np.zeros_like(a)    # landlord with two flats
    prw = np.zeros_like(a)     # rent paid, for the rental price index
    hd = np.ones_like(a)       # housing demand, in units of housing

    hoo[0] = 1     # own/own
    hoo[5] = 1     # landlord/own
    hoo[8] = 1     # ownF/own

    hr[1] = 1      # own/rent
    hr[3] = 1      # rent/rent
    hr[10] = 1     # ownF/rent
    hr[14] = 1     # rent/rent*

    hll[4] = 1     # own/landlord
    hll[6] = 1     # landlord/landlord
    hll[12] = 1    # landlord2/landlord
    hll[15] = 1    # landlord/landlord*

    hll2[11] = 1   # landlord/landlord2
    hll2[13] = 1   # landlord2/landlord2
    hll2[16] = 1   # landlord2/landlord2*

    hooF[7] = 1    # own/ownF
    hooF[2] = 1    # rent/ownF
    hooF[9] = 1    # ownF/ownF

    # housing transactions, for the transaction cost rebate; two for moves that
    # both buy and sell a property
    htrans = np.zeros_like(a)
    htrans[[1, 2, 4, 5, 10, 11, 12]] = 1
    htrans[[7, 8]] = 2

    # sales volumes, compared with the SVAR evidence of Figure 3
    hsv = np.zeros_like(a)
    hsv[[1, 5, 7, 8, 10, 12]] = 1

    # landlord exit rate, a steady-state calibration target (Table 2)
    llexit = np.zeros_like(a)
    llexit[[5, 12]] = 1

    # owner exit rate, a steady-state calibration target (Table 2)
    ownexit = np.zeros_like(a)
    ownexit[1] = 1
    ownexit[10] = 1

    # rent actually paid by each renter, which aggregates to the rental price index
    prw[3] = np.exp(pri[3])
    prw[1] = np.exp(pri[1])
    prw[14] = np.exp(pri[14])
    prw[10] = np.exp(pri[10])

    # housing demanded: a flat for renters and flat owners, a house otherwise
    hd[[1, 3, 10, 14]] = Fsize
    hd[[2, 7, 9]] = Fsize

    hlltot = hll + hll2 * 2    # total flats let

    # Consumption by household type.  Filtering by productivity band as well as
    # tenure controls for compositional differences across groups.
    cmort = c * 0
    cmort[[9], 6:9, :, :] = c[[9], 6:9, :, :]
    cmort[[0], 9:12, :, :] = c[[0], 9:12, :, :]
    cmort[agridexpand > 0] = 0            # mortgagors: negative assets only
    cmortwgt = 1 * (cmort > 0)

    coo = c * 0
    coo[[9], 6:9, :, :] = c[[9], 6:9, :, :]
    coo[[0], 9:12, :, :] = c[[0], 9:12, :, :]
    coo[agridexpand < 0] = 0              # outright owners: positive assets only
    coowgt = 1 * (coo > 0)

    crent = c * 0
    crent[[3, 14], 0:5, :, :] = c[[3, 14], 0:5, :, :]
    crentwgt = 1 * (crent > 0)

    cll = c * 0
    cll[[6, 13, 15, 16], 12:, :, :] = c[[6, 13, 15, 16], 12:, :, :]
    cllwgt = 1 * (cll > 0)

    # disposable income of the same groups
    ydis_mort = (coh - agridexpand) * cmortwgt
    ydis_oo = (coh - agridexpand) * coowgt
    ydis_rent = (coh - agridexpand) * crentwgt
    ydis_ll = (coh - agridexpand) * cllwgt

    return hoo, hr, hll, hooF, htrans, hsv, llexit, ownexit, hll2, prw, hd, hlltot, crent, coo, cmort, crentwgt, coowgt, cmortwgt, cll, cllwgt, ydis_ll, ydis_mort, ydis_oo, ydis_rent


def assetDemand(a, c, coh, bchh, a_grid_full_t0, rexpand):
    """Defaults and interest flows, which enter the government budget constraint.

    `addefault` is the shortfall written off when a household defaults,
    `mortpmt` interest paid on mortgages and `deppmt` interest received on
    deposits.  The government absorbs defaults and the borrowing wedge
    (equation 11).
    """
    # amount written off in default
    addefault = np.zeros_like(a)
    addefault = ((coh - bchh) - a - c) * (a + c > (coh - bchh))

    # interest paid on borrowing
    mortpmt = np.zeros_like(a)
    mortpmt[a_grid_full_t0 < 0] = -(a_grid_full_t0[a_grid_full_t0 < 0]) * rexpand[a_grid_full_t0 < 0]

    # interest received on deposits
    deppmt = np.zeros_like(a)
    deppmt[a_grid_full_t0 > 0] = (a_grid_full_t0[a_grid_full_t0 > 0]) * rexpand[a_grid_full_t0 > 0]

    return addefault, mortpmt, deppmt


################################################################################################
## NOT USED BY Main.ipynb
##
## The functions below implement the extensions of Sections 4.4 and 4.5 of the
## paper.  They are kept here for reference and are not part of the baseline
## replication; substituting them into the stage block would reproduce the
## corresponding panels of Figures 9 and 10.  See also the matching section at
## the end of `dynamics.py`.
################################################################################################


def prices_mort(ph, Fsize):
    """Hetinput for the sticky mortgage rate extension (Section 4.5.3).

    Replaces `prices_baseline`.  `rmort` is no longer pinned to the safe rate
    here; it is determined by the `mortrate` block in `dynamics.py`, which
    averages the policy rate over the fixation period.
    """
    phF = np.log(np.exp(ph) * Fsize)     # rental flat price

    return phF


def prices_seg(r):
    """Hetinput for the segmented housing market extension (Section 4.5.1).

    Replaces `prices_baseline`.  `phF` is supplied separately because flats and
    houses trade in segmented markets and so no longer have proportional prices.
    """
    rmort = r * 1

    return rmort


def selling_friction(etasell, n_h):
    """Markov matrix for the search-and-matching extension (Section 4.5.4).

    An intended sale succeeds with probability `etasell`; otherwise the
    household stays in a fallback state where the sale did not happen.  Set
    `etasell = 1` (the baseline calibration) to switch the friction off.
    """
    sell_markov = np.eye(n_h, dtype=float)

    # intended selling state -> fallback state if the sale fails
    sell_map = {
        1: 0,    # own -> rent      : remain own -> own
        5: 15,   # landlord -> own  : remain landlord -> landlord*
        7: 0,    # own -> ownF      : remain own -> own
        8: 9,    # ownF -> own      : remain ownF -> ownF
        10: 9,   # ownF -> rent     : remain ownF -> ownF
        12: 16,  # landlord2 -> landlord : remain landlord2 -> landlord2*
    }

    # realised next state is the intended one with probability etasell and the
    # fallback with probability 1 - etasell
    for intended, fallback in sell_map.items():
        if 0 <= intended < n_h and 0 <= fallback < n_h:
            sell_markov[intended, :] = 0.0
            sell_markov[intended, intended] = float(etasell)
            sell_markov[intended, fallback] = float(1.0 - etasell)

    return sell_markov


def labor_income_RELL(z_grid, z_dist, r, rmort, w, ph, phF, transac, pr, hours, Tax, Div, MPC,
                      a_grid_exp_t0, borwedge, Fsize, n_h, kappah, kappahLL, kappay, borlim,
                      hmcost, TransfAgg, pri_grid, pri_grid_full, pi, rLL, prLL, phLL):
    """Variant of `labor_income` giving landlords their own prices (Section 4.4).

    Identical to `labor_income` except that the landlord states face `rLL`,
    `prLL` and `phLL` rather than `r`, `pr` and `ph`.  Feeding landlords a
    rational-expectations price path while other households keep behavioural
    expectations is how the counterfactual commercial rental sector of Figure 9
    is constructed.
    """
    phFLL = np.log(np.exp(phLL) * Fsize)     # rental flat price faced by landlords

    nz = z_grid.shape[0]
    na = a_grid_exp_t0.shape[1]

    agridexpand = (np.zeros([nz, ])[np.newaxis, :, np.newaxis, np.newaxis]
                   + a_grid_exp_t0[:, np.newaxis, :, np.newaxis]
                   + 0 * pri_grid[np.newaxis, np.newaxis, np.newaxis, :])

    # state-contingent interest rate
    rexpand = r * np.ones_like(agridexpand)
    rexpand[agridexpand < 0] = rmort + borwedge

    # landlord states get their own rate, accommodating different expectations
    LLints = [5, 6, 11, 12, 13, 15, 16]
    for i in LLints:
        rexpand[i][agridexpand[i] < 0] = rLL + borwedge
        rexpand[i][agridexpand[i] >= 0] = rLL

    divi = np.exp(Div) * z_grid / (np.sum(z_grid * z_dist))
    labinc = z_grid * np.exp(w) * np.exp(hours) * (1 - Tax)

    MPC = MPC * np.ones([n_h, ])

    disinc = (labinc[np.newaxis, :, np.newaxis, np.newaxis]
              + divi[np.newaxis, :, np.newaxis, np.newaxis]
              + 0 * a_grid_exp_t0[:, np.newaxis, :, np.newaxis]
              + 0 * pri_grid[np.newaxis, np.newaxis, np.newaxis, :]
              + MPC[:, np.newaxis, np.newaxis, np.newaxis] + TransfAgg)

    housecost = np.array([0 - hmcost, np.exp(ph) - transac - 0,                                      # o/o, o/r
                          -Fsize * np.exp(ph) - transac - Fsize * hmcost, -0,                        # r/oF, r/r
                          0 - np.exp(phFLL) - transac - hmcost * (1 + Fsize),
                          np.exp(phFLL) - transac - hmcost, 0 - hmcost * (1 + Fsize),                # o/l, l/o, l/l
                          np.exp(ph) - Fsize * np.exp(ph) - 2 * transac - hmcost * Fsize,            # o/oF
                          -np.exp(ph) + Fsize * np.exp(ph) - 2 * transac - hmcost,
                          0 - hmcost * Fsize, Fsize * np.exp(ph) - transac - 0,                      # oF/o, oF/oF, oF/r
                          2 * 0 - np.exp(phFLL) - transac - hmcost * (1 + 2 * Fsize),
                          0 + np.exp(phFLL) - transac - hmcost * (1 + Fsize),
                          2 * 0 - hmcost * (1 + 2 * Fsize),                                          # l/l2, l2/l, l2/l2
                          -0, 0 - hmcost * (1 + Fsize), 2 * 0 - hmcost * (1 + 2 * Fsize)])           # r/r*, ll/ll*, ll2/ll2*

    rentcost = pri_grid_full * 0
    rentcost[[3], :, :, :] = -1 * np.exp(pri_grid_full[[3], :, :, :]) / (1 + pi)     # renters pay rent
    rentcost[[6], :, :, :] = np.exp(pri_grid_full[[6], :, :, :]) / (1 + pi)          # landlords receive rent
    rentcost[[13], :, :, :] = 2 * np.exp(pri_grid_full[[13], :, :, :]) / (1 + pi)

    rentcost[[1, 10, 14], :, :, :] = -np.exp(pr)         # renters transact at the spot price
    rentcost[[4, 12, 15], :, :, :] = np.exp(prLL)        # landlords at their own spot price
    rentcost[[11, 16], :, :, :] = 2 * np.exp(prLL)

    y = (labinc[np.newaxis, :, np.newaxis, np.newaxis] + divi[np.newaxis, :, np.newaxis, np.newaxis]
         + housecost[:, np.newaxis, np.newaxis, np.newaxis]
         + MPC[:, np.newaxis, np.newaxis, np.newaxis] + rentcost + TransfAgg)

    coh = (disinc + (1 + rexpand) * agridexpand
           + housecost[:, np.newaxis, np.newaxis, np.newaxis] + rentcost)

    bchh = np.zeros_like(coh)
    bchh[0] = np.minimum(agridexpand[0], np.maximum(-np.exp(ph) * kappah, -disinc[0] * kappay))     # own to own
    bchh[1] = borlim                                                                                # own to rent
    bchh[2] = np.maximum(-Fsize * np.exp(ph) * kappah, -disinc[2] * kappay)                         # rent to ownF
    bchh[3] = borlim                                                                                # rent to rent

    bchh[4] = np.maximum(-np.exp(phLL) * kappah - np.exp(phFLL) * kappahLL,
                         -np.exp(phFLL) * kappahLL - disinc[4] * kappay)                            # own to landlord
    bchh[5] = np.minimum(agridexpand[5] + np.exp(phFLL) - transac,
                         np.maximum(-np.exp(phLL) * kappah, -disinc[5] * kappay))                   # landlord to own
    bchh[6] = np.minimum(agridexpand[6],
                         np.maximum(-np.exp(phLL) * kappah - np.exp(phFLL) * kappahLL,
                                    -np.exp(phFLL) * kappahLL - disinc[6] * kappay))                # landlord to landlord

    bchh[7] = np.maximum(-Fsize * np.exp(ph) * kappah, -disinc[7] * kappay)                         # own to ownF

    bchh[8] = np.maximum(-np.exp(ph) * kappah, -disinc[8] * kappay)                                 # ownF to own
    bchh[9] = np.minimum(agridexpand[9],
                         np.maximum(-Fsize * np.exp(ph) * kappah, -disinc[9] * kappay))             # ownF to ownF
    bchh[10] = borlim                                                                               # ownF to rent

    bchh[11] = np.maximum(-np.exp(phLL) * kappah - np.exp(phFLL) * kappahLL,
                          -np.exp(phFLL) * kappahLL - disinc[11] * kappay)                          # landlord to landlord2
    bchh[12] = np.minimum(agridexpand[12] + np.exp(phFLL) - transac,
                          np.maximum(-np.exp(phLL) * kappah - np.exp(phFLL) * kappahLL,
                                     -np.exp(phFLL) * kappahLL - disinc[12] * kappay))              # landlord2 to landlord
    bchh[13] = np.minimum(agridexpand[13],
                          np.maximum(-np.exp(phLL) * kappah - np.exp(phFLL) * kappahLL,
                                     -np.exp(phFLL) * kappahLL - disinc[13] * kappay))              # landlord2 to landlord2

    bchh[14] = borlim                                                                               # rent to rent*
    bchh[15] = np.minimum(agridexpand[15],
                          np.maximum(-np.exp(phLL) * kappah - np.exp(phFLL) * kappahLL,
                                     -np.exp(phFLL) * kappahLL - disinc[15] * kappay))              # landlord to landlord*
    bchh[16] = np.minimum(agridexpand[16],
                          np.maximum(-np.exp(phLL) * kappah - np.exp(phFLL) * kappahLL,
                                     -np.exp(phFLL) * kappahLL - disinc[16] * kappay))              # landlord2 to landlord2*

    return disinc, coh, y, rexpand, bchh, agridexpand, rentcost


def dcegm_RELL(V, Va, coh, y, rexpand, beta, eis, a_grid_exp_t0, a_grid_exp_t1, disinc, bchh,
               hsizes, phihs, pr, pri_grid_full, pi, prLL):
    """Variant of `dcegm` for the landlord-expectations counterfactual (Section 4.4).

    Identical to `dcegm` except that landlord states move to the landlord spot
    rent `prLL` rather than the household spot rent `pr`.
    """
    W = beta * V
    uc_endo = beta * Va

    c_endo = inv_marg_util_c_KMV(uc_endo,
                                 hsizes[:, np.newaxis, np.newaxis, np.newaxis] + np.zeros_like(uc_endo),
                                 eis, phihs)

    a_endo = (c_endo + a_grid_exp_t1[:, np.newaxis, :, np.newaxis] - y) / (1 + rexpand)

    V_uc, c_uc, a_uc = upperenv_vec(W, a_endo, coh, a_grid_exp_t0, a_grid_exp_t1, eis, hsizes, phihs)

    V, c, a = constrain_policies(V_uc, c_uc, a_uc, coh, eis, W, a_grid_exp_t1, bchh, disinc,
                                 hsizes, phihs)

    uc = marg_util_c_KMV(c, hsizes[:, np.newaxis, np.newaxis, np.newaxis] + np.zeros_like(c), eis, phihs)
    Va = (1 + rexpand) * uc

    pri = pri_grid_full * 0
    pri[[3, 6, 13], :, :, :] = np.log(np.exp(pri_grid_full[[3, 6, 13], :, :, :]) / (1 + pi))
    pri[[1, 10, 14], :, :, :] = pr        # households move to the spot price
    pri[[4, 12, 15, 11, 16], :, :, :] = prLL   # landlords move to their own spot price

    return V, Va, a, c, pri
