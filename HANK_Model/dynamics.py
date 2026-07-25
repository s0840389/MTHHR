"""Aggregate blocks of the dynamic model.

These are the `sequence_jacobian` blocks that close the model around the
household block of `household.py`: production, the price and wage Phillips
curves, the Taylor rule, the fiscal rule, the CPI index, and market clearing.
They correspond to Section 3.4 of the paper.

Main.ipynb assembles the baseline model from

    hh, firm, fisher, monetary, fiscal, mkt_clearing, wagepc, nkpc,
    rentindexB, cpiindex

solved for the unknowns `r, w, Y, Tax, pr, ph, rentindex` against the targets
`asset_mkt, fisher_res, realwage_res, Tax_res, rental_mkt, housing_mkt,
rentindex_res`.

Blocks below the 'NOT USED BY Main.ipynb' banner implement the extensions of
Sections 4.4 and 4.5 and are not part of the baseline replication.
"""

import numpy as np

from sequence_jacobian import simple, solved


###############################################################################################
## Dynamic blocks
###############################################################################################


@simple
def mkt_clearing(Y, C, HR, HLL, HOO, A, B, Hbar, Fsize, HA_supply, G, HOOF, HA_com, ph, pr,
                 HLL2, hmcost, Fcm, rentindex, PRW, HbarF, epsH, COO, COOWGT, CRENT, CRENTWGT,
                 CMORT, CMORTWGT, CLL, CLLWGT, YDIS_OO, YDIS_RENT, YDIS_LL, YDIS_MORT):
    """Market clearing, plus the tenure-level consumption aggregates of Figure 6.

    Housing supply is fixed at `Hbar` (Section 3.2), so the house price `ph`
    must move to clear equation (3), and the rental price `pr` to clear
    equation (4).  Rental supply is the sum of the fixed housing association
    stock `HA_supply`, any commercial sector `HA_com`, and the flats let by
    private landlords `HLL + 2 * HLL2` -- private landlords are therefore the
    marginal supplier of rental housing.

    `rentindex_res` defines the average rent actually paid, `PRW / HR`, which
    differs from the spot price `pr` because most renters are still on an old
    contract.
    """
    asset_mkt = A - B

    housing_mkt = Fsize * epsH + Hbar - Fsize * (HR + HOOF) - HOO - HLL - HLL2   # housing supply equals housing demand
    flat_mkt = 0
    goods_mkt = C + G + (Hbar + epsH * Fsize) * hmcost + Fcm * HA_com - Y        # goods market clearing

    rental_mkt = HA_com + (HA_supply + epsH) + HLL + 2 * HLL2 - HR               # rental supply equals rental demand

    rentindex_res = rentindex - PRW / HR

    # consumption by tenure (population averages: total over the count included)
    COObar = COO / COOWGT
    CRENTbar = CRENT / CRENTWGT
    CMORTbar = CMORT / CMORTWGT
    CLLbar = CLL / CLLWGT

    # disposable income by tenure
    YDIS_OObar = YDIS_OO / COOWGT
    YDIS_RENTbar = YDIS_RENT / CRENTWGT
    YDIS_LLbar = YDIS_LL / CLLWGT
    YDIS_MORTbar = YDIS_MORT / CMORTWGT

    return asset_mkt, rental_mkt, housing_mkt, goods_mkt, rentindex_res, flat_mkt, COObar, CRENTbar, CMORTbar, CLLbar, YDIS_OObar, YDIS_RENTbar, YDIS_LLbar, YDIS_MORTbar


@simple
def firm(Y, Z, w, alphah, NEss, prcom_prof):
    """Production and dividends, equation (6).

    Linear in labour services at the baseline calibration (`alphah = 1`).
    Dividends are paid out to households in proportion to labour productivity.
    """
    hours = ((Y / Z) ** (1 / alphah) / NEss).apply(np.log)
    Div = (Y - w.apply(np.exp) * hours.apply(np.exp) * NEss + prcom_prof).apply(np.log)
    SP = w.apply(np.exp) / (alphah * Y / (hours.apply(np.exp) * NEss))

    return hours, Div, SP


@solved(unknowns={'pi': 0}, targets=['pi_res'], solver="brentq")
def nkpc(pi, beta, SP, SPss, kappa, r):
    """Price Phillips curve, equation (7). `SP` is the real marginal cost."""
    pi_res = (1) / (1 + r(+1)) * pi(+1) + kappa * (SP / SPss).apply(np.log) - pi

    return pi_res


@solved(unknowns={'i': 0.01}, targets=['i_res'], solver="brentq")
def monetary(i, rstar, phi, pi_cpi, rhom, epsr, Y, Yss, phiy):
    """Smoothed Taylor rule, equation (9).

    Note the rule responds to total CPI inflation `pi_cpi`, which includes
    rents, not to goods inflation alone.
    """
    ygap = (Y - Yss) / Yss
    i_res = (rstar + pi_cpi * phi + phiy * ygap) * (1 - rhom) + (1 + i(-1)) * rhom + epsr - i

    return i_res, ygap


@simple
def fisher(r, pi, i, piw, w):
    """Fisher equation and the real wage identity.

    Deposits and mortgages are nominal promises, so the ex-post real rate is
    `(1 + i_{t-1}) / (1 + pi_t)`; unexpected inflation therefore redistributes
    between borrowers and savers.
    """
    fisher_res = 1 + r - (1 + i(-1)) / (1 + pi)
    realwage_res = piw - pi - (w - w(-1))

    return fisher_res, realwage_res


@simple
def rentindexB(rentindex):
    """Lag of the rental price index, needed by the CPI."""
    prlag = rentindex(-1).apply(np.log)

    return prlag


@solved(unknowns={'B': 2.7}, targets=['B_res'], solver="brentq")
def fiscal(Tax, B, r, G, gammatax, Taxss, Bss, Yss, w, hours, NEss, rentindex, HA_supply,
           transac, HTRANS, MORTPMT, DEPPMT, TransfAgg, hmcost, Fsize, ADDEFAULT, epsH):
    """Government budget constraint and tax rule, equations (10) and (11).

    The labour tax adjusts slowly towards the debt target with elasticity
    `gammatax`.  The government absorbs household defaults `ADDEFAULT`, receives
    the borrowing wedge and the rents on its housing association stock, and
    rebates housing transaction costs so they have no output implications.
    """
    B_res = (B(-1) + G + TransfAgg - Tax * w.apply(np.exp) * NEss * hours.apply(np.exp)
             - MORTPMT + DEPPMT + (HA_supply + epsH) * Fsize * hmcost + ADDEFAULT
             - rentindex * (HA_supply + epsH) - HTRANS * transac - B)
    Tax_res = Taxss + gammatax * (B(-1) - Bss) / Yss - Tax

    return B_res, Tax_res


@solved(unknowns={'piw': 0}, targets=['piw_res'], solver="brentq")
def wagepc(piw, beta, SW, SWss, kappaw):
    """Wage Phillips curve, equation (8).

    `SW` is the average marginal rate of substitution across households,
    aggregated by the `labor_supply` hetoutput, so household heterogeneity
    enters wage setting.
    """
    piw_res = beta * piw(+1) + kappaw * (SW / SWss).apply(np.log) - piw,

    return piw_res


@simple
def cpiindex(pi, rentindex, rentweight):
    """Total CPI inflation: goods inflation and rent inflation, weighted by the
    steady-state share of rental expenditure (footnote 14)."""
    pi_cpi = (1 - rentweight) * pi + rentweight * (rentindex / rentindex(-1) - 1 + pi)

    return pi_cpi


################################################################################################
## NOT USED BY Main.ipynb
##
## The blocks below implement the counterfactual and extensions of Sections 4.4
## and 4.5 of the paper.  They are kept here for reference and are not part of
## the baseline replication; swapping them into the block list would reproduce
## the corresponding panels of Figures 9 and 10.  See also the matching section
## at the end of `household.py`.
################################################################################################


@simple
def nkpc_simple(pi, beta, SP, SPss, kappa, r):
    """Unsolved version of `nkpc`, for variants that need `pi` as a model unknown."""
    pi_res = (1) / (1 + r(+1)) * pi(+1) + kappa * (SP / SPss).apply(np.log) - pi

    return pi_res


@simple
def block_RELL(pr, r, ph):
    """Pass-through of prices to landlords (Section 4.4).

    In the baseline landlords face the same prices as everyone else, which is
    what this block imposes.  Replacing it lets landlords face a different
    (e.g. rational-expectations) price path, which together with
    `labor_income_RELL` and `dcegm_RELL` builds the commercial-rental-sector
    counterfactual of Figure 9.
    """
    prLL = 1 * pr
    phLL = 1 * ph
    rLL = 1 * r

    return prLL, phLL, rLL


##############################################################################################################
## Commercial rental sector (Section 4.4)
##############################################################################################################


@solved(unknowns={'vr1': 3.94, 'vr2': 3.94, 'vr3': 15.53}, targets=['vr1_res', 'vr2_res', 'vr3_res'],
        solver="broyden")
def rentstar(ph, r, Fsize, hmcost, pr_calvo, vr1, vr2, vr3, pi):
    """Present-value recursions for the optimal reset rent of a commercial landlord.

    `vr1` discounts the real rent stream over the expected contract length,
    `vr2` the running costs, and `vr3` the resale value of the flat when the
    contract ends.
    """
    vr1_res = 1 + ((1 - pr_calvo) / ((1 + r(+1)) * (1 + pi(+1)))) * vr1(+1) - vr1
    vr2_res = 1 + ((1 - pr_calvo) / ((1 + r(+1)))) * vr2(+1) - vr2
    vr3_res = ((pr_calvo / (1 + r(+1))) * ph(+1).apply(np.exp) * Fsize
               + ((1 - pr_calvo) / (1 + r(+1))) * vr3(+1) - vr3)

    return vr1_res, vr2_res, vr3_res


@simple
def rentsector(ph, r, HA_com, Fsize, hmcost, vr1, vr2, vr3, rentindex, Fcm, pr_calvo):
    """Rent set by an unconstrained commercial rental sector with rational expectations.

    The counterfactual of Section 4.4: rather than being the marginal decision
    of a credit-constrained, behavioural individual landlord, the rent is a
    forward-looking asset price that fully internalises expected capital gains.
    This is what roughly doubles the output sacrifice ratio.

    Caveat: `prcom_prof` uses the economy-wide `rentindex`, whereas the average
    rent collected by the commercial sector specifically could differ from it.
    """
    pr = ((Fsize * ph.apply(np.exp) + (Fcm + Fsize * hmcost) * vr2 - vr3) / vr1).apply(np.log)

    new_HA = HA_com - HA_com(-1) * (1 - pr_calvo)

    # realised profits
    prcom_prof = (HA_com * (rentindex - Fcm - hmcost * Fsize)
                  + (HA_com(-1) * pr_calvo - new_HA) * ph.apply(np.exp) * Fsize)

    return pr, prcom_prof


################################################################################################
## Endogenous housing supply (Section 4.5.2)
################################################################################################


@solved(unknowns={'Hbar': 0.859}, targets=['Hbar_res'], solver="brentq")
def houseinvest_IH(Lbar, alpha_ih, deltahstar, ph, Hbar):
    """Housing investment against a fixed factor `Lbar`, and the housing stock's
    law of motion, which replaces the fixed-supply assumption of Section 3.2."""
    Ih = (alpha_ih * ph.apply(np.exp)) ** ((alpha_ih) / (1 - alpha_ih)) * Lbar   # investment in housing

    Hbar_res = (Hbar - Hbar(-1)) + deltahstar * Hbar - Ih   # law of motion, i.e. housing market clearing

    return Hbar_res, Ih


@simple
def mkt_clearing_IH(Hbarss, Y, C, HR, HLL, HOO, A, B, Hbar, Fsize, HA_supply, G, HOOF, HA_com,
                    ph, pr, HLL2, hmcost, Fcm, rentindex, PRW, HbarF, epsH, COO, COOWGT, CRENT,
                    CRENTWGT, CMORT, CMORTWGT, CLL, CLLWGT, YDIS_OO, YDIS_RENT, YDIS_LL,
                    YDIS_MORT, Ih, Ihss, phss, alpha_ih):
    """`mkt_clearing` with housing investment added to goods market clearing."""
    asset_mkt = A - B

    housing_mkt = Fsize * epsH + Hbar - Fsize * (HR + HOOF) - HOO - HLL - HLL2   # housing supply equals housing demand
    flat_mkt = 0

    # goods market clearing, now including final goods used for housing
    # investment and the fixed cost of the housing investment firms
    goods_mkt = (C + G + (Ih) * ph.apply(np.exp) * alpha_ih + (1 - alpha_ih) * phss * Ihss
                 + Fcm * HA_com - Y)

    rental_mkt = HA_com + (HA_supply + epsH) + HLL + 2 * HLL2 - HR               # rental supply equals rental demand

    rentindex_res = rentindex - PRW / HR

    # consumption by tenure
    COObar = COO / COOWGT
    CRENTbar = CRENT / CRENTWGT
    CMORTbar = CMORT / CMORTWGT
    CLLbar = CLL / CLLWGT

    # disposable income by tenure
    YDIS_OObar = YDIS_OO / COOWGT
    YDIS_RENTbar = YDIS_RENT / CRENTWGT
    YDIS_LLbar = YDIS_LL / CLLWGT
    YDIS_MORTbar = YDIS_MORT / CMORTWGT

    return asset_mkt, rental_mkt, housing_mkt, goods_mkt, rentindex_res, flat_mkt, COObar, CRENTbar, CMORTbar, CLLbar, YDIS_OObar, YDIS_RENTbar, YDIS_LLbar, YDIS_MORTbar


@solved(unknowns={'B': 2.7}, targets=['B_res'], solver="brentq")
def fiscal_IH(Tax, B, r, gammatax, Taxss, Bss, Yss, Gss, w, hours, NEss, rentindex, HA_supply,
              transac, HTRANS, MORTPMT, DEPPMT, TransfAgg, hmcost, Fsize, ADDEFAULT, epsH,
              alpha_ih, Ih, Ihss, ph, phss, Hbar, deltahstar):
    """`fiscal` for the housing investment extension.

    The government budget additionally carries the housing subsidy for changes
    in the cost of depreciation as house prices move, and the profits of the
    housing investment firm, which equal the land share net of the fixed cost.
    """
    G = Gss
    B_res = (B(-1) + G + TransfAgg - Tax * w.apply(np.exp) * NEss * hours.apply(np.exp)
             - MORTPMT + DEPPMT + (HA_supply + epsH) * Fsize * hmcost + ADDEFAULT
             - rentindex * (HA_supply + epsH) - HTRANS * transac
             - (ph.apply(np.exp) * Ih - phss * Ihss) * (1 - alpha_ih)
             + (ph.apply(np.exp) - phss) * Hbar * deltahstar - B)
    Tax_res = Taxss + gammatax * (B(-1) - Bss) / Yss - Tax

    return B_res, Tax_res, G


###################################################################################################
## Additional market segmentation (Section 4.5.1)
###################################################################################################


@simple
def mkt_clearing_seg(Y, C, HR, HLL, HOO, A, B, Hbar, Fsize, HA_supply, G, HOOF, HA_com, ph, pr,
                     HLL2, hmcost, Fcm, rentindex, PRW, HbarF, epsH, COO, COOWGT, CRENT,
                     CRENTWGT, CMORT, CMORTWGT, CLL, CLLWGT, YDIS_OO, YDIS_RENT, YDIS_LL,
                     YDIS_MORT):
    """`mkt_clearing` with flats and houses in fully segmented markets.

    Flats now clear in their own market (`flat_mkt`) at their own price, rather
    than at a fixed proportion of the house price, so demand for rental units
    can no longer spill into the market for houses.
    """
    asset_mkt = A - B

    housing_mkt = Fsize * epsH + Hbar - Fsize * HOOF - HOO - HLL - HLL2          # housing supply equals housing demand

    flat_mkt = HbarF - Fsize * (HA_com + (HA_supply + epsH) + HLL + 2 * HLL2)    # flat market clearing

    goods_mkt = C + G + (Hbar + epsH * Fsize) * hmcost + Fcm * HA_com - Y        # goods market clearing

    rental_mkt = HA_com + (HA_supply + epsH) + HLL + 2 * HLL2 - HR               # rental supply equals rental demand

    rentindex_res = rentindex - PRW / HR

    # consumption by tenure
    COObar = COO / COOWGT
    CRENTbar = CRENT / CRENTWGT
    CMORTbar = CMORT / CMORTWGT
    CLLbar = CLL / CLLWGT

    # disposable income by tenure
    YDIS_OObar = YDIS_OO / COOWGT
    YDIS_RENTbar = YDIS_RENT / CRENTWGT
    YDIS_LLbar = YDIS_LL / CLLWGT
    YDIS_MORTbar = YDIS_MORT / CMORTWGT

    return asset_mkt, rental_mkt, housing_mkt, goods_mkt, rentindex_res, flat_mkt, COObar, CRENTbar, CMORTbar, CLLbar, YDIS_OObar, YDIS_RENTbar, YDIS_LLbar, YDIS_MORTbar


###########################################################################################
## Sticky mortgage rates (Section 4.5.3)
###########################################################################################


@solved(unknowns={'imort': 0.005}, targets=['imort_res'], solver="broyden")
def mortrate(i, pi, imort, FixP):
    """Mortgage rate with a fixation period of `FixP` quarters.

    New borrowers fix at the average expected policy rate over the fixation
    period, and a fraction `1 / FixP` of the stock resets each quarter, so the
    effective mortgage rate responds only sluggishly to policy.  `FixP = 12`
    matches the UK average fixation period.
    """
    thetamort = 1 / FixP

    imortstar = 0.
    for ii in range(FixP):
        imortstar = imortstar + i(+ii) / FixP

    imort_res = thetamort * imortstar + (1 - thetamort) * imort(-1) - imort

    rmort = (1 + imort(-1)) / (1 + pi) - 1

    return rmort, imortstar, imort_res
