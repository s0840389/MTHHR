"""Steady-state blocks.

The static counterparts of the blocks in `dynamics.py`, used to solve for the
model's steady state in Main.ipynb.  Together with the household block they
form the system whose solution is reported in Tables 2 to 4 of the paper.

Several quantities that are unknowns in the dynamic model are pinned down here
by calibration targets instead: the house price is set from the house price to
income ratio `phy`, the rental price `pr` and preference parameters are chosen
to hit the tenure shares and transition rates of Table 2, and government
spending clears the government budget.
"""

import numpy as np

from sequence_jacobian import simple, solved


##############################################################
## Steady state blocks
##############################################################


@simple
def firmSS(Y, Z, alphah, NEss, mu):
    """Production, wages and dividends in steady state.

    The real wage is the marginal product divided by the steady-state markup
    `mu`; `SP` is real marginal cost, which is `1 / mu` here and is the driving
    variable of the price Phillips curve out of steady state.
    """
    hours = np.log((Y / Z) ** (1 / alphah) / NEss)
    w = np.log(1 / mu * alphah * Y) / (np.exp(hours) * NEss)
    prcom_prof = 0
    Div = np.log(Y - np.exp(w) * np.exp(hours) * NEss + prcom_prof)
    SP = np.exp(w) / (alphah * Y / (np.exp(hours) * NEss))

    return w, Div, hours, SP, prcom_prof


@simple
def housingSS(r, w, hours, NEss, borwedge, Fsize, deltah, pr, phy):
    """House prices, transaction costs, asset grid bounds and housing costs.

    The house price is set to hit the target house price to income ratio `phy`
    (Table 3).  `hmcost` is the per-period maintenance cost, calibrated as a
    share `deltah` of the steady-state rent so that it equals the 0.021 of the
    ONS CPI-H housing share.

    `Fcm` is the fixed cost of the (inactive in the baseline) commercial rental
    sector, set as the residual that makes such a firm exactly break even at the
    steady-state rent, so introducing it in Section 4.4 does not change the
    steady state.

    `min_a` and `max_a` are the outer bounds of the asset grids of Table C.1.
    """
    ph = np.log(phy * np.exp(w) * np.exp(hours) * NEss)
    phss = np.exp(ph)
    transac = 0.02 * phss                    # transaction cost
    min_a = -np.exp(ph) * 1.0
    max_a = 8 * np.exp(ph)
    prss = pr * 1
    phF = np.log(np.exp(ph) * Fsize)         # price of a rental flat

    HA_com = 0                               # commercial rental sector, inactive in the baseline

    hmcost = deltah * np.exp(pr) / Fsize     # maintenance cost
    Fcm = np.exp(pr) - phss * Fsize * r / (1 + r) - hmcost * Fsize   # rental company fixed cost

    return ph, phss, transac, min_a, max_a, HA_com, hmcost, Fcm, phF, prss


@simple
def TaxTransfss(pr, transac, ph, r, TransfAgg, Gguess):
    """Steady-state labour tax rate, set to balance the government budget."""
    Transf = 0

    Tax = (Gguess + TransfAgg - 0.2 * np.exp(pr) - 0.02 / 4 * transac * np.exp(ph)
           + r * 0.75 * 4 + 0.2 * 2 / 3 * 0.02)

    return Transf, Tax


@simple
def market_clearingSS(Y, C, NEss, hours, A, HR, HLL, HOO, HOOF, Fsize, Tax, w, DEPPMT, MORTPMT,
                      pr, borwedge, transac, HTRANS, r, TransfAgg, HLL2, hmcost, ADDEFAULT):
    """Steady-state market clearing and aggregate identities.

    Housing supply `Hbar` and the housing association stock `HA_supply` are
    residuals here: they are set to whatever makes the housing and rental
    markets clear given the tenure shares the calibration targets, rather than
    being imposed.  Government spending `G` is likewise the residual of the
    budget constraint.  The market clearing residuals are therefore zero by
    construction in steady state, and only bind out of steady state.
    """
    L = NEss * np.exp(hours)      # labour services

    B = A * 1                     # steady state debt equals household saving

    HA_supply = HR - HLL - 2 * HLL2   # social housing supply

    Hbar = HLL + HLL2 + HOO + Fsize * (HR + HOOF)   # total housing supply
    HbarF = 0

    # government purchases, as the residual of the budget constraint
    G = (Tax * np.exp(w) * L + np.exp(pr) * HA_supply + MORTPMT - DEPPMT + transac * HTRANS
         - TransfAgg - HA_supply * hmcost * Fsize - ADDEFAULT)

    asset_mkt = 0
    housing_mkt = 0
    rental_mkt = 0
    goods_mkt = Y - C - G - Hbar * hmcost
    flat_mkt = 0

    epsH = 0                      # rental supply shock, zero in steady state

    rentweight = np.exp(pr) * HR / (C + HR * np.exp(pr))   # weight of rent in the CPI

    return L, B, HA_supply, G, Hbar, asset_mkt, housing_mkt, rental_mkt, goods_mkt, HbarF, flat_mkt, epsH, rentweight


@simple
def pricesSS(r, pr):
    """Nominal variables and residuals, at their steady-state values.

    Zero steady-state inflation, so the nominal rate equals the real rate and
    the mortgage rate equals both.  Defined here so that every variable the
    dynamic model needs is present in the steady-state dictionary.
    """
    i_res = 0
    pi_res = 0
    piw_res = 0
    B_res = 0
    Tax_res = 0
    fisher_res = 0
    realwage_res = 0

    pi = 0
    piw = 0
    i = r * 1
    rstar = r * 1
    rentindex = np.exp(pr) * 1
    prlag = pr * 1

    pi_cpi = 0

    rmort = r * 1

    return i_res, pi_res, piw_res, B_res, Tax_res, fisher_res, realwage_res, pi, piw, i, rstar, rentindex, prlag, pi_cpi, rmort


@simple
def SSparams(SW, SP, w, hours, Tax, B, Y, frisch, kappaw, kappa, phi, gammatax, rhom, pr_calvo,
             phiy, pr, deltaWK, deltaWKph, gammaWK, gammaWKph):
    """Record steady-state levels and carry the dynamic parameters into `ss0`.

    The Phillips curve slopes, policy rule and expectations parameters are not
    used in steady state, but taking them as inputs puts them in the
    steady-state dictionary, where `dynIRFs` can overwrite them with candidate
    values during IRF matching.
    """
    # parameters based on steady state values
    SWss = SW * 1
    SPss = SP * 1
    wss = w * 1
    hourss = hours * 1
    Taxss = Tax * 1
    Bss = B * 1
    Yss = Y * 1

    # shocks
    epsr = 0

    return SWss, SPss, wss, hourss, Taxss, Bss, Yss, epsr


@simple
def SSsolve(B, Btarget, HR, HRtarget, HLL, HLLtarget, LLEXIT, OWNEXIT, HOO, LLEXITtarget,
            OWNEXITtarget, HLL2, HOOF, HOOFtarget):
    """Distance between the model's steady state and the calibration targets.

    The six residuals correspond to the internally estimated moments of Table 2:
    liquid saving, the renter share, the landlord share, the annual landlord
    exit rate, the annual owner exit rate and the flat-owner share.  `uloss`
    aggregates them into a single scalar for a minimiser, with the housing
    moments weighted ten times the debt target.

    Not part of the baseline solve in Main.ipynb, which uses the already
    calibrated parameters; provided for re-running the internal calibration.
    """
    Bsolve = B - Btarget                                    # debt target
    HRsolve = HR - HRtarget                                 # renter share target
    HLLsolve = HLL + HLL2 - HLLtarget                       # landlord share target
    LLEXITsolve = 4 * LLEXIT / (HLL + HLL2) - LLEXITtarget  # landlord exit rate target
    OWNEXITsolve = 4 * OWNEXIT / HOO - OWNEXITtarget        # owner exit rate target
    HOOFsolve = HOOF - HOOFtarget                           # flat owner share target

    uloss = ((Bsolve ** 2) + 10 * (HRsolve ** 2) + 10 * (HLLsolve ** 2) + 10 * (LLEXITsolve ** 2)
             + 10 * (OWNEXITsolve ** 2) + 10 * (HOOFsolve ** 2))

    return Bsolve, HRsolve, HLLsolve, LLEXITsolve, OWNEXITsolve, uloss
