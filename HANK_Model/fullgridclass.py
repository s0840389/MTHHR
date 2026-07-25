"""State-dependent ("full grid") endogenous choice stages for `sequence_jacobian`.

This module is the first of the two code innovations in this replication package
(the other is `redform_jacob` in `behave_algo.py`).  It supplies a drop-in
replacement for the `Continuous2D` stage of the `sequence_jacobian` (SSJ)
package that allows the grid a policy is chosen from to *vary across the state
space*, and it repairs a bug in SSJ's shocked two-dimensional policy lottery.


Why we need it
--------------
An SSJ `Continuous1D` / `Continuous2D` stage assumes each endogenous policy is
chosen from a single grid shared by every point of the state space: the stage
looks up `inputs['a_grid']`, a 1-D vector of length `n_a`.  That assumption
fails on both endogenous dimensions of this model.

* **Assets, `a`.**  The asset grid has to depend on the housing transition `h`
  (the first state dimension).  A renter cannot borrow at all, so their grid
  starts at zero; an owner-occupier may borrow against a house up to the LTV /
  LTI limit, so their grid starts at `-p_h H_2`; a landlord may borrow against
  both their home and their rental flat, so their grid starts at
  `-p_h (kappa_h + kappa_h,LL)`.  The top of each grid differs too, being a
  multiple of the value of the property that tenure owns.  Table C.1 of the
  paper lists the four grids.

  Forcing all tenures onto one common grid would mean either spending most of
  the 250 gridpoints on regions that households of a given tenure can never
  reach, or resolving the borrowing-constrained region far too coarsely -- and
  that region is precisely where the policy functions kink and where the high
  MPCs that drive the paper's results are located.

* **Rental price, `pri`.**  Renters and landlords carry a household-specific
  nominal rent, tracked on a three-point grid around the steady-state rent
  `p_r,ss` (a household-level state, because rents reset only with probability
  `theta_r` each period).  Owner-occupiers have no rent to carry and sit on a
  degenerate grid around zero.  This grid is therefore state-dependent by
  construction, not by choice.

`Continuous2DFullGrid` relaxes the assumption.  Instead of `<policy>_grid`, of
shape `(n_a,)`, it reads `<policy>_grid_full`, an array with the *same shape as
the state space*, `(n_h, n_z, n_a, n_pr)`, built by `make_grids` in
`household.py`.  Interpolation of the policy onto the grid, the resulting
policy lottery, and the grid spacing used to differentiate that lottery are all
then done grid-by-grid rather than once for the whole state space.

In this model the grids vary only along the *first* (housing-transition)
dimension.  The interpolation routine exploits that by reading one
representative slice per `h`, which keeps the cost of the extra generality to a
single Python-level loop over the 17 housing transitions.


The SSJ bug this also fixes
---------------------------
Independently of the full-grid extension, SSJ's `ShockedPolicyLottery2D` is
broken: it inherits the constructor of `PolicyLottery2D` (which stores
`i1, pi1, i2, pi2`) but its `__matmul__` calls
`het_compiled.forward_policy_shock_2d(X, self.i, self.pi)`, referring to
attributes that only exist on the *one*-dimensional lottery and passing three
arguments to a function that takes seven.  `Continuous2D` compounds this by
calling `ShockedPolicyLottery2D(i1, dpi1, i2, dpi2, ...)`, i.e. passing the
*derivatives* of the lottery weights in the slots meant for the steady-state
weights, so the steady-state weights are discarded entirely.

`forward_policy_shock_2d` needs both: the steady-state weights `pi1, pi2` to
know how mass is currently split between bracketing gridpoints, and the shocks
`dpi1, dpi2` to know how that split moves.  `ShockedPolicyLottery2DFullGrid`
below therefore carries both sets and passes all seven arguments.


Usage
-----
Build the stage exactly as you would an SSJ `Continuous2D`, and make sure the
backward function's inputs include `<policy>_grid_full` for each policy::

    consav_stage = Continuous2DFullGrid(backward=['V', 'Va'], policy=['a', 'pri'],
                                        f=dcegm, name='consav')

State-space layout is assumed to be `(n_h, n_z, n_a, n_pr)`: the two policy
dimensions are the last two, and the grids vary along the first.
"""

import copy

import numpy as np

from sequence_jacobian.blocks.support import het_compiled
from sequence_jacobian.blocks.support.law_of_motion import LawOfMotion
from sequence_jacobian.blocks.support.stages import Stage
from sequence_jacobian.utilities.function import ExtendedFunction
from sequence_jacobian.utilities.interpolate import interpolate_coord_robust_vector
from sequence_jacobian.utilities.misc import make_tuple
from sequence_jacobian.utilities.ordered_set import OrderedSet


def interpolate_coord_robust_full_grid_2D(x, xq, idim):
    """Locate query points `xq` on state-dependent grids `x`.

    The full-grid analogue of SSJ's `interpolate_coord_robust`.  Rather than
    interpolating every query point onto one common grid, we interpolate the
    query points belonging to each value of the first state dimension onto that
    state's own grid, so that

        xq = xqpi * x[xqi] + (1 - xqpi) * x[xqi + 1]

    holds grid-by-grid.  Monotonicity is exploited in the data `x` only, not in
    the query points `xq`, since the policy functions here are not monotone in
    the state (a discrete tenure choice can make the saving policy jump).

    Parameters
    ----------
    x    : array (n_h, n_z, n_a, n_pr), the full grid; ascending along `idim`
           and, in this model, constant across the dimensions other than the
           first, so one representative slice per `n_h` is read
    xq   : array (n_h, n_z, n_a, n_pr), query points, in any order
    idim : 1 to interpolate onto the `a` (third) dimension of the grid,
           2 to interpolate onto the `pri` (fourth) dimension

    Returns
    -------
    xqi  : array, same shape as `xq`, indices of lower bracketing gridpoints
    xqpi : array, same shape as `xq`, weights on lower bracketing gridpoints
    """
    if x.ndim < 2:
        raise ValueError('Data input to interpolate_coord_robust_full_grid_2D '
                         'must have at least two dimensions')
    if xq.ndim == 1:
        raise ValueError('Query points must retain the full state-space shape '
                         'so that each one can be matched to its own grid')

    hold_shape = xq.shape

    # collapse everything except the first (grid-varying) dimension
    xq = xq.reshape((xq.shape[0], -1))

    xqi = np.empty(xq.shape, dtype=np.uint32)
    xqpi = np.empty(xq.shape)

    for ik in range(xq.shape[0]):
        if idim == 1:
            grid_ik = x[ik, 0, :, 0]    # asset grid for this housing transition
        elif idim == 2:
            grid_ik = x[ik, 0, 0, :]    # rental-price grid for this housing transition
        else:
            raise ValueError('idim must be 1 (assets) or 2 (rental price)')

        xqi[ik, :], xqpi[ik, :] = interpolate_coord_robust_vector(grid_ik, xq[ik, :])

    return xqi.reshape(hold_shape), xqpi.reshape(hold_shape)


def lottery_2d_full_grid(a, b, a_grid_full, b_grid_full):
    """Build the two-dimensional policy lottery on state-dependent grids."""
    return PolicyLottery2DFullGrid(*interpolate_coord_robust_full_grid_2D(a_grid_full, a, 1),
                                   *interpolate_coord_robust_full_grid_2D(b_grid_full, b, 2),
                                   a_grid_full, b_grid_full)


class PolicyLottery2DFullGrid(LawOfMotion):
    """Law of motion for two endogenous policies chosen on state-dependent grids.

    Identical in interface to SSJ's `PolicyLottery2D`.  The only substantive
    difference is the reshaping: `grid1` and `grid2` are now full state-space
    arrays rather than vectors, so the flattened shape expected by the compiled
    forward/expectation routines, `(n_h * n_z, n_a, n_pr)`, has to be read off
    the relevant axes of those arrays instead of from their length.

    The compiled routines themselves need no change: once the interpolation has
    been done they only ever see bracketing indices and weights, never the grids.
    """

    def __init__(self, i1, pi1, i2, pi2, grid1, grid2, forward=True):
        # flatten the non-policy dimensions into one, which is what
        # het_compiled's routines accept: (n_h * n_z, n_a, n_pr)
        self.i1 = i1.reshape((-1,) + grid1.shape[-2:-1] + grid2.shape[-1:])
        self.flatshape = self.i1.shape

        self.i2 = i2.reshape(self.flatshape)
        self.pi1 = pi1.reshape(self.flatshape)
        self.pi2 = pi2.reshape(self.flatshape)

        # keep the original shape so outputs can be converted back to it
        self.shape = i1.shape
        self.grid1 = grid1
        self.grid2 = grid2

        # shape of the endogenous grid itself
        self.endog_shape = self.shape[-2:]

        self.forward = forward

    @property
    def T(self):
        newself = copy.copy(self)
        newself.forward = not self.forward
        return newself

    def __matmul__(self, X):
        if self.forward:
            return het_compiled.forward_policy_2d(X.reshape(self.flatshape), self.i1, self.i2,
                                                  self.pi1, self.pi2).reshape(self.shape)
        else:
            return het_compiled.expectation_policy_2d(X.reshape(self.flatshape), self.i1, self.i2,
                                                      self.pi1, self.pi2).reshape(self.shape)


class ShockedPolicyLottery2DFullGrid(PolicyLottery2DFullGrid):
    """First-order shock to the two-dimensional policy lottery.

    Carries both the steady-state lottery weights (`pi1`, `pi2`, inherited) and
    their derivatives (`dpi1`, `dpi2`), because `forward_policy_shock_2d` needs
    both to push a distribution through a perturbed lottery.  SSJ's own
    `ShockedPolicyLottery2D` supplies neither correctly; see the module
    docstring.
    """

    def __init__(self, i1, pi1, dpi1, i2, pi2, dpi2, grid1, grid2, forward=True):
        # store the steady-state indices and weights
        super().__init__(i1, pi1, i2, pi2, grid1, grid2, forward)

        # and the shocks to the weights
        self.dpi1 = dpi1.reshape(self.flatshape)
        self.dpi2 = dpi2.reshape(self.flatshape)

    def __matmul__(self, X):
        if self.forward:
            return het_compiled.forward_policy_shock_2d(X.reshape(self.flatshape),
                                                        self.i1, self.i2,
                                                        self.pi1, self.pi2,     # steady-state weights
                                                        self.dpi1, self.dpi2    # shocks to weights
                                                        ).reshape(self.shape)
        else:
            raise NotImplementedError


class Continuous2DFullGrid(Stage):
    """Stage that makes a two-dimensional endogenous continuous choice on
    state-dependent grids.

    Use in place of SSJ's `Continuous2D`.  The backward function `f` must take
    `<policy>_grid_full` for each policy as an input -- a full state-space array
    -- rather than the 1-D `<policy>_grid` that `Continuous2D` expects.  Here
    the policies are `a` (saving) and `pri` (the household-specific rent
    carried into next period), and `make_grids` in `household.py` supplies
    `a_grid_full` and `pri_grid_full`.
    """

    def __init__(self, backward, policy, f, name=None, hetoutputs=None):
        # subclass-specific attributes
        self.f = ExtendedFunction(f)
        self.policy = OrderedSet(policy)

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.backward_outputs = OrderedSet(make_tuple(backward))
        self.report = self.f.outputs - self.backward_outputs
        self.inputs = self.f.inputs

        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Continuous2DFullGrid '{self.name}' with policies {self.policy}>"

    def backward_step(self, inputs, lawofmotion=False):
        outputs = self.f(inputs)

        if not lawofmotion:
            return outputs
        else:
            return outputs, lottery_2d_full_grid(outputs[self.policy[0]], outputs[self.policy[1]],
                                                 inputs[self.policy[0] + '_grid_full'],
                                                 inputs[self.policy[1] + '_grid_full'])

    def backward_step_shock(self, ss, shocks, precomputed):
        """Differentiate the stage: a change in the policy moves mass between
        the bracketing gridpoints at rate `-dpolicy / spacing`."""
        space1, space2, i1, i2, pi1, pi2, grid1, grid2, f = precomputed

        outputs = f.diff(shocks)

        dpi1 = -outputs[self.policy[0]] / space1
        dpi2 = -outputs[self.policy[1]] / space2

        return outputs, ShockedPolicyLottery2DFullGrid(i1, pi1, dpi1, i2, pi2, dpi2, grid1, grid2)

    def precompute(self, ss, ss_lawofmotion):
        """Cache the objects `backward_step_shock` needs at every horizon.

        The one full-grid-specific piece is the grid spacing.  With a common
        grid the spacing at the steady-state lottery point is just
        `grid[i + 1] - grid[i]`; with a state-dependent grid the relevant
        gridpoints differ across the state space, so we gather them with
        `take_along_axis` along the asset axis (2) and the rental-price
        axis (3) respectively.
        """
        i1 = ss_lawofmotion.i1.reshape(ss_lawofmotion.shape)
        i2 = ss_lawofmotion.i2.reshape(ss_lawofmotion.shape)
        pi1 = ss_lawofmotion.pi1.reshape(ss_lawofmotion.shape)
        pi2 = ss_lawofmotion.pi2.reshape(ss_lawofmotion.shape)
        grid1 = ss_lawofmotion.grid1
        grid2 = ss_lawofmotion.grid2

        space1 = np.take_along_axis(grid1, i1 + 1, axis=2) - np.take_along_axis(grid1, i1, axis=2)
        space2 = np.take_along_axis(grid2, i2 + 1, axis=3) - np.take_along_axis(grid2, i2, axis=3)

        return (space1, space2, i1, i2, pi1, pi2, grid1, grid2, self.f.differentiable(ss))
