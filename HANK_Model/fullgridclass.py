from math import isnan
import numpy as np
import copy

from sequence_jacobian.utilities.function import ExtendedFunction, CombinedExtendedFunction

from sequence_jacobian.utilities.ordered_set import OrderedSet

from sequence_jacobian.utilities.misc import make_tuple

from sequence_jacobian.utilities.interpolate import interpolate_coord_robust_vector,interpolate_coord,interpolate_coord_robust

from sequence_jacobian.blocks.support.stages import Stage

from sequence_jacobian.blocks.support.law_of_motion import PolicyLottery1D, LawOfMotion

from sequence_jacobian.blocks.support import het_compiled


##############################################################################################################################
## One dimensional - full grid (grid can vary by 1st dimension)
##############################################################################################################################

# new stage block

class Continuous1DFullGrid(Stage):

    """Stage that does one-dimensional endogenous continuous choice"""

    def __init__(self, backward, policy, f, name=None, hetoutputs=None):

        # subclass-specific attributes

        self.f = ExtendedFunction(f)

        self.policy = policy

        # attributes needed for any stage

        if name is None:

            name = self.f.name

        self.name = name

        self.backward_outputs = OrderedSet(make_tuple(backward))

        self.report = self.f.outputs - self.backward_outputs

        self.inputs = self.f.inputs

        super().__init__(hetoutputs)


    def __repr__(self):

        return f"<Stage-Continuous1DFullGrid '{self.name}' with policy '{self.policy}'>"

 
    def backward_step(self, inputs, lawofmotion=False):

        outputs = self.f(inputs)

        if not lawofmotion:

            return outputs

        else:

            # TODO: option for monotonic?!

            return outputs, lottery_1d_full_grid(outputs[self.policy], inputs[self.policy + '_grid_full'])

   
    def backward_step_shock(self, ss, shocks, precomputed):
        space, i, grid, f = precomputed

        outputs = f.diff(shocks)

        dpi = -outputs[self.policy] / space
        return outputs, ShockedPolicyLottery1DFullGrid(i, dpi, grid)

    def precompute(self, ss, ss_lawofmotion):
        i = ss_lawofmotion.i.reshape(ss_lawofmotion.shape)
        grid = ss_lawofmotion.grid
        space=np.zeros(i.shape)
        for ii in range(i.shape[0]):
            for jj in range(i.shape[1]):
                space[ii,jj]=grid[ii,jj,i[ii,jj]+1]-grid[ii,jj,i[ii,jj]]
                
        return space, i, grid, self.f.differentiable(ss)

 

class PolicyLottery1DFullGrid(LawOfMotion):

    # TODO: always operates on final dimension, make more general!

    # Point is, each point only goes to one point - this isn't true in general

    def __init__(self, i, pi, grid, forward=True):

        # flatten non-policy dimensions into one because that's what methods accept - the 'grid' shape is of the 'a' which is the last dimension
        # Now grid, i and pi should all be the same shape
        self.i = i.reshape((-1,) + grid.shape[-1:]) # <<<
        #self.i = i.reshape((-1,) + grid.shape)

        self.flatshape = self.i.shape
        self.pi = pi.reshape(self.flatshape)

        # but store original shape so we can convert all outputs to it
        self.shape = i.shape
        self.grid = grid

        # also store shape of the endogenous grid itself
        self.endog_shape = self.shape[-1:]
        self.forward = forward

        self.ishock=i.reshape(self.flatshape) # when doing the shock it's the policy that is i not the grid #<<<



    @property

    def T(self):

        newself = copy.copy(self)

        newself.forward = not self.forward

        return newself


    def __matmul__(self, X):

        if self.forward:

            return het_compiled.forward_policy_1d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)

        else:

            return het_compiled.expectation_policy_1d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)

 

class ShockedPolicyLottery1DFullGrid(PolicyLottery1DFullGrid): # Note, inherits from PolicyLottery1D**FullGrid** class

    def __matmul__(self, X):
        if self.forward:

            return het_compiled.forward_policy_shock_1d(X.reshape(self.flatshape), self.ishock, self.pi).reshape(self.shape)

        else:

            raise NotImplementedError

 

def interpolate_coord_robust_full_grid(x, xq, check_increasing=False):

    """Linear interpolation exploiting monotonicity only in data x, not in query points xq.

    Simple binary search, less efficient but more robust.

    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

 

    Main application intended to be universally-valid interpolation of policy rules.

    Dimension k is optional.

 

    Parameters

    ----------

    x    : array (k, n), ascending data points

    xq   : array (k, nq), query points (in any order)

 

    Returns

    ----------

    xqi  : array (k, nq), indices of lower bracketing gridpoints

    xqpi : array (k, nq), weights on lower bracketing gridpoints

    """

    if x.ndim < 2:

        raise ValueError('Data input to interpolate_coord_robust_full_grid must have at least two dimensions')

 

    # if check_increasing and np.any(x[:-1] >= x[1:]):

    #     raise ValueError('Data input to interpolate_coord_robust must be strictly increasing')

 

    if xq.ndim == 1:

        assert False

        # return interpolate_coord_robust_vector(x, xq)

    else:

        hold_shape = xq.shape


        xq = xq.reshape((-1,) + xq.shape[-1:])

        x = x.reshape((-1,) + x.shape[-1:])

    
        xqi = np.empty(xq.shape, dtype=np.uint32)

        xqpi = np.empty(xq.shape)


        for ik in range(xq.shape[0]):


            xqi[ik,:], xqpi[ik,:] = interpolate_coord_robust_vector(x[ik,:], xq[ik,:])

 

        xqi = xqi.reshape(hold_shape)

        xqpi = xqpi.reshape(hold_shape)

        return xqi, xqpi

def lottery_1d_full_grid(a, a_grid_full):

    return PolicyLottery1DFullGrid(*interpolate_coord_robust_full_grid(a_grid_full, a), a_grid_full)


##############################################################################################################################
## Two dimensional - fix for SSJ bug to allow for 2d grid in stage block
##############################################################################################################################


class Continuous2D(Stage):
    """Stage that does two-dimensional endogenous continuous choice"""
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
        return f"<Stage-Continuous2D '{self.name}' with policies {self.policy}>"

    def backward_step(self, inputs, lawofmotion=False):
        outputs = self.f(inputs)

        if not lawofmotion:
            return outputs
        else:
            # TODO: option for monotonic?!
            return outputs, lottery_2d(outputs[self.policy[0]], outputs[self.policy[1]],
                                       inputs[self.policy[0] + '_grid'], inputs[self.policy[1] + '_grid'])
    

    def backward_step_shock(self, ss, shocks, precomputed):
        space1, space2, i1, i2, pi1, pi2, grid1, grid2, f = precomputed  # Unpack pi1, pi2
        outputs = f.diff(shocks)
        dpi1 = -outputs[self.policy[0]] / space1
        dpi2 = -outputs[self.policy[1]] / space2

        return outputs, ShockedPolicyLottery2D(i1, pi1, dpi1, i2, pi2, dpi2, grid1, grid2)

    def precompute(self, ss, ss_lawofmotion):
        i1 = ss_lawofmotion.i1.reshape(ss_lawofmotion.shape)
        i2 = ss_lawofmotion.i2.reshape(ss_lawofmotion.shape)
        pi1 = ss_lawofmotion.pi1.reshape(ss_lawofmotion.shape)  # Add this
        pi2 = ss_lawofmotion.pi2.reshape(ss_lawofmotion.shape)  # Add this
        grid1 = ss_lawofmotion.grid1
        grid2 = ss_lawofmotion.grid2

        return (grid1[i1 + 1] - grid1[i1], grid2[i2 + 1] - grid2[i2],
                i1, i2, pi1, pi2, grid1, grid2, self.f.differentiable(ss))  # Include pi1, pi2



def lottery_2d(a, b, a_grid, b_grid, monotonic=False):
    if not monotonic:
        return PolicyLottery2D(*interpolate_coord_robust(a_grid, a),
                           *interpolate_coord_robust(b_grid, b), a_grid, b_grid)
    if monotonic:
        # right now we have no monotonic 2D examples, so this shouldn't be called
        return PolicyLottery2D(*interpolate_coord(a_grid, a),
                           *interpolate_coord(b_grid, b), a_grid, b_grid)


class PolicyLottery2D(LawOfMotion):
    def __init__(self, i1, pi1, i2, pi2, grid1, grid2, forward=True):
        # flatten non-policy dimensions into one because that's what methods accept
        self.i1 = i1.reshape((-1,) + grid1.shape + grid2.shape)
        self.flatshape = self.i1.shape

        self.i2 = i2.reshape(self.flatshape)
        self.pi1 = pi1.reshape(self.flatshape)
        self.pi2 = pi2.reshape(self.flatshape)



        # but store original shape so we can convert all outputs to it
        self.shape = i1.shape
        self.grid1 = grid1
        self.grid2 = grid2

        # also store shape of the endogenous grid itself
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


#class ShockedPolicyLottery2D(PolicyLottery2D):
#    def __matmul__(self, X):
#        if self.forward:
#            return het_compiled.forward_policy_shock_2d(X.reshape(self.flatshape), self.i1,self.i2, self.pi1,self.pi2,self.pi1,self.pi2).reshape(self.shape)
#        else:
#            raise NotImplementedError
        

class ShockedPolicyLottery2D(PolicyLottery2D):
    def __init__(self, i1, pi1, dpi1, i2, pi2, dpi2, grid1, grid2, forward=True):
        # Call parent constructor with actual steady-state weights
        super().__init__(i1, pi1, i2, pi2, grid1, grid2, forward)
        
        # Store the shocked policy weights
        self.dpi1 = dpi1.reshape(self.flatshape)
        self.dpi2 = dpi2.reshape(self.flatshape)
    
    def __matmul__(self, X):
        if self.forward:
            return het_compiled.forward_policy_shock_2d(X.reshape(self.flatshape), 
                                                       self.i1, self.i2, 
                                                       self.pi1, self.pi2,  # steady-state weights
                                                       self.dpi1, self.dpi2).reshape(self.shape)  # shocked weights
        else:
            raise NotImplementedError



##############################################################################################################################
## Two dimensional Full grid - (allow varying grid by first dimension)
##############################################################################################################################







def interpolate_coord_robust_full_grid_2D(x, xq,idim,check_increasing=False):

    """Linear interpolation exploiting monotonicity only in data x, not in query points xq.

    Simple binary search, less efficient but more robust.

    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

 

    Main application intended to be universally-valid interpolation of policy rules.

    Dimension k is optional.

 

    Parameters

    ----------

    x    : array (k, n), ascending data points

    xq   : array (k, nq), query points (in any order)

 

    Returns

    ----------

    xqi  : array (k, nq), indices of lower bracketing gridpoints

    xqpi : array (k, nq), weights on lower bracketing gridpoints

    """

    if x.ndim < 2:

        raise ValueError('Data input to interpolate_coord_robust_full_grid must have at least two dimensions')

 

    # if check_increasing and np.any(x[:-1] >= x[1:]):

    #     raise ValueError('Data input to interpolate_coord_robust must be strictly increasing')

 

    if xq.ndim == 1:

        assert False

        # return interpolate_coord_robust_vector(x, xq)

    else:
        
        hold_shape = xq.shape




        xq = xq.reshape((xq.shape[0], -1))
        #x = x.reshape((x.shape[0], -1))



        xqi = np.empty(xq.shape, dtype=np.uint32)

        xqpi = np.empty(xq.shape)



        for ik in range(xq.shape[0]):

            #print(x[ik,:].shape)
            #print(xq[ik,:].shape)
            #print(np.sum(x[ik,:]))
            #print(np.sum(xq[ik,:]))

            if idim==1:
                xqi[ik,:], xqpi[ik,:] = interpolate_coord_robust_vector(x[ik,0,:,0], xq[ik,:])
            if idim==2:
                xqi[ik,:], xqpi[ik,:] = interpolate_coord_robust_vector(x[ik,0,0,:], xq[ik,:])

 

        xqi = xqi.reshape(hold_shape)

        xqpi = xqpi.reshape(hold_shape)

        return xqi, xqpi
    


class Continuous2DFullGrid(Stage):
    """Stage that does two-dimensional endogenous continuous choice"""
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
        return f"<Stage-Continuous2D '{self.name}' with policies {self.policy}>"

    def backward_step(self, inputs, lawofmotion=False):
        outputs = self.f(inputs)

        if not lawofmotion:
            return outputs
        else:
            # TODO: option for monotonic?!
            return outputs, lottery_2d(outputs[self.policy[0]], outputs[self.policy[1]],
                                       inputs[self.policy[0] + '_grid_full'], inputs[self.policy[1] + '_grid_full'])


    def backward_step_shock(self, ss, shocks, precomputed):
        space1, space2, i1, i2, pi1, pi2, grid1, grid2, f = precomputed  # Unpack pi1, pi2
        outputs = f.diff(shocks)
        dpi1 = -outputs[self.policy[0]] / space1
        dpi2 = -outputs[self.policy[1]] / space2

        return outputs, ShockedPolicyLottery2DFullGrid(i1, pi1, dpi1, i2, pi2, dpi2, grid1, grid2)

    def precompute(self, ss, ss_lawofmotion):
        i1 = ss_lawofmotion.i1.reshape(ss_lawofmotion.shape)
        i2 = ss_lawofmotion.i2.reshape(ss_lawofmotion.shape)
        pi1 = ss_lawofmotion.pi1.reshape(ss_lawofmotion.shape)  # Add this
        pi2 = ss_lawofmotion.pi2.reshape(ss_lawofmotion.shape)  # Add this
        grid1 = ss_lawofmotion.grid1
        grid2 = ss_lawofmotion.grid2

        space1 = np.take_along_axis(grid1, i1 + 1, axis=2) - np.take_along_axis(grid1, i1, axis=2)
        space2 = np.take_along_axis(grid2, i2 + 1, axis=3) - np.take_along_axis(grid2, i2, axis=3)
        
        return (space1, space2,
                i1, i2, pi1, pi2, grid1, grid2, self.f.differentiable(ss))  # Include pi1, pi2



def lottery_2d(a, b, a_grid, b_grid, monotonic=False):
    if not monotonic:
        return PolicyLottery2DFullGrid(*interpolate_coord_robust_full_grid_2D(a_grid, a,1),
                           *interpolate_coord_robust_full_grid_2D(b_grid, b,2), a_grid, b_grid)
    if monotonic:
        # right now we have no monotonic 2D examples, so this shouldn't be called
        return PolicyLottery2DFullGrid(*interpolate_coord_robust_full_grid_2D(a_grid, a,1),
                           *interpolate_coord_robust_full_grid_2D(b_grid, b,2), a_grid, b_grid)


class PolicyLottery2DFullGrid(LawOfMotion):
    def __init__(self, i1, pi1, i2, pi2, grid1, grid2, forward=True):
        # flatten non-policy dimensions into one because that's what methods accept

        self.i1 = i1.reshape((-1,) + grid1.shape[-2:-1] + grid2.shape[-1:])

        self.flatshape = self.i1.shape

        self.i2 = i2.reshape(self.flatshape)
        self.pi1 = pi1.reshape(self.flatshape)
        self.pi2 = pi2.reshape(self.flatshape)



        # but store original shape so we can convert all outputs to it
        self.shape = i1.shape
        self.grid1 = grid1
        self.grid2 = grid2

        # also store shape of the endogenous grid itself
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
    def __init__(self, i1, pi1, dpi1, i2, pi2, dpi2, grid1, grid2, forward=True):
        # Call parent constructor with actual steady-state weights
        super().__init__(i1, pi1, i2, pi2, grid1, grid2, forward)
        
        # Store the shocked policy weights
        self.dpi1 = dpi1.reshape(self.flatshape)
        self.dpi2 = dpi2.reshape(self.flatshape)
    
    def __matmul__(self, X):
        if self.forward:
            return het_compiled.forward_policy_shock_2d(X.reshape(self.flatshape), 
                                                       self.i1, self.i2, 
                                                       self.pi1, self.pi2,  # steady-state weights
                                                       self.dpi1, self.dpi2).reshape(self.shape)  # shocked weights
        else:
            raise NotImplementedError

