
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from numba import njit,prange
import numpy as np
from matplotlib import pyplot as plt
import copy




@njit
def normx(x,minx,maxx): # normalise x to be between 0 and 1
    xout=(x-minx)/(maxx-minx)
    return xout

@njit
def expandx(x,minx,maxx): # expand x from 0 to 1 to minx to maxx
    xout=x*(maxx-minx)+minx
    return xout


@njit
def mapx(a,a0,a1,b0,b1): # normalise x to be between 0 and 1
    b=(a-a0)/(a1-a0)*(b1-b0)+b0
    return b

@njit
def mapx_exp(a,a0,a1,b0,b1,phi): # normalise x to be between 0 and 1
    b=(np.exp(a*phi)-np.exp(a0*phi))/(np.exp(a1*phi)-np.exp(a0*phi))*(b1-b0)+b0
    return b

@njit
def mapx_exp_inv(b,b0,b1,a0,a1,phi): # normalise x to be between 0 and 1
    a=np.log((b-b0)/(b1-b0)*(np.exp(a1*phi)-np.exp(a0*phi))+np.exp(a0*phi))/phi    
    return a


def mapgrid(ugrid,cutvalues,Gbreaks):

    G=ugrid*0

    #cutvalues=np.quantile(Cugrid,ubreaks,interpolation='higher')

    for i in range(cutvalues.shape[0]-1):
        G[(ugrid>=cutvalues[i])&(ugrid<=cutvalues[i+1])]=mapx(ugrid[(ugrid>=cutvalues[i])&(ugrid<=cutvalues[i+1])],cutvalues[i],cutvalues[i+1],Gbreaks[i],Gbreaks[i+1])    

    return G

def mapgrid_inv(G,cutvalues,Gbreaks):
    # mapping from expand grid back to uniform grid
    u=G*0

    #cutvalues=np.quantile(ugrid,ubreaks,interpolation='higher')

    for i in range(cutvalues.shape[0]-1):
        u[(G>=Gbreaks[i])&(G<=Gbreaks[i+1])]=mapx(G[(G>=Gbreaks[i])&(G<=Gbreaks[i+1])],Gbreaks[i],Gbreaks[i+1],cutvalues[i],cutvalues[i+1])
        
    u[G>Gbreaks[-1]]=1

    return u


@njit
def buildJK(Fk): # function to build jacobian from fake news matrix

    Jkout=Fk*0

    Jkout[0,:]=Fk[0,:]    
    Jkout[:,0]=Fk[:,0]

    for i in range(1,T):
        Jkout[i,1:]=Jkout[i-1,:-1]+Fk[i,1:]

    return Jkout
    



############################################################
## Sticky Expectations
############################################################



@njit
def JtoF(Jyp):
    '''Function to convert Jacobian back to Fake news matrix'''
    T=Jyp.shape[0]

    Fyp=np.zeros((T,T))

    Fyp[0,:]=Jyp[0,:]
    Fyp[:,0]=Jyp[:,0]

    for i in range(1,T):
        Fyp[i,1:]=Jyp[i,1:]-Jyp[i-1,0:T-1]

    return Fyp


def JtoFdict(Jk):
    '''Function to convert Jacobian dictionary back to Fake news matrix dictionary'''
    Fk=copy.deepcopy(Jk)

    for i in Jk.outputs:
        for j in Jk.inputs:
                try:
                    Fk[i][j]=JtoF(Jk[i][j])
                except Exception:
                    pass
    return Fk


@njit
def makesticky(theta,x): # see appendix D3 of micro jumps macro humps paper

    xsticky=x*0

    xsticky[:,0]=x[:,0]    
    xsticky[0,1:x.shape[1]]=(1-theta)*x[0,1:x.shape[1]]    

    for t in range(1,x.shape[0]):
        for s in range(1,x.shape[1]):

            xsticky[t,s]=theta*xsticky[t-1,s-1]+(1-theta)*x[t,s]

    return xsticky 



def stick_jacob_sel(J,theta,selp): # just selected prices sticky

    Jsticky=copy.deepcopy(J)

    for i in J.outputs:

        for j in J.inputs:
            if j in selp:
                x=J[i][j]

                if j=='r':
                    xsticky=makesticky(theta,x)
                else:
                    xsticky=makesticky(theta,x)

                Jsticky[i][j]=xsticky
            else:
                Jsticky[i][j]=J[i][j]
    return Jsticky


############################################################
## reduced form behavioural stuff
############################################################


@njit
def calc_extrap_eff(T, gamma, delta):
    
    vdelta=1/(1+delta)

    forcwgt_y0=np.zeros((T,T)) # matrix describing the contribution of y0 to to forecast of p_s at time t
    forcwgt_y0[0,0]=1
    forcwgt_y0[0,1:T]=-gamma*vdelta
    
    # after period 0 y0 contributs via laggedforecasts of yk and yk-h

    for t in range(1,T):
        
        forcwgt_y0[t,t+1:T]=delta*vdelta*forcwgt_y0[t-1,t+1:T]

        forcwgt_y0[t,1:T-t]=forcwgt_y0[t,t+1:T]
        #forcwgt_y0[t,T-t:]=0
        #forcwgt_y0[t,0:t+1]=0 ****** this shoudl not have been zeroed out before! **************
    return forcwgt_y0


@njit
def calc_news_eff(T, delta):
    vdelta=1/(1+delta)

    deltaseq=vdelta*np.power(delta*vdelta,np.arange(T-1))    

    fts0=np.zeros(T)

    for t in range(0,T):

        fts0[t]=np.sum(deltaseq[0:t+1]) # coef on news effect at time t 
    return fts0,deltaseq

@njit(parallel=True)
def compute_Jbehave(Fkin,fts0,extrap_effg):

    '''
    - extrap_effg is the extrapolative effect, as you go down the rows you move forward in time in terms of the extrapolative effect -gamma*delta**k/(1+delta)**(k+1) from k==0,
    along the columns you move forward in terms of distance from impact i.e. F_:,j
    '''

    T=Fkin.shape[0]
    Jkg=np.zeros(Fkin.shape)
    for t in prange(T):
        for s in prange(0,T):        
            for k in range(0,t+1):
                if k<s:
                    Jkg[t,s]+=Fkin[t-k,s-k]*fts0[k]
                else:
                    Jkg[t,s]+=extrap_effg[k-s,t-k]
    return Jkg


@njit
def behave_forc(pt,gamma0,delta0):
    T=pt.shape[0]
    fpt=np.zeros((T,T))

    vdelta0=1/(1+delta0)

    fpt[0]=pt[0]
    fpt[0,1:]=vdelta0*(pt[1:]-gamma0*pt[0])

    for tt in range(1,T):
        fpt[tt,:tt+1]=pt[:tt+1]
        fpt[tt,tt+1:]=vdelta0*(delta0*fpt[tt-1,tt+1:T]+pt[tt+1:T]-gamma0*pt[tt])
        
    return fpt



#@njit
def extrap_jacob(Jk,Fk,gamma,p,type,ss0,theta=0,T=None): # make Extrapolative Jacobian from fake news matrix for all inputs and outputs
    # expected prices are AR1 in levels from current price

    if T is None:
        T=Jk['C']['r'].shape[0]

    Jkout=copy.deepcopy(Jk)

    if type=='behave':
        fts0,deltaseq=calc_news_eff(T,theta)
        extrap_effg=calc_extrap_eff(T,gamma,theta)
    

    for i in Fk.outputs:

        for j in p: # selected prices
            if type=='behave':
                Jkout[i][j]=compute_Jbehave(Fk[i][j],fts0,extrap_effg@Fk[i][j].T)

    return Jkout




########################################################################################################
## IRF loss function
########################################################################################################

from behave_algo import redform_jacob


def dynIRFs(x,modssin,hhJk,hhFk,dynmod,irftarget,irfW,sdsIRF,unknowns,targets,exogenous,T,Etype):

    # load in dynamic parameters

    modssout=copy.deepcopy(modssin) 

    modssout['kappa']=x[0]*1
    modssout['kappaw']=x[1]*1
    modssout['phi']=x[2]*1
    modssout['gammatax']=x[3]*1
    modssout['rhom']=x[4]*1
    modssout['phiy']=x[5]*1
    
    # alter jacobian

    if Etype==2: #sticky expectations
        modssout['deltaWK']=1/(1-x[6])-1
        modssout['deltaWKph']=1/(1-x[7])-1

        hhJk=redform_jacob(hhJk,hhFk,['r','pr','w','hours','Tax','Div','ph','pi'],0.0,modssout['deltaWK'])

    if Etype==5:  # sticky and extrapolative expectations
        modssout['deltaWK']=1/(1-x[6])-1
        modssout['gammaWK']=x[7]*1       
        hhJk=redform_jacob(hhJk,hhFk,['r','pr','w','hours','Tax','Div','ph','pi'],modssout['gammaWK'],modssout['deltaWK'],Trunc=350)


    if Etype==6:  # sticky and extrapolative expectations (seperate param for house price)
        modssout['deltaWK']=1/(1-x[6])-1
        modssout['deltaWKph']=1/(1-x[7])-1
        modssout['gammaWK']=x[8]*1
        modssout['gammaWKph']=x[9]*1

        hhJk=redform_jacob(hhJk,hhFk,['r','pr','w','hours','Tax','Div','ph','pi'],modssout['gammaWK'],modssout['deltaWK'],Trunc=350)
        hhJk=redform_jacob(hhJk,hhFk,['ph'],modssout['gammaWKph'],modssout['deltaWKph'],Trunc=350)

    # solve model dynamics 

    Gsolve = dynmod.solve_jacobian(modssout, unknowns, targets, exogenous, T=T,Js={'hh': hhJk})

    # Model IRFs

    #targetr=np.zeros(T)
    #targetr[0:12]=irftarget[:,0]/400
    #targetr[12:T]=irftarget[11,0]*np.power(modssout['rhom'],np.arange(1,T-11))/400
    #epsr=np.linalg.pinv(Gsolve['i']['epsr'])@targetr

    epsr=np.arange(T)*0.0
    epsr[0]=0.0025

    IRFi=(Gsolve['i']['epsr']@epsr)[0:20]
    IRF1yr=(1+IRFi[0:12])*(1+IRFi[1:13])*(1+IRFi[2:14])*(1+IRFi[3:15])-1
    
    scal=0.25/(IRFi[0])
    #scal=1

    model_BR=scal*IRFi[0:12]*4
    model_1yr=scal*IRF1yr
    model_cpi=np.cumsum(Gsolve['pi']['epsr']@epsr*scal)[0:12]
    model_gdp=(Gsolve['Y']['epsr']@epsr*scal)[0:12]
    model_hpi=(Gsolve['ph']['epsr']@epsr*scal)[0:12]+model_cpi
    model_rent=(Gsolve['rentindex']['epsr']@epsr*scal)[0:12]/modssout['rentindex']+model_cpi   

    modirfs=np.array([model_BR,model_gdp,model_cpi,model_hpi,model_rent]).transpose()

    # loss function
    
    irflossC=(modirfs-irftarget)

    irfloss=irflossC.flatten(order='F')[np.newaxis,:]@irfW@irflossC.flatten(order='F')[:,np.newaxis]

    #print(x)
    #print(irfloss)
    
    return irfloss,modirfs,modssout,Gsolve,hhJk

def irfloss(x,modssin,hhJk,hhFk,dynmod,irftarget,irfW,sdsIRF,unknowns,targets,exogenous,T,Etype):

    irflossV,modirfs,modssout, Gsolve,hhJk=dynIRFs(x,modssin,hhJk,hhFk,dynmod,irftarget,irfW,sdsIRF,unknowns,targets,exogenous,T,Etype)

    return irflossV


########################################################################################################
# adding fake news matrix to stage block
########################################################################################################

from sequence_jacobian.classes import SteadyStateDict, JacobianDict

from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.block import Block
from sequence_jacobian.blocks.het_block import HetBlock
from typing import  Dict, Optional, List

T=4*100


def _jacobianplusF2(self, ss, inputs, outputs, T):
    ss = self.extract_ss_dict(ss)
    outputs = self.M_outputs.inv @ outputs
    differentiable_hetinput = self.preliminary_hetinput(ss, h=1E-4)
    backward_data, forward_data, expectations_data = self.preliminary_all_stages(ss)

    # step 1
    curlyYs, curlyDs = {}, {}
    for i in inputs:
        curlyYs[i], curlyDs[i] = self.backward_fakenews(i, outputs, T, backward_data, forward_data, differentiable_hetinput)
    
    # step 2
    curlyEs = {}
    for o in outputs:
        curlyEs[o] = self.expectation_vectors(o, T-1, expectations_data)

    # steps 3-4
    F, J = {}, {}
    for o in outputs:
        for i in inputs:
            if o.upper() not in F:
                F[o.upper()] = {}
            if o.upper() not in J:
                J[o.upper()] = {}
            F[o.upper()][i] = HetBlock.build_F(curlyYs[i][o], curlyDs[i], curlyEs[o])
            J[o.upper()][i] = HetBlock.J_from_F(F[o.upper()][i])
    
    return JacobianDict(J, name=self.name, T=T),JacobianDict(F, name=self.name, T=T)




def jacobianplusF2(self, ss: SteadyStateDict, inputs: List[str],
            outputs: Optional[List[str]] = None,
            T: Optional[int] = None, Js: Dict[str, JacobianDict] = {},
            options: Dict[str, dict] = {}, **kwargs) -> JacobianDict:
    """Calculate a partial equilibrium Jacobian to a set of `input` shocks at a steady state `ss`."""
    own_options = self.get_options(options, kwargs, 'jacobian')
    inputs = self.make_ordered_set(inputs)
    outputs, _ = self.process_outputs(ss, {}, self.make_ordered_set(outputs))

    J, F = self._jacobianplusF2(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, T=T,**own_options)

    return J, F  # Return both J and F


######################################################################################################
## dynamic income variances
######################################################################################################


@njit
def income_variance_n_quarters(z, pi_z, z_markov, n):
    """
    Compute the variance of n-quarter change in log income.
    
    Parameters:
    z: array of income values
    pi_z: stationary distribution
    z_markov: transition matrix
    n: number of quarters
    
    Returns:
    variance of n-quarter log income change
    """
    nz = len(z)
    log_z = np.log(z)
    
    # Compute n-step transition matrix
    trans_n = z_markov.copy()
    trans_n= trans_n.T
    for i in range(n-1):
        trans_n = trans_n @ z_markov

    trans_n=trans_n.T
        
    # Compute E[log(z_t+n) - log(z_t)]^2
    variance = 0.0
    for i in range(nz):
        for j in range(nz):
            log_diff = log_z[j] - log_z[i]
            variance += pi_z[i] * trans_n[i, j] * log_diff**2
    
    return variance



##################################################################################
## untargetd moments
##################################################################################

def weighted_percentile(a, d, percentile):
    # Flatten the arrays
    a_flat = a.flatten()
    d_flat = d.flatten()
    
    # Sort the arrays based on the values array
    sorted_indices = np.argsort(a_flat)
    a_sorted = a_flat[sorted_indices]
    d_sorted = d_flat[sorted_indices]
    
    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(d_sorted)
    cdf /= cdf[-1]  # Normalize to turn into a proper CDF
    
    # Find the value at the given percentile
    percentile_value = np.interp(percentile / 100.0, cdf, a_sorted)
    
    return percentile_value


###################################################################################
## rental yield calculation
###################################################################################

def rentalyield3(Dpr,Dph,ss,com=True,dev=True):
    # Probabilistic holding period: contract ends each period with prob pr_calvo.
    # Investor locks in the entry-period rent; receives it each period the contract stays active.
    # On termination receives the house price but no rent that period.
    # Expected holding period = 1/pr_calvo quarters; annualise over that horizon.

    calvo=ss['pr_calvo']
    k_eff=1.0/calvo  # expected holding period in quarters

    prss=np.exp(ss['pr'])
    phss=np.exp(ss['ph'])*ss['Fsize']

    pr=np.exp(Dpr)*prss-ss['hmcost']*ss['Fsize']
    ph=(np.exp(Dph))*phss

    if com:
        pr=pr-ss['Fcm']

    T=len(pr)
    expected_i1=np.zeros(T)

    for t in range(T):
        ei1=pr[t]          # period 0: rent always received
        survive=1.0
        for s in range(1,T-t):
            p_end=survive*calvo
            survive=survive*(1-calvo)
            ei1+=p_end*ph[t+s]+survive*pr[t]
        # sell at last available price for any residual survival probability
        ei1+=survive*ph[T-1]
        expected_i1[t]=ei1

    if com:
        i0=ph
        i1_adj=expected_i1
    else:
        i0=ph+ss['transac']
        i1_adj=expected_i1-ss['transac']

    f=(i1_adj/i0)**(4.0/k_eff)-1

    # steady-state yield: geometric series gives total expected rent = pr_ss/calvo, plus house price
    if com:
        ssyield=((prss-ss['hmcost']*ss['Fsize']-ss['Fcm'])/calvo+phss)/phss
    else:
        ssyield=((prss-ss['hmcost']*ss['Fsize'])/calvo+phss-ss['transac'])/(phss+ss['transac'])
    ssyield=ssyield**(4.0/k_eff)-1

    if dev:
        f=f-ssyield

    return f


def rentalyield2(Dpr,Dph,ss,com=True,dev=True,k=4):
    
    prss=np.exp(ss['pr'])
    phss=np.exp(ss['ph'])*ss['Fsize']
                
    pr=np.exp(Dpr)*prss-ss['hmcost']*ss['Fsize']
    ph=(np.exp(Dph))*phss

    if com:
        pr=pr-ss['Fcm']
    
    if com:
        i0=ph[0:-k]
        i1=k*pr[0:-k]+ph[k:]
    else:
        i0=ph[0:-k]+ss['transac']
        i1=k*pr[0:-k]+ph[k:]-ss['transac']

    f=(i1/i0)**(4/k)-1

    if com:
        ssyield=((k*(prss-ss['hmcost']*ss['Fsize']-ss['Fcm'])+phss)/phss)**(4/k)-1
    else:
        ssyield=((k*(prss-ss['hmcost']*ss['Fsize'])-ss['transac']+phss)/(phss+ss['transac']))**(4/k)-1

    if dev:
        f=f-ssyield

    return f


def priceduration(eta, phi, beta):
    """
    Solves for theta where eta/phi = (1-theta)(1-theta*beta)/theta
    
    Rearranging: eta/phi = (1-theta)(1-theta*beta)/theta
                 eta*theta/phi = (1-theta)(1-theta*beta)
                 eta*theta/phi = 1 - theta - theta*beta + theta^2*beta
                 theta^2*beta - theta*(1 + beta + eta/phi) + 1 = 0
    
    This is a quadratic equation: beta*theta^2 - (1 + beta + eta/phi)*theta + 1 = 0
    
    Parameters:
    -----------
    eta : float
        Parameter eta
    phi : float
        Parameter phi
    beta : float
        Parameter beta
        
    Returns:
    --------
    theta : float
        The solution for theta
    """
    import numpy as np
    
    # Quadratic coefficients: a*theta^2 + b*theta + c = 0
    a = beta
    b = -(1 + beta + eta/phi)
    c = 1
    
    # Solve using quadratic formula
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        raise ValueError("No real solution exists for the given parameters")
    
    theta1 = (-b + np.sqrt(discriminant)) / (2*a)
    theta2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    # Return the solution that makes economic sense (typically 0 < theta < 1)
    # Choose the solution in (0, 1) if possible
    solutions = [theta1, theta2]
    valid_solutions = [t for t in solutions if 0 < t < 1]
    
    if valid_solutions:
        return 1/(1-valid_solutions[0])
    else:
        # If no solution in (0,1), return the positive one
        positive_solutions = [t for t in solutions if t > 0]
        if positive_solutions:
            return positive_solutions[0]
        else:
            return theta1  # Return first solution as fallback
