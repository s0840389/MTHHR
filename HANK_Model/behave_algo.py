import numpy as np
from numba import njit, prange
import copy

# Kohlas & Wahlter

@njit
def calc_extrap_eff(T, gamma, delta,Trunc=20,TruncDecay=0.86):
    
    vdelta=1/(1+delta)

    forcwgt_y0=np.zeros((T,T)) # matrix describing the contribution of y0 to to forecast of p_s at time t
    forcwgt_y0[0,0]=1
    forcwgt_y0[0,1:Trunc]=-gamma*vdelta# At T-k we assume they expected steady state
    forcwgt_y0[0,Trunc:T-1]=-gamma*vdelta*np.power(TruncDecay, np.arange(T-Trunc-1)) # At T-k we assume they expected steady state
    
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


def redform_jacob(Jk,Fk,p,gamma,delta,Trunc=20,TruncDecay=0.86): # make Extrapolative Jacobian from fake news matrix for all inputs and outputs


    Jkout=copy.deepcopy(Jk)
    T=Fk[Jk.outputs[0]][Jk.inputs[0]].shape[0]

    fts0,deltaseq=calc_news_eff(T,delta)
    extrap_eff=calc_extrap_eff(T,gamma,delta,Trunc,TruncDecay)
    
    for i in Fk.outputs:

        for j in p: # selected prices
            try:
                Jkout[i][j]=compute_Jbehave(Fk[i][j],fts0,extrap_eff@Fk[i][j].T)
            except:
                pass

    # because we are amending a deep copy of the FIRE Jacobian, if j input is not in the selected prices, then function just returns
    # the FIRE jacobian (key for discount factor shock)
    return Jkout