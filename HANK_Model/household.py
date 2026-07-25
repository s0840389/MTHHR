

import numpy as np
from numba import njit, prange
from copy import deepcopy

from sequence_jacobian import grids, interpolate

from sequence_jacobian.utilities.discretize import nonlinspace


from aux_fns import  mapgrid, mapgrid_inv



############################################################################
## Utility functions
############################################################################

 # utility and marginal utility functions
@njit
def util_c_KMV(c, h, eis,phihs): # utility function with housing 
    
    crra=1/eis
    
    # cobb douglas

    if crra==1:
        u=np.log(c**(1-phihs)*h**phihs)
    else:
        u=1/(1-crra)*(c**(1-phihs)*h**phihs)**(1-crra)

 
    u=u*(c>0)-100000*(c<=0)

    return u


@njit
def marg_util_c_KMV(c, h, eis,phihs): # marginal consumption utility 
    
    crra=1/eis



    # cobb douglas
    uc=(1-phihs)*c**(-phihs)*h**phihs*(c**(1-phihs)*h**phihs)**(-crra)
    
    uc=uc*(c>0)+100000*(c<=0)    

    return uc 

@njit
def inv_marg_util_c_KMV(uc, h, eis,phihs): # marginal consumption utility 
    
    crra=1/eis

    # linear 
    #c=uc**(-1/crra)

    # attanasio, low et al 2012
    #c=(uc/np.exp(phihs*h))**(-1/crra)

    # cobb douglas
    c=(uc/((1-phihs)*h**(phihs*(1-crra))))**(1/(-crra-phihs+crra*phihs)) 
    return c

#####################################################################################################
## Mixex AR1 income process
#####################################################################################################



def mixedAR1(rho1,sig1,n1,rho2,sig2,n2,minz):

    z1,z1pi,z1markov=grids.markov_rouwenhorst(rho1,(sig1**2/(1-rho1**2))**0.5,n1)  
    z2,z2pi,z2markov=grids.markov_rouwenhorst(rho2,(sig2**2/(1-rho2**2))**0.5,n2)

    z=np.kron(z1,z2)
    z_markov=np.kron(z1markov,z2markov)

    ez,vz=np.linalg.eig(z_markov.T)    

    ezi=np.argmin(np.abs(ez-1))

    z_pi=np.abs(vz[:,ezi])/np.sum(np.abs(vz[:,ezi]))

    z=np.maximum(z,np.ones_like(z)*minz) # avoid very low values of z 

    zmu=np.sum(z*z_pi)

    z=z/zmu


    return z,z_pi,z_markov



################################################################################
## Intialize household
################################################################################


def make_grids(rho_z1, sd_z1, n_z1,rho_z2, sd_z2, n_z2, n_h,min_z, min_a, max_a, n_a,borlim,phss,Fsize,omegah,pr_calvo,kappah,kappahLL,prss):
    
    #z_grid, z_dist, z_markov = grids.markov_rouwenhorst(rho_z, sd_z, n_z)
    z_grid,z_dist,z_markov=mixedAR1(rho_z1, sd_z1, n_z1, rho_z2, sd_z2, n_z2,min_z) 
    n_z=n_z1*n_z2

    #def nonlinspace(amax, n, phi, amin=0):
        # If phi =1 then points are equidistant. If phi > 1 then points are bunched near min point. Large phi, dense near the min.
    a_grid = nonlinspace(1.0,n_a ,1.0, amin = 0.0) # normalized grid between zero and one with n_a number of points

    cutvaluesR=np.quantile(a_grid,[0,0.6,0.95,1.0],interpolation='higher') # this makes sure that the grid is mapped to points on the normalised grid
    cutvaluesO=np.quantile(a_grid,[0,0.6,0.95,1.0],interpolation='higher')
    cutvaluesLL=np.quantile(a_grid,[0,0.6,0.95,1.0],interpolation='higher')
    cutvaluesOF=np.quantile(a_grid,[0,0.6,0.95,1.0],interpolation='higher')

    GbreakR=np.array([borlim,4,phss,max_a])
    GbreakO=np.array([-phss,0,phss,max_a])
    GbreakLL=np.array([-1.0*phss*(kappah+kappahLL*Fsize),0,phss,max_a])
    GbreakLL2=np.array([-1.0*phss*(kappah+kappahLL*Fsize),0,phss,max_a])
    GbreakOF=np.array([-phss*Fsize,0,phss,max_a])    


    a_grid_r=mapgrid(a_grid,cutvaluesR,GbreakR) # owners grid
    a_grid_o=mapgrid(a_grid,cutvaluesO,GbreakO) # owners grid
    a_grid_ll=mapgrid(a_grid,cutvaluesLL,GbreakLL) # landlords grid
    a_grid_of=mapgrid(a_grid,cutvaluesOF,GbreakOF) # flatowners grid
    a_grid_ll2=mapgrid(a_grid,cutvaluesLL,GbreakLL2) # landlords grid


    # this is end of period assets

    a_grid_exp_t1 = np.zeros((n_h,n_a))
    
    a_grid_exp_t1[0] = a_grid_o # own to own
    a_grid_exp_t1[1] = a_grid_r # own to rent
    a_grid_exp_t1[2] = a_grid_of # rent to ownF
    a_grid_exp_t1[3] = a_grid_r # rent to rent

    a_grid_exp_t1[4] = a_grid_ll # own to landlord
    a_grid_exp_t1[5] = a_grid_o # landlord to own
    a_grid_exp_t1[6] = a_grid_ll # landlord to landlord

    a_grid_exp_t1[7] = a_grid_of # own to ownF

    a_grid_exp_t1[8] = a_grid_o # ownF to own
    a_grid_exp_t1[9] = a_grid_of # ownF to ownF
    a_grid_exp_t1[10] = a_grid_r # ownF to rent

    a_grid_exp_t1[11] = a_grid_ll # landlord to landlord2
    a_grid_exp_t1[12] = a_grid_ll2 # landlord2 to landlord
    a_grid_exp_t1[13] = a_grid_ll2 # landlord2 to landlord2
    
    a_grid_exp_t1[14] = a_grid_r # rent to rent
    a_grid_exp_t1[15] = a_grid_ll # landlord to landlord
    a_grid_exp_t1[16] = a_grid_ll2 # landlord2 to landlord2


    # this is beginning of period assets
    a_grid_exp_t0 = np.zeros((n_h,n_a))

    a_grid_exp_t0[0] = a_grid_o # own to own
    a_grid_exp_t0[1] = a_grid_o # own to rent
    a_grid_exp_t0[2] = a_grid_r # rent to ownF
    a_grid_exp_t0[3] = a_grid_r # rent to rent

    a_grid_exp_t0[4] = a_grid_o # own to landlord
    a_grid_exp_t0[5] = a_grid_ll # landlord to own
    a_grid_exp_t0[6] = a_grid_ll # landlord to landlord

    a_grid_exp_t0[7] = a_grid_o # own to ownF

    a_grid_exp_t0[8] = a_grid_of # ownF to own
    a_grid_exp_t0[9] = a_grid_of # ownF to ownF
    a_grid_exp_t0[10] = a_grid_of # ownF to rent

    a_grid_exp_t0[11] = a_grid_ll2 # landlord to landlord2
    a_grid_exp_t0[12] = a_grid_ll # landlord2 to landlord
    a_grid_exp_t0[13] = a_grid_ll2 # landlord2 to landlord2

    a_grid_exp_t0[14] = a_grid_r # rent to rent
    a_grid_exp_t0[15] = a_grid_ll # landlord to landlord
    a_grid_exp_t0[16] = a_grid_ll2 # landlord2 to landlord2


    hsizes=np.ones([n_h])*omegah
    hsizes[[1,3,10,14]]=Fsize
    hsizes[[2,7,9]]=Fsize*omegah

    h_markov=np.eye(n_h)

    #renters in preivous period

    h_markov[3,14]=pr_calvo
    h_markov[3,3]=1-pr_calvo

    h_markov[14,14]=pr_calvo
    h_markov[14,3]=1-pr_calvo

    h_markov[10,14]=pr_calvo
    h_markov[10,3]=1-pr_calvo
    h_markov[10,10]=0
    
    h_markov[1,14]=pr_calvo
    h_markov[1,3]=1-pr_calvo
    h_markov[1,1]=0

    # landlord in previous period
    h_markov[6,15]=pr_calvo
    h_markov[6,6]=1-pr_calvo
    
    h_markov[15,15]=pr_calvo
    h_markov[15,6]=1-pr_calvo
    
    h_markov[4,15]=pr_calvo
    h_markov[4,6]=1-pr_calvo
    h_markov[4,4]=0

    h_markov[12,15]=pr_calvo
    h_markov[12,6]=1-pr_calvo
    h_markov[12,12]=0

    # landlord2 in previous period
    h_markov[13,16]=pr_calvo
    h_markov[13,13]=1-pr_calvo
    
    h_markov[16,16]=pr_calvo
    h_markov[16,13]=1-pr_calvo

    h_markov[11,16]=pr_calvo
    h_markov[11,13]=1-pr_calvo
    h_markov[11,11]=0


    # rental price grid

    pri_grid=np.array([np.log(np.exp(prss)/(1+1E-4)),prss,prss+1E-4])
    pri_grid2=np.array([-1E-4,0.0,1E-4])

    a_grid_full=np.zeros(n_z)[np.newaxis,:,np.newaxis,np.newaxis] + a_grid_exp_t1[:,np.newaxis,:,np.newaxis]+0*pri_grid[np.newaxis,np.newaxis,np.newaxis,:]


    pri_grid_full=np.zeros_like(a_grid_full)
    pri_grid_full[[1,3,4,6,10,11,12,13,14,15,16],:,:,:]=pri_grid[np.newaxis,np.newaxis,np.newaxis,:] + np.zeros((n_z,n_a,1))
    pri_grid_full[[0,2,5,7,8,9],:,:,:]=pri_grid2[np.newaxis,np.newaxis,np.newaxis,:] + np.zeros((n_z,n_a,1))

    # initial grid 
    a_grid_full_t0=np.zeros(n_z)[np.newaxis,:,np.newaxis,np.newaxis] + a_grid_exp_t0[:,np.newaxis,:,np.newaxis]+0*pri_grid[np.newaxis,np.newaxis,np.newaxis,:]

    return z_grid, z_dist, z_markov, a_grid , a_grid_exp_t0, a_grid_exp_t1,a_grid_r,a_grid_o,a_grid_ll,a_grid_of,a_grid_full,hsizes,h_markov,n_z,pri_grid,pri_grid_full,a_grid_full_t0


def prices_baseline(r,ph,Fsize):
    rmort=r*1
    phF=np.log(np.exp(ph)*Fsize) # rental flat price
    return rmort, phF

def prices_mort(ph,Fsize):
    phF=np.log(np.exp(ph)*Fsize) # rental flat price
    return phF

def prices_seg(r):
    rmort=r*1
    return rmort

def labor_income( z_grid,z_dist, r,rmort, w,ph, phF,transac, pr,hours,Tax,Div,MPC,a_grid_exp_t0,borwedge,Fsize,n_h,kappah,kappahLL,kappay,borlim,hmcost,TransfAgg,pri_grid,pri_grid_full,pi):

    # interest rate 
    nz=z_grid.shape[0]
    na=a_grid_exp_t0.shape[1]

    # state contigent interst rate

    agridexpand=np.zeros([nz,])[np.newaxis,:,np.newaxis,np.newaxis]+a_grid_exp_t0[:,np.newaxis,:,np.newaxis]+0*pri_grid[np.newaxis,np.newaxis,np.newaxis,:] # on (n,z,a,pr)
    rexpand=r*np.ones_like(agridexpand) 

    rexpand[agridexpand<0] = rmort + borwedge
 
    divi=np.exp(Div)*z_grid/(np.sum(z_grid*z_dist)) # incidence rule for dividend

    labinc=z_grid *np.exp(w)*np.exp(hours)*(1-Tax) # labour income
    
    MPC=MPC*np.ones([n_h,])
    
    # disposable income
    disinc=labinc[np.newaxis,:,np.newaxis,np.newaxis]+\
        divi[np.newaxis,:,np.newaxis,np.newaxis]+0*a_grid_exp_t0[:,np.newaxis, :,np.newaxis]+0*pri_grid[np.newaxis,np.newaxis,np.newaxis,:]+\
        MPC[:,np.newaxis,np.newaxis,np.newaxis]+TransfAgg# on (n,z,a)


    housecost=np.array([0-hmcost,  np.exp(ph) - transac-0,  # o/o, o/r
                         -Fsize*np.exp(ph)-transac-Fsize*hmcost, -0, # r/oF, r/r
                         0-np.exp(phF)-transac-hmcost*(1+Fsize),np.exp(phF) - transac-hmcost, 0-hmcost*(1+Fsize), # o/l, l/o, l/l
                         np.exp(ph)-Fsize*np.exp(ph)-2*transac-hmcost*Fsize, # o/oF
                         -np.exp(ph)+Fsize*np.exp(ph)-2*transac-hmcost,0-hmcost*Fsize,Fsize*np.exp(ph)-transac-0, # oF/o, oF/oF, oF/r
                         2*0-np.exp(phF)-transac-hmcost*(1+2*Fsize),0+np.exp(phF) - transac-hmcost*(1+Fsize), 2*0-hmcost*(1+2*Fsize), # l/l2, l2/l, l2/l2
                         -0,0-hmcost*(1+Fsize),2*0-hmcost*(1+2*Fsize)]) #r/r* , ll/ll*, ll2/ll2*


    rentcost=pri_grid_full*0
    rentcost[[3],:,:,:]=-1*np.exp(pri_grid_full[[3],:,:,:])/(1+pi) # renters pay rent
    rentcost[[6],:,:,:]=np.exp(pri_grid_full[[6],:,:,:])/(1+pi) # landlords receive rent
    rentcost[[13],:,:,:]=2*np.exp(pri_grid_full[[13],:,:,:])/(1+pi) #  landlords receive rent

    rentcost[[1,10,14],:,:,:]=-np.exp(pr) # get spot sprice price
    rentcost[[4,12,15],:,:,:]=np.exp(pr)
    rentcost[[11,16],:,:,:]=2*np.exp(pr)

    # non asset part of budget constrain
    y = labinc[np.newaxis,:,np.newaxis,np.newaxis]+divi[np.newaxis,:,np.newaxis,np.newaxis]+\
        housecost[:,np.newaxis,np.newaxis,np.newaxis] + MPC[:,np.newaxis,np.newaxis,np.newaxis] + rentcost+TransfAgg # on (n,z)
    
    # cash on hand today
    coh=disinc+(1+rexpand)*agridexpand+housecost[:,np.newaxis,np.newaxis,np.newaxis]+rentcost # on (n, z, a)

        # borrowing constraint 
    bchh=np.zeros_like(coh)
    bchh[0]=np.minimum(agridexpand[0],np.maximum(-np.exp(ph)*kappah,-disinc[0]*kappay)) # own to own
    bchh[1]=borlim # own to rent
    bchh[2]=np.maximum(-Fsize*np.exp(ph)*kappah,-disinc[2]*kappay) # rent to ownF
    bchh[3]=borlim # rent to rent

    bchh[4]=np.maximum(-np.exp(ph)*kappah-np.exp(phF)*kappahLL,-np.exp(phF)*kappahLL-disinc[4]*kappay) # own to landlord
    bchh[5]=np.minimum(agridexpand[5]+np.exp(phF)-transac,np.maximum(-np.exp(ph)*kappah,-disinc[5]*kappay) ) # landlord to own
    bchh[6]=np.minimum(agridexpand[6],np.maximum(-np.exp(ph)*kappah-np.exp(phF)*kappahLL,-np.exp(phF)*kappahLL-disinc[6]*kappay) ) # landlord to landlord

    bchh[7]=np.maximum(-Fsize*np.exp(ph)*kappah,-disinc[7]*kappay) # own to ownF

    bchh[8]=np.maximum(-np.exp(ph)*kappah,-disinc[8]*kappay) # ownF to own
    bchh[9]=np.minimum(agridexpand[9],np.maximum(-Fsize*np.exp(ph)*kappah,-disinc[9]*kappay)) # ownF to ownF
    bchh[10]=borlim # ownF to rent
    
    bchh[11]=np.maximum(-np.exp(ph)*kappah-np.exp(phF)*kappahLL,-np.exp(phF)*kappahLL-disinc[11]*kappay) # landlord to landlord2
    bchh[12]=np.minimum(agridexpand[12]+np.exp(phF)-transac,np.maximum(-np.exp(ph)*kappah-np.exp(phF)*kappahLL,-np.exp(phF)*kappahLL-disinc[12]*kappay) ) # landlord2 to landlord
    bchh[13]=np.minimum(agridexpand[13],np.maximum(-np.exp(ph)*kappah-np.exp(phF)*kappahLL,-np.exp(phF)*kappahLL-disinc[13]*kappay) ) # landlord2 to landlord2

    bchh[14]=borlim # rent to rent*
    bchh[15]=np.minimum(agridexpand[15],np.maximum(-np.exp(ph)*kappah-np.exp(phF)*kappahLL,-np.exp(phF)*kappahLL-disinc[15]*kappay) ) # landlord to landlord
    bchh[16]=np.minimum(agridexpand[16],np.maximum(-np.exp(ph)*kappah-np.exp(phF)*kappahLL,-np.exp(phF)*kappahLL-disinc[16]*kappay) ) # landlord2 to landlord2


    return disinc, coh, y,rexpand,bchh,agridexpand,rentcost




def labor_income_RELL( z_grid,z_dist, r,rmort, w,ph, phF,transac, pr,hours,Tax,Div,MPC,a_grid_exp_t0,borwedge,Fsize,n_h,kappah,kappahLL,kappay,borlim,hmcost,TransfAgg,pri_grid,pri_grid_full,pi,rLL,prLL,phLL):

    phFLL=np.log(np.exp(phLL)*Fsize) # rental flat price

    # interest rate 
    nz=z_grid.shape[0]
    na=a_grid_exp_t0.shape[1]

    # state contigent interst rate

    agridexpand=np.zeros([nz,])[np.newaxis,:,np.newaxis,np.newaxis]+a_grid_exp_t0[:,np.newaxis,:,np.newaxis]+0*pri_grid[np.newaxis,np.newaxis,np.newaxis,:] # on (n,z,a,pr)
    rexpand=r*np.ones_like(agridexpand) 

    # interest rate assignment 
    rexpand[agridexpand<0] = rmort + borwedge

    LLints=[5,6,11,12,13,15,16] # starting landlord states

    for i in LLints:
        rexpand[i][agridexpand[i]<0] = rLL+borwedge  # accomdoating different expectations for landlords
        rexpand[i][agridexpand[i]>=0] = rLL 


    divi=np.exp(Div)*z_grid/(np.sum(z_grid*z_dist)) # incidence rule for dividend

    labinc=z_grid *np.exp(w)*np.exp(hours)*(1-Tax) # labour income
    
    MPC=MPC*np.ones([n_h,])
    
    # disposable income
    disinc=labinc[np.newaxis,:,np.newaxis,np.newaxis]+\
        divi[np.newaxis,:,np.newaxis,np.newaxis]+0*a_grid_exp_t0[:,np.newaxis, :,np.newaxis]+0*pri_grid[np.newaxis,np.newaxis,np.newaxis,:]+\
        MPC[:,np.newaxis,np.newaxis,np.newaxis]+TransfAgg# on (n,z,a)


    housecost=np.array([0-hmcost,  np.exp(ph) - transac-0,  # o/o, o/r
                         -Fsize*np.exp(ph)-transac-Fsize*hmcost, -0, # r/oF, r/r
                         0-np.exp(phFLL)-transac-hmcost*(1+Fsize),np.exp(phFLL) - transac-hmcost, 0-hmcost*(1+Fsize), # o/l, l/o, l/l
                         np.exp(ph)-Fsize*np.exp(ph)-2*transac-hmcost*Fsize, # o/oF
                         -np.exp(ph)+Fsize*np.exp(ph)-2*transac-hmcost,0-hmcost*Fsize,Fsize*np.exp(ph)-transac-0, # oF/o, oF/oF, oF/r
                         2*0-np.exp(phFLL)-transac-hmcost*(1+2*Fsize),0+np.exp(phFLL) - transac-hmcost*(1+Fsize), 2*0-hmcost*(1+2*Fsize), # l/l2, l2/l, l2/l2
                         -0,0-hmcost*(1+Fsize),2*0-hmcost*(1+2*Fsize)]) #r/r* , ll/ll*, ll2/ll2*


    rentcost=pri_grid_full*0
    rentcost[[3],:,:,:]=-1*np.exp(pri_grid_full[[3],:,:,:])/(1+pi) # renters pay rent
    rentcost[[6],:,:,:]=np.exp(pri_grid_full[[6],:,:,:])/(1+pi) # landlords receive rent
    rentcost[[13],:,:,:]=2*np.exp(pri_grid_full[[13],:,:,:])/(1+pi) #  landlords receive rent

    rentcost[[1,10,14],:,:,:]=-np.exp(pr) # get spot sprice price
    rentcost[[4,12,15],:,:,:]=np.exp(prLL)
    rentcost[[11,16],:,:,:]=2*np.exp(prLL)

    # non asset part of budget constrain
    y = labinc[np.newaxis,:,np.newaxis,np.newaxis]+divi[np.newaxis,:,np.newaxis,np.newaxis]+\
        housecost[:,np.newaxis,np.newaxis,np.newaxis] + MPC[:,np.newaxis,np.newaxis,np.newaxis] + rentcost+TransfAgg # on (n,z)
    
    # cash on hand today
    coh=disinc+(1+rexpand)*agridexpand+housecost[:,np.newaxis,np.newaxis,np.newaxis]+rentcost # on (n, z, a)

        # borrowing constraint 
    bchh=np.zeros_like(coh)
    bchh[0]=np.minimum(agridexpand[0],np.maximum(-np.exp(ph)*kappah,-disinc[0]*kappay)) # own to own
    bchh[1]=borlim # own to rent
    bchh[2]=np.maximum(-Fsize*np.exp(ph)*kappah,-disinc[2]*kappay) # rent to ownF
    bchh[3]=borlim # rent to rent

    bchh[4]=np.maximum(-np.exp(phLL)*kappah-np.exp(phFLL)*kappahLL,-np.exp(phFLL)*kappahLL-disinc[4]*kappay) # own to landlord
    bchh[5]=np.minimum(agridexpand[5]+np.exp(phFLL)-transac,np.maximum(-np.exp(phLL)*kappah,-disinc[5]*kappay) ) # landlord to own
    bchh[6]=np.minimum(agridexpand[6],np.maximum(-np.exp(phLL)*kappah-np.exp(phFLL)*kappahLL,-np.exp(phFLL)*kappahLL-disinc[6]*kappay) ) # landlord to landlord

    bchh[7]=np.maximum(-Fsize*np.exp(ph)*kappah,-disinc[7]*kappay) # own to ownF

    bchh[8]=np.maximum(-np.exp(ph)*kappah,-disinc[8]*kappay) # ownF to own
    bchh[9]=np.minimum(agridexpand[9],np.maximum(-Fsize*np.exp(ph)*kappah,-disinc[9]*kappay)) # ownF to ownF
    bchh[10]=borlim # ownF to rent
    
    bchh[11]=np.maximum(-np.exp(phLL)*kappah-np.exp(phFLL)*kappahLL,-np.exp(phFLL)*kappahLL-disinc[11]*kappay) # landlord to landlord2
    bchh[12]=np.minimum(agridexpand[12]+np.exp(phFLL)-transac,np.maximum(-np.exp(phLL)*kappah-np.exp(phFLL)*kappahLL,-np.exp(phFLL)*kappahLL-disinc[12]*kappay) ) # landlord2 to landlord
    bchh[13]=np.minimum(agridexpand[13],np.maximum(-np.exp(phLL)*kappah-np.exp(phFLL)*kappahLL,-np.exp(phFLL)*kappahLL-disinc[13]*kappay) ) # landlord2 to landlord2

    bchh[14]=borlim # rent to rent*
    bchh[15]=np.minimum(agridexpand[15],np.maximum(-np.exp(phLL)*kappah-np.exp(phFLL)*kappahLL,-np.exp(phFLL)*kappahLL-disinc[15]*kappay) ) # landlord to landlord
    bchh[16]=np.minimum(agridexpand[16],np.maximum(-np.exp(phLL)*kappah-np.exp(phFLL)*kappahLL,-np.exp(phFLL)*kappahLL-disinc[16]*kappay) ) # landlord2 to landlord2


    return disinc, coh, y,rexpand,bchh,agridexpand,rentcost


## households states (n-labour supply,z-productivyt,a-liquid wealth)
def hh_init(disinc, eis,beta,phihs,hsizes):
        
    # this is to indicate owners in utility function

    V = util_c_KMV(disinc, hsizes[:,np.newaxis,np.newaxis,np.newaxis]*np.ones_like(disinc) ,eis,phihs) / (1-beta)

    Va = np.empty_like(V)

    Va=marg_util_c_KMV(disinc, hsizes[:,np.newaxis,np.newaxis,np.newaxis]*np.ones_like(disinc) ,eis,phihs)


    return V, Va

################################################################################
## Stage 2 - Tenure choice
################################################################################



# tenure state is d= {own/own, own/rent, rent/own, rent/rent}


'''
states:
0. own  to own
1. own  to rent
2. rent to ownF
3. rent to rent

4. own to landlord
5. landlord to own
6. landlord to landlord

7. own to ownF

8. ownF to own
9. ownF to ownF
10. ownF to rent
11. landlord to landlord2
12. landlord2 to landlord
13. landlord2 to landlord2
14. rent to rent*
15. landlord to landlord*
16. landlord2 to landlord2* 
'''


def util_l(V,coh,movecost,movecostLL,defaultcost,llcost,borlim,n_h,bchh):
# on (n| n_)

    flow_u=np.ones([n_h,n_h])*-np.inf

    # Valid transitions
    flow_u[[0,1,4,7],0]=[0,-movecost,-movecostLL-llcost,-movecost] # (x |  O/O) 
    flow_u[[2,3],1]=[-movecost,0] # (x |  O/R)
    flow_u[[8,9,10],2]=[-movecost,0,-movecost]# (x |  R/OF) 
    flow_u[[3],3]=[0]# (x |  R/R) 

    flow_u[[5,6,11],4]=[-movecostLL,-llcost,-movecostLL-llcost] # (x |  O/LL) 
    flow_u[[0,1,4,7],5]=[0,-movecost,-movecostLL-llcost,-movecost] # (x |  LL/O) 
    flow_u[[6],6]=[-llcost] # (x |  LL/LL) 

    flow_u[[8,9,10],7]=[-movecost,0,-movecost] # (x |  O/OF)
    flow_u[[0,1,4,7],8]=[0,-movecost,-movecostLL-llcost,-movecost] # (x |  OF/O)
    flow_u[[8,9,10],9]=[-movecost,0,-movecost] # (x |  OF/OF)
    flow_u[[2,3],10]=[-movecost,0] # (x |  OF/R)
    
    flow_u[[12,13],11]=[-movecostLL-llcost,-llcost] # (x |  LL/LL2) 
    flow_u[[5,6,11],12]=[-movecostLL,-llcost,-movecostLL-llcost] # (x |  LL2/LL) 
    flow_u[[13],13]=[-llcost] # (x |  LL2/LL2) 

    flow_u[[2,14],14]=[-movecost,0] # (x |  r/r) 
    flow_u[[5,15,11],15]=[-movecostLL,-llcost,-movecostLL-llcost] # (x |  LL/LL) 
    flow_u[[12,16],16]=[-movecostLL-movecost,-llcost] # (x |  LL2/LL2) 


    # on (n| n_, z, a)
    shape = np.zeros((n_h, n_h,) + V.shape[1:])
    flow_u = flow_u[..., np.newaxis, np.newaxis,np.newaxis] + shape
    

    # Ruling out impossible transitons given borrowing constraints, or persting in bad states
    for i in range(n_h):
        flow_u[i,:,coh[i]-bchh[i]<=0.0]= -np.inf 

    # default
        # default incurs large utily cost but is possible
        # household with negative resources can default on property, in doing so they incur a large utility cost,
        # consume their dissposable income start from the borrowing constraint of the state they transition to 

    flow_u[1,0,( coh[1]   <= borlim  )] = -defaultcost # own/own to own/rent
    flow_u[1,5,( coh[1]   <= borlim  )] = -defaultcost # landlord/own to own/rent
    flow_u[1,8,( coh[1]   <= borlim  )] = -defaultcost # ownF/own to own/rent

    flow_u[5,6,( coh[5]-bchh[5]<=0.0  )] = -defaultcost # landlord/landlord to landlord/own
    flow_u[5,15,( coh[5]-bchh[5]<=0.0  )] = -defaultcost # landlord/landlord to landlord/own
    flow_u[5,4,( coh[5]-bchh[5]<=0.0  )] = -defaultcost # own/landlord to landlord/own
    flow_u[5,12,( coh[5]-bchh[5]<=0.0  )] = -defaultcost # landlord2/landlord to landlord/own


    flow_u[10,9,( coh[10]   <= 0.0  )] = -defaultcost # ownF/ownF to ownF/rent
    flow_u[10,2,( coh[10]   <= 0.0  )] = -defaultcost # rent/ownF to ownF/rent
    flow_u[10,7,( coh[10]   <= 0.0  )] = -defaultcost # own/ownF to ownF/rent

    flow_u[12,13,( coh[12]-bchh[12]<=0.0  )] = -defaultcost # landlord2/landlord2 to landlord2/landlord
    flow_u[12,11,( coh[12]-bchh[12]<=0.0 )] = -defaultcost # landlord/landlord2 to landlord2/landlord
    flow_u[12,16,( coh[12]-bchh[12]<=0.0 )] = -defaultcost # landlord2/landlord2 to landlord2/landlord



    return flow_u



def selling_friction(etasell,n_h):
## selling frictions function 

    # selling prob grid

    # selling happens only with probability etasell (otherwise the intended sale fails and
    # the agent remains in the "no-sale" state)

    sell_markov = np.eye(n_h, dtype=float)

    # mapping: intended_sell_state -> fallback_no_sale_state
    # (based on the state enumeration in the file's comment block)
    sell_map = {
        1: 0,   # own -> rent  : if sale fails remain own->own
        5: 15,  # landlord -> own : if sale fails remain landlord landlord*
        7: 0,  # own -> ownF  : if sale fails remain own->own
        8: 9,   # ownF -> own  : if sale fails remain ownF stays (map to ownF->ownF as fallback)
        10: 9, # ownF -> rent : if sale fails remain ownF->ownF
        12: 16, # landlord2 -> landlord : fallback to landlord2 landlord2*
    }

    # build the matrix: when the intended next-state is a selling state, realized next-state
    # is the intended one with probability alpha and the fallback with probability 1-alpha.
    for intended, fallback in sell_map.items():
        if 0 <= intended < n_h and 0 <= fallback < n_h:
            sell_markov[intended, :] = 0.0
            sell_markov[intended, intended] = float(etasell)
            sell_markov[intended, fallback] = float(1.0 - etasell)

    return sell_markov



################################################################################
## Stage 3 - Consumption Saving
################################################################################


# consumpton savings stage conditional on housing decision, solved using EGM with upper envelope
def dcegm(V, Va, coh,y,rexpand, beta, eis,a_grid_exp_t0,a_grid_exp_t1,disinc,bchh,hsizes,phihs,pr,pri_grid_full,pi):
    
    """DC-EGM algorithm"""
    # use all FOCs on endogenous grid

    W = beta * V                                                  # end-of-stage vfun
    uc_endo = beta * Va                                           # envelope condition
    
    # THE EULER BELOW WILL NEED TO BE CHANGED SLIGHTLY TO ACCOMODATE VALUES OF ALPHAH<1
    c_endo=inv_marg_util_c_KMV(uc_endo,hsizes[:,np.newaxis,np.newaxis,np.newaxis]+np.zeros_like(uc_endo),eis,phihs)
                                         # Euler equation
    a_endo = (c_endo + a_grid_exp_t1[:,np.newaxis,:,np.newaxis] - y ) / (1 + rexpand) 

    # interpolate with upper envelope, compute unconstrained values and policies
    V_uc, c_uc, a_uc = upperenv_vec(W, a_endo, coh, a_grid_exp_t0,a_grid_exp_t1,  eis,hsizes,phihs)

    #enforce borrowing constraint
    V, c, a  = constrain_policies(V_uc, c_uc, a_uc ,coh,eis,W,a_grid_exp_t1,bchh,disinc,hsizes,phihs)
 
    # update Va on exogenous grid
    uc = marg_util_c_KMV(c, hsizes[:,np.newaxis,np.newaxis,np.newaxis]+np.zeros_like(c), eis, phihs)           # Computer marginal utility
    Va = (1 + rexpand) * uc                                             # envelope condition


    pri=pri_grid_full*0
    pri[[3,6,13],:,:,:]=np.log(np.exp(pri_grid_full[[3,6,13],:,:,:])/(1+pi)) # not moving to spot price
    pri[[1,10,14],:,:,:]=pr # get spot sprice price
    pri[[4,12,15,11,16],:,:,:]=pr # get spot sprice price

    return V, Va, a, c, pri


# consumpton savings stage conditional on housing decision, solved using EGM with upper envelope
def dcegm_RELL(V, Va, coh,y,rexpand, beta, eis,a_grid_exp_t0,a_grid_exp_t1,disinc,bchh,hsizes,phihs,pr,pri_grid_full,pi,prLL):
    
    """DC-EGM algorithm"""
    # use all FOCs on endogenous grid

    W = beta * V                                                  # end-of-stage vfun
    uc_endo = beta * Va                                           # envelope condition
    
    # THE EULER BELOW WILL NEED TO BE CHANGED SLIGHTLY TO ACCOMODATE VALUES OF ALPHAH<1
    c_endo=inv_marg_util_c_KMV(uc_endo,hsizes[:,np.newaxis,np.newaxis,np.newaxis]+np.zeros_like(uc_endo),eis,phihs)
                                         # Euler equation
    a_endo = (c_endo + a_grid_exp_t1[:,np.newaxis,:,np.newaxis] - y ) / (1 + rexpand) 

    # interpolate with upper envelope, compute unconstrained values and policies
    V_uc, c_uc, a_uc = upperenv_vec(W, a_endo, coh, a_grid_exp_t0,a_grid_exp_t1,  eis,hsizes,phihs)

    #enforce borrowing constraint
    V, c, a  = constrain_policies(V_uc, c_uc, a_uc ,coh,eis,W,a_grid_exp_t1,bchh,disinc,hsizes,phihs)
 
    # update Va on exogenous grid
    uc = marg_util_c_KMV(c, hsizes[:,np.newaxis,np.newaxis,np.newaxis]+np.zeros_like(c), eis, phihs)           # Computer marginal utility
    Va = (1 + rexpand) * uc                                             # envelope condition


    pri=pri_grid_full*0
    pri[[3,6,13],:,:,:]=np.log(np.exp(pri_grid_full[[3,6,13],:,:,:])/(1+pi)) # not moving to spot price
    pri[[1,10,14],:,:,:]=pr # get spot sprice price
    pri[[4,12,15,11,16],:,:,:]=prLL # get spot sprice price

    return V, Va, a, c, pri


@njit(parallel=True) # need to add prange
def upperenv_vec(W, a_endo, coh, a_grid_exp_t0, a_grid_exp_t1, eis ,hsizes,phihs): # a_grid here is taking in the normalized grid, even though when called I directly specify the tiled grid
    """Interpolate value function and consumption to exogenous grid."""
    n_n, n_z, n_a, n_pr = W.shape 
    a = np.zeros((n_n,n_z,n_a,n_pr)) # asset policy
    c = np.zeros((n_n,n_z,n_a,n_pr)) # consumption policy
    V = -np.inf * np.ones((n_n,n_z,n_a,n_pr))# Value functions

 
    for i_n in prange(n_n): # ownership state
        
        for i_z in prange(n_z): # productivity state
            
            for i_pr in prange(n_pr): # rental price state
            
                for ja in prange(n_a - 1): # loop over a_endo, generating segments of the a_endo grid
                
                    
                    a_low, a_high = a_endo[i_n,i_z, ja, i_pr], a_endo[i_n,i_z, ja + 1, i_pr] 
                    W_low, W_high = W[i_n,i_z, ja, i_pr], W[i_n,i_z, ja + 1, i_pr]
                    ap_low, ap_high = a_grid_exp_t1[i_n,ja], a_grid_exp_t1[i_n, ja + 1]
                        
                   # loop over exogenous asset grid (increasing) 
                    for ia in range(n_a):  
                
                        acur = a_grid_exp_t0[i_n,ia]  # this point is for looping over the segment for this iteration [a_grid is the grid you transition from ]
                                                      # The bounds of the segments for this iteration are a_low,a_high
                                                      # we use this agrid because this periods a_endo will be yesterday's end of period assets
                                                      # So we need the values of V today on the a_grid, this will be tomorrow's endofprd Vs 
                                                      # and used to compute tomorrow's endofprd marginal values
                        coh_cur = coh[i_n,i_z, ia, i_pr]

                        interp = (a_low <= acur <= a_high) 
                        extrap = (ja == n_a - 2) and (acur > a_endo[i_n,i_z, n_a - 1, i_pr])
                        extrap_low = (ja == 0) and (acur < a_endo[i_n,i_z, 0, i_pr]) # This was added evaluate acur points below bottom of segment of the endogenous grid

                        # exploit that a_grid is increasing
                        if (a_high < acur < a_endo[i_n,i_z, n_a - 1, i_pr]):
                            break

                        if interp or extrap or extrap_low:
                            W0 = interpolate.interpolate_point(acur, a_low, a_high, W_low, W_high)
                            a0 = interpolate.interpolate_point(acur, a_low, a_high, ap_low, ap_high)
                            c0 = coh_cur - a0 # This is the consumption today 
                                              # Overall,
                                              # a0 is the end or period assets you would choose if you were had acur amount assets today when making your consumption saving decision
                                              # c0 is the consumption you would choose if you were had acur amount assets today when making your consumption saving decision
                                              # W0 is the value tomorrow if you were had acur amount assets today when making your consumption saving decision
                                
                            V0 = util_c_KMV(c0, hsizes[i_n], eis, phihs) + W0

                                
                            # upper envelope, update if new is better
                            if V0 > V[i_n,i_z, ia, i_pr]:
                                
                                a[i_n,i_z, ia, i_pr] = a0 
                                c[i_n,i_z, ia, i_pr] = c0
                                V[i_n,i_z, ia, i_pr] = V0
            
    return V, c, a


@njit
def constrain_policies(V,c,a_x,coh,eis,W,a_grid_exp_t1,bchh,y,hsizes,phihs):
    
     for i_n in range(len(V[:,0,0,0])): # Ownership state
         for i_z in range(len(V[0,:,0,0])): # productivity state
             for i_pr in range(len(V[0,0,0,:])): # rental price state
                    
                     W0=np.interp(bchh[i_n,i_z,:,i_pr],a_grid_exp_t1[i_n],  W[i_n,i_z,:,i_pr])
                                    
                     tbc = (a_x[i_n,i_z,:,i_pr] < bchh[i_n,i_z,:,i_pr]) # tbc = to become constrained
                    
                     W[i_n,i_z,:,i_pr][tbc] = W0[tbc] 

                     a_x[i_n,i_z,:,i_pr][tbc] = bchh[i_n,i_z,:,i_pr][tbc]  # applying borrowing constraint to households below constraint
                    
                     c[i_n,i_z,:,i_pr][tbc] = coh[i_n,i_z,:,i_pr][tbc] - a_x[i_n,i_z,:,i_pr][tbc] # compute consumption

                    #default=(c[i_n,i_z,:,i_pr][tbc]<1e-6) # consumption under default
                     default=(c[i_n,i_z,:,i_pr]<=0) # consumption under default
        
                     c[i_n,i_z,:,i_pr][default]=y[3,i_z,i_pr][default] # if consumption is negative, set it to disposable income
                    
                     V[i_n,i_z,:,i_pr][tbc] = util_c_KMV(c[i_n,i_z,:,i_pr][tbc],hsizes[i_n], eis, phihs)  + W[i_n,i_z,:,i_pr][tbc] # compute value

                     V[i_n,i_z,:,i_pr][default] = util_c_KMV(c[i_n,i_z,:,i_pr][default],hsizes[i_n], eis, phihs)  + W[i_n,i_z,:,i_pr][default] # compute value
                    

     return V,c, a_x


# het outputs


def labor_supply(c,eis,z_grid,hours,w,Tax,frisch,hsizes,phihs):
    swn=np.exp(hours)**(1/frisch)
    swd=marg_util_c_KMV(c,hsizes[:,np.newaxis,np.newaxis,np.newaxis]+np.zeros_like(c), eis, phihs) *z_grid[np.newaxis,:,np.newaxis,np.newaxis]*np.exp(w)*(1-Tax)
    sw=swn/swd

    #uhh=util_c_KMV(c,hsizes[:,np.newaxis,np.newaxis,np.newaxis]+np.zeros_like(c), eis, phihs) - 1/(1+1/frisch)*np.exp(hours)**(1+1/frisch)  

    return sw


def housingDemand(c,a,Fsize,pr,phss,agridexpand,coh,pri): # tracking housing demand

    hoo=np.zeros_like(a) # share of owners
    hr = np.zeros_like(a) # share of renters
    hll= np.zeros_like(a) # share of landlords
    hooF=np.zeros_like(a) # share of owners
    hll2= np.zeros_like(a) # share of landlords
    prw= np.zeros_like(a) # share of landlords
    hd=np.ones_like(a) # housing demand

    hoo[0]=1 # own/own
    hoo[5]=1 # landlord/own
    hoo[8]=1 # ownF/own

    hr[1]=1 # own/rent
    hr[3]=1 # rent/rent
    hr[10]=1 # ownF/rent
    hr[14]=1 # rent/rent

    hll[4]=1 # own/landlord
    hll[6]=1 # landlord/landlord
    hll[12]=1 # landlord2/landlord
    hll[15]=1 # landlord/landlord

    hll2[11]=1 # landlord/landlord2
    hll2[13]=1 # landlord2/landlord2
    hll2[16]=1 # landlord2/landlord2

    hooF[7]=1 # own/ownF
    hooF[2]=1 # rent/ownF
    hooF[9]=1 # ownF/ownF

    htrans=np.zeros_like(a) # housing transactions

    htrans[[1,2,4,5,10,11,12]]=1
    htrans[[7,8]]=2

    hsv=np.zeros_like(a) # sales volumes

    hsv[[1,5,7,8,10,12]]=1

    llexit=np.zeros_like(a) # landlord exit rate

    llexit[[5,12]]=1

    ownexit=np.zeros_like(a) # owner exit rate

    ownexit[1]=1    
    ownexit[10]=1    

    prw[3]=np.exp(pri[3])
    prw[1]=np.exp(pri[1])
    prw[14]=np.exp(pri[14])
    prw[10]=np.exp(pri[10])

    hd[[1,3,10,14]]=Fsize
    hd[[2,7,9]]=Fsize

    hlltot=hll+hll2*2

    # consumption by household type [ additional filtering by income furthe controls for compositional effects]

    cmort=c*0
    cmort[[9],6:9,:,:]=c[[9],6:9,:,:]
    cmort[[0],9:12,:,:]=c[[0],9:12,:,:]
    cmort[agridexpand>0]=0
    cmortwgt=1*(cmort>0)

    coo=c*0
    coo[[9],6:9,:,:]=c[[9],6:9,:,:]
    coo[[0],9:12,:,:]=c[[0],9:12,:,:]
    coo[agridexpand<0]=0
    coowgt=1*(coo>0)

    crent=c*0
    crent[[3,14],0:5,:,:]=c[[3,14],0:5,:,:]
    crentwgt=1*(crent>0)

    cll=c*0
    cll[[6,13,15,16],12:,:,:]=c[[6,13,15,16],12:,:,:]
    cllwgt=1*(cll>0)

    ydis_mort=(coh-agridexpand)*cmortwgt
    ydis_oo=(coh-agridexpand)*coowgt
    ydis_rent=(coh-agridexpand)*crentwgt
    ydis_ll=(coh-agridexpand)*cllwgt

    return hoo,hr,hll,hooF,htrans,hsv,llexit,ownexit,hll2,prw,hd,hlltot,crent,coo,cmort,crentwgt,coowgt,cmortwgt,cll,cllwgt,ydis_ll,ydis_mort,ydis_oo,ydis_rent

def assetDemand(a,c,coh,bchh,a_grid_full_t0,rexpand): # tracking asset demand

 
    # default amount 
    addefault=np.zeros_like(a)
    addefault=((coh-bchh)-a-c)*(a+c>(coh-bchh))

    #borrowing

    mortpmt=np.zeros_like(a)
    mortpmt[a_grid_full_t0<0]=-(a_grid_full_t0[a_grid_full_t0<0])*rexpand[a_grid_full_t0<0]
    
    deppmt=np.zeros_like(a)
    deppmt[a_grid_full_t0>0]=(a_grid_full_t0[a_grid_full_t0>0])*rexpand[a_grid_full_t0>0]
    
    return addefault,mortpmt,deppmt
