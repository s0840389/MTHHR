
import numpy as np

from sequence_jacobian import simple, solved

##############################################################
## Steady state blocks
##############################################################



# production 

#z_grid, z_dist, z_markov = grids.markov_rouwenhorst(cali['rho_z'],cali['sd_z'],cali['n_z']) # generate labor productivity grid, distribution, and transition matrix

#cali['NEss']=np.sum(z_dist*z_grid)

#cali['Y']=1.0
#cali['Z']=1.0

@simple
def firmSS(Y, Z,alphah,NEss,mu):
    hours = np.log((Y/Z)**(1/alphah)/NEss)
    w=np.log(1/mu*alphah*Y)/(np.exp(hours)*NEss)
    prcom_prof=0
    Div = np.log(Y - np.exp(w) * np.exp(hours)*NEss+prcom_prof)
    SP=np.exp(w)/(alphah*Y/(np.exp(hours)*NEss))

    return w, Div,hours,SP,prcom_prof


@simple
def housingSS(r,w,hours,NEss,borwedge,Fsize,deltah,pr,phy):
    
    ph=np.log(phy*np.exp(w)*np.exp(hours)*NEss)
    phss=np.exp(ph)
    transac=0.02*phss # transaction cost
    min_a=-np.exp(ph)*1.0
    max_a=8*np.exp(ph)
    #pr=(Fsize*ph*(r+0.00*borwedge))*1.5 # rental price
    prss=pr*1
    phF=np.log(np.exp(ph)*Fsize)

    HA_com=0

    hmcost=deltah*np.exp(pr)/Fsize # maintenance cost
    Fcm=np.exp(pr)-phss*Fsize*r/(1+r)-hmcost*Fsize #rental company fixed cost

    return ph,phss,transac,min_a,max_a, HA_com,hmcost,Fcm,phF,prss


@simple 
def TaxTransfss(pr,transac,ph,r,TransfAgg,Gguess):

    Transf=0
    
    Tax=Gguess+TransfAgg-0.2*np.exp(pr)-0.02/4*transac*np.exp(ph)+r*0.75*4+0.2*2/3*0.02

    return Transf,Tax

# taxes & transfers


@simple 
def market_clearingSS(Y,C,NEss,hours,A,HR,HLL,HOO,HOOF,Fsize,Tax,w,DEPPMT,MORTPMT,pr,borwedge,transac,HTRANS,r,TransfAgg,HLL2,hmcost,ADDEFAULT):

    # market clearing

    L=NEss*np.exp(hours) # labour services
    
    B=A*1 # steady state debt
    
    HA_supply=HR-HLL-2*HLL2# social housing suppply

    Hbar=HLL+HLL2+HOO+Fsize*(HR+HOOF) # total housing supply
    HbarF=0

    G= Tax*np.exp(w)*L+np.exp(pr)*HA_supply+MORTPMT-DEPPMT+transac*HTRANS -TransfAgg -HA_supply*hmcost*Fsize-ADDEFAULT# Government purchases

    asset_mkt=0
    housing_mkt=0
    rental_mkt=0
    goods_mkt=Y-C-G-Hbar*hmcost
    flat_mkt=0

    epsH=0

    rentweight= np.exp(pr)*HR/(C+HR*np.exp(pr)) # weight of rent in cpi

    return L,B,HA_supply,G,Hbar,asset_mkt,housing_mkt,rental_mkt,goods_mkt,HbarF,flat_mkt,epsH,rentweight



@simple
def pricesSS(r,pr): # other steady state values relevant for dynamic model

    i_res=0
    pi_res=0
    piw_res=0
    B_res=0
    Tax_res=0
    fisher_res=0
    realwage_res=0

    pi=0
    piw=0
    i=r*1
    rstar=r*1
    rentindex=np.exp(pr)*1
    prlag=pr*1

    pi_cpi=0
    
    rmort=r*1

    return i_res,pi_res,piw_res,B_res,Tax_res,fisher_res,realwage_res,pi,piw,i,rstar,rentindex,prlag,pi_cpi,rmort

@simple 
def SSparams(SW,SP,w,hours,Tax,B,Y,frisch,kappaw,kappa,phi,gammatax,rhom,pr_calvo,phiy,pr,deltaWK,deltaWKph,gammaWK,gammaWKph): #include dynamic parameters as inputs so they go into ss0

    ## parameters based on steady state values
    SWss=SW*1
    SPss=SP*1
    wss=w*1
    hourss=hours*1
    Taxss=Tax*1
    Bss=B*1
    Yss=Y*1
       
    # shocks
    epsr=0
    
    return SWss,SPss,wss,hourss,Taxss,Bss,Yss,epsr


@simple 
def SSsolve(B,Btarget,HR,HRtarget,HLL,HLLtarget,LLEXIT,OWNEXIT,HOO,LLEXITtarget,OWNEXITtarget,HLL2,HOOF,HOOFtarget): # moment targeting

    Bsolve=B-Btarget # debt target
    HRsolve=HR-HRtarget # renters target,
    HLLsolve=HLL+HLL2-HLLtarget # landlord target
    LLEXITsolve=4*LLEXIT/(HLL+HLL2)-LLEXITtarget # landlord exit target
    OWNEXITsolve=4*OWNEXIT/HOO-OWNEXITtarget # onwer exit target
    HOOFsolve=HOOF-HOOFtarget

    uloss=(Bsolve**2)+10*(HRsolve**2)+10*(HLLsolve**2)+10*(LLEXITsolve**2)+10*(OWNEXITsolve**2)+10*(HOOFsolve**2)
    return Bsolve,HRsolve,HLLsolve,LLEXITsolve,OWNEXITsolve,uloss





