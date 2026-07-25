
import numpy as np

from sequence_jacobian import simple, solved

###############################################################################################
## Dyanmic blocks
###############################################################################################

@simple
def mkt_clearing(Y,C,HR,HLL,HOO,A,B,Hbar,Fsize,HA_supply,G,HOOF,HA_com,ph,pr,HLL2,hmcost,Fcm,rentindex,PRW,HbarF,epsH,COO,COOWGT,CRENT,CRENTWGT,CMORT,CMORTWGT,CLL,CLLWGT,YDIS_OO,YDIS_RENT,YDIS_LL,YDIS_MORT):
    
    asset_mkt = A - B
 
    housing_mkt=Fsize*epsH+Hbar-Fsize*(HR+HOOF)-HOO-HLL-HLL2 # housing supply equals housing demand
    flat_mkt=0
    goods_mkt=C+G+(Hbar+epsH*Fsize)*hmcost+Fcm*HA_com-Y # goods market clearing

    rental_mkt= HA_com+(HA_supply+epsH) + HLL+2*HLL2 - HR # rental supply equals rental demand

    rentindex_res=rentindex-PRW/HR


    # consumption by teunure

    COObar=COO/COOWGT
    CRENTbar=CRENT/CRENTWGT
    CMORTbar=CMORT/CMORTWGT
    CLLbar=CLL/CLLWGT

    # dis inc by tenure

    YDIS_OObar=YDIS_OO/COOWGT
    YDIS_RENTbar=YDIS_RENT/CRENTWGT
    YDIS_LLbar=YDIS_LL/CLLWGT
    YDIS_MORTbar=YDIS_MORT/CMORTWGT
    

    return asset_mkt,rental_mkt,housing_mkt,goods_mkt,rentindex_res,flat_mkt,COObar,CRENTbar,CMORTbar,CLLbar,YDIS_OObar,YDIS_RENTbar,YDIS_LLbar,YDIS_MORTbar


@simple
def firm(Y, Z,w,alphah,NEss,prcom_prof):
    hours = ((Y/Z)**(1/alphah)/NEss).apply(np.log)
    Div = (Y - w.apply(np.exp) * hours.apply(np.exp)*NEss+prcom_prof).apply(np.log)
    SP=w.apply(np.exp)/(alphah*Y/(hours.apply(np.exp)*NEss))
    return hours, Div,SP


@solved(unknowns={'pi': 0 }, targets=['pi_res'], solver="brentq")
def nkpc(pi,beta,SP,SPss,kappa,r):
    pi_res=(1)/(1+r(+1))*pi(+1)+kappa*(SP/SPss).apply(np.log)-pi
    return pi_res

@simple # some version of model need to guess pi
def nkpc_simple(pi,beta,SP,SPss,kappa,r):
    pi_res=(1)/(1+r(+1))*pi(+1)+kappa*(SP/SPss).apply(np.log)-pi
    return pi_res

@solved(unknowns={'i': 0.01 }, targets=['i_res'], solver="brentq")
def monetary(i,rstar, phi,pi_cpi,rhom,epsr,Y,Yss,phiy):
    ygap=(Y-Yss)/Yss
    i_res=(rstar+pi_cpi*phi+phiy*ygap)*(1-rhom)+(1+i(-1))*rhom+epsr-i
    return i_res,ygap

@simple
def fisher(r,pi,i,piw,w):
    fisher_res = 1+r-(1+i(-1))/(1+pi)
    realwage_res = piw-pi-(w-w(-1))
    return fisher_res,realwage_res

@simple
def rentindexB(rentindex):
    prlag=rentindex(-1).apply(np.log)
    return prlag

@simple 
def block_RELL(pr,r,ph):
    prLL=1*pr
    phLL=1*ph
    rLL=1*r
    return prLL,phLL,rLL


@solved(unknowns={'B': 2.7}, targets=['B_res'], solver="brentq")
def fiscal(Tax,B,r,G,gammatax,Taxss,Bss,Yss,w,hours,NEss,rentindex,HA_supply,transac,HTRANS,MORTPMT,DEPPMT,TransfAgg,hmcost,Fsize,ADDEFAULT,epsH):
    B_res=B(-1)+G+TransfAgg-Tax*w.apply(np.exp)*NEss*hours.apply(np.exp)-MORTPMT+DEPPMT+(HA_supply+epsH)*Fsize*hmcost+ADDEFAULT-rentindex*(HA_supply+epsH)-HTRANS*transac-B
    Tax_res = Taxss+gammatax*(B(-1)-Bss)/Yss - Tax
    return B_res,Tax_res

@solved(unknowns={'piw': 0 }, targets=['piw_res'], solver="brentq")
def wagepc(piw,beta,SW,SWss,kappaw):
    piw_res=beta*piw(+1)+kappaw*(SW/SWss).apply(np.log)-piw,
    return piw_res

@simple
def cpiindex(pi,rentindex,rentweight):

    pi_cpi=(1-rentweight)*pi+rentweight*(rentindex/rentindex(-1)-1+pi)

    return pi_cpi

##############################################################################################################
## commercial renter version of model
##############################################################################################################

@solved(unknowns={'vr1': 3.94,'vr2':3.94,'vr3': 15.53 }, targets=['vr1_res','vr2_res','vr3_res'], solver="broyden")
def rentstar(ph,r,Fsize,hmcost,pr_calvo,vr1,vr2,vr3,pi):

    vr1_res=1+((1-pr_calvo)/((1+r(+1))*(1+pi(+1))))*vr1(+1)-vr1
    vr2_res=1+((1-pr_calvo)/((1+r(+1))))*vr2(+1)-vr2
    vr3_res=(pr_calvo/(1+r(+1)))*ph(+1).apply(np.exp)*Fsize+((1-pr_calvo)/(1+r(+1)))*vr3(+1)-vr3

    return vr1_res,vr2_res,vr3_res

@simple
def rentsector(ph,r,HA_com,Fsize,hmcost,vr1,vr2,vr3,rentindex,Fcm,pr_calvo):

    pr=((Fsize*ph.apply(np.exp)+(Fcm+Fsize*hmcost)*vr2-vr3)/vr1).apply(np.log)

    new_HA = HA_com - HA_com(-1) * (1-pr_calvo)

    prcom_prof = HA_com * (rentindex - Fcm - hmcost * Fsize) + (HA_com(-1) * pr_calvo - new_HA)* ph.apply(np.exp) * Fsize  # realised profits
    # I'm not sure the above is exactly correct because I think the rentindex of the commercial sector specifically can be different from that of the private landlords

    return pr,prcom_prof

################################################################################################
## Housing Investment version of model
################################################################################################

@solved(unknowns={'Hbar': 0.859 }, targets=['Hbar_res'], solver="brentq")
def houseinvest_IH(Lbar,alpha_ih,deltahstar,ph,Hbar):

    Ih=(alpha_ih*ph.apply(np.exp))**((alpha_ih)/(1-alpha_ih))*Lbar # investment in housing

    Hbar_res=(Hbar-Hbar(-1))+deltahstar*Hbar-Ih # law of motion for housing, which is also the "market clearing" condition for housing investment

    return Hbar_res,Ih


@simple
def mkt_clearing_IH(Hbarss,Y,C,HR,HLL,HOO,A,B,Hbar,Fsize,HA_supply,G,HOOF,HA_com,ph,pr,HLL2,hmcost,Fcm,rentindex,PRW,HbarF,epsH,COO,COOWGT,CRENT,CRENTWGT,CMORT,CMORTWGT,CLL,CLLWGT,YDIS_OO,YDIS_RENT,YDIS_LL,YDIS_MORT,Ih,Ihss,phss,alpha_ih):
    
    asset_mkt = A - B
    
    housing_mkt=Fsize*epsH+Hbar-Fsize*(HR+HOOF)-HOO-HLL-HLL2 # housing supply equals housing demand
    flat_mkt=0

    goods_mkt=C+G+(Ih)*ph.apply(np.exp)*alpha_ih+(1-alpha_ih)*phss*Ihss+Fcm*HA_com-Y # goods market clearing, including now the final goods used for housing investment and the fixed cost for the housing investment firms


    rental_mkt= HA_com+(HA_supply+epsH) + HLL+2*HLL2 - HR # rental supply equals rental demand

    rentindex_res=rentindex-PRW/HR


    # consumption by teunure

    COObar=COO/COOWGT
    CRENTbar=CRENT/CRENTWGT
    CMORTbar=CMORT/CMORTWGT
    CLLbar=CLL/CLLWGT

    # dis inc by tenure

    YDIS_OObar=YDIS_OO/COOWGT
    YDIS_RENTbar=YDIS_RENT/CRENTWGT
    YDIS_LLbar=YDIS_LL/CLLWGT
    YDIS_MORTbar=YDIS_MORT/CMORTWGT
    

    return asset_mkt,rental_mkt,housing_mkt,goods_mkt,rentindex_res,flat_mkt,COObar,CRENTbar,CMORTbar,CLLbar,YDIS_OObar,YDIS_RENTbar,YDIS_LLbar,YDIS_MORTbar



@solved(unknowns={'B': 2.7}, targets=['B_res'], solver="brentq")
def fiscal_IH(Tax,B,r,gammatax,Taxss,Bss,Yss,Gss,w,hours,NEss,rentindex,HA_supply,transac,HTRANS,MORTPMT,DEPPMT,TransfAgg,hmcost,Fsize,ADDEFAULT,epsH,alpha_ih,Ih,Ihss,ph,phss,Hbar,deltahstar):
    G=Gss 
    B_res=B(-1)+G+TransfAgg-Tax*w.apply(np.exp)*NEss*hours.apply(np.exp)-MORTPMT+DEPPMT+(HA_supply+epsH)*Fsize*hmcost+ADDEFAULT-rentindex*(HA_supply+epsH)-HTRANS*transac-(ph.apply(np.exp)*Ih-phss*Ihss)*(1-alpha_ih)+(ph.apply(np.exp)-phss)*Hbar*deltahstar  -B # including in the government budget the housing subsidy for changes in the cost of depreciation due to changes in house prices and also the profits of the housing investment firm, which is equal to the land share minus the fixed cost part
    Tax_res = Taxss+gammatax*(B(-1)-Bss)/Yss - Tax
    return B_res,Tax_res,G


###################################################################################################
## Market segmetnation version of model
###################################################################################################

@simple
def mkt_clearing_seg(Y,C,HR,HLL,HOO,A,B,Hbar,Fsize,HA_supply,G,HOOF,HA_com,ph,pr,HLL2,hmcost,Fcm,rentindex,PRW,HbarF,epsH,COO,COOWGT,CRENT,CRENTWGT,CMORT,CMORTWGT,CLL,CLLWGT,YDIS_OO,YDIS_RENT,YDIS_LL,YDIS_MORT):
    
    asset_mkt = A - B

    housing_mkt=Fsize*epsH+Hbar-Fsize*HOOF-HOO-HLL-HLL2 # housing supply equals housing demand

    flat_mkt=HbarF-Fsize*(HA_com+(HA_supply+epsH) + HLL+2*HLL2) # flat market clearing

    goods_mkt=C+G+(Hbar+epsH*Fsize)*hmcost+Fcm*HA_com-Y # goods market clearing

    rental_mkt= HA_com+(HA_supply+epsH) + HLL+2*HLL2 - HR # rental supply equals rental demand

    rentindex_res=rentindex-PRW/HR


    # consumption by teunure

    COObar=COO/COOWGT
    CRENTbar=CRENT/CRENTWGT
    CMORTbar=CMORT/CMORTWGT
    CLLbar=CLL/CLLWGT

    # dis inc by tenure

    YDIS_OObar=YDIS_OO/COOWGT
    YDIS_RENTbar=YDIS_RENT/CRENTWGT
    YDIS_LLbar=YDIS_LL/CLLWGT
    YDIS_MORTbar=YDIS_MORT/CMORTWGT
    

    return asset_mkt,rental_mkt,housing_mkt,goods_mkt,rentindex_res,flat_mkt,COObar,CRENTbar,CMORTbar,CLLbar,YDIS_OObar,YDIS_RENTbar,YDIS_LLbar,YDIS_MORTbar

###########################################################################################
## Sticky mortgage rates
###########################################################################################

@solved(unknowns={'imort':0.005 }, targets=['imort_res'], solver="broyden")
def mortrate(i,pi,imort,FixP):
    thetamort=1/FixP

    imortstar=0.
    for ii in range(FixP):
        imortstar=imortstar+i(+ii)/FixP

    imort_res=thetamort*imortstar+(1-thetamort)*imort(-1)-imort

    rmort=(1+imort(-1))/(1+pi)-1

    return rmort,imortstar,imort_res