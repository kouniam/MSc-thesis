'''

DICE 2016 Optimization

'''

import numpy as np
from numba import njit,guvectorize,float64
import scipy.optimize as opt


def DICE_optimizer(year_shift,t2xco2,ipop0,ia0,igsigma1,isig0,icumetree0,miu0,K0,YGROSS0,EIND0,E0,CCA0,CCATOT0,MAT0,ML0,MU0,FOC0,TATM0,TOCEAN0,DAMFRAC0,DAMAGES0,ABATECOST0,MCABATE0,CPRICE0,YNET0,Y0,I0,C0,CPC0,RI0,PERIODU0,CEMUTOTPER0):

    #Set
    ot = np.arange(1, 101)
    oNT = len(ot)
    
    # year_shift = 5  # shift in time step (TEST)
    
    #Parameters
    
    fosslim = 6000 # Maximum cumulative extraction fossil fuels (GtC); denoted by CCum
    tstep  = 5 # Years per Period
    ifopt  = 0 # Indicator where optimized is 1 and base is 0
    
    #Preferences
    
    elasmu = 1.45 #  Elasticity of marginal utility of consumption
    prstp = 0.015 #   Initial rate of social time preference per year 
    
    #** Population and technology
    gama  = 0.300 #   Capital elasticity in production function         /.300 /
    
    popadj = 0.134 #  Growth rate to calibrate to 2050 pop projection  /0.134/
    popasym = 11500 # Asymptotic population (millions)                 /11500/
    dk  = 0.100 #     Depreciation rate on capital (per year)           /.100 /
    q0  = 105.5 #     Initial world gross output 2015 (trill 2010 USD) /105.5/
    k0  = 223 #     Initial capital value 2015 (trill 2010 USD)        /223  /
    
    ga0  = 0.076 #    Initial growth rate for TFP per 5 years          /0.076/
    dela  = 0.005 #   Decline rate of TFP per 5 years                  /0.005/
    
    #** Emissions parameters
    
    dsig  = -0.001 #   Decline rate of decarbonization (per period)    /-0.001 /
    eland0 = 2.6 #  Carbon emissions from land 2015 (GtCO2 per year)   / 2.6   /
    deland = 0.115 # Decline rate of land emissions (per period)        / .115  /
    e0 = 35.85 #    Industrial emissions 2015 (GtCO2 per year)       /35.85  /
    
    #** Carbon cycle
    #* Initial Conditions
    mat0 = 851 #  Initial Concentration in atmosphere 2015 (GtC)       /851  /
    mu0  = 460 #  Initial Concentration in upper strata 2015 (GtC)     /460  /
    ml0  = 1740 #  Initial Concentration in lower strata 2015 (GtC)    /1740 /
    mateq = 588 # mateq Equilibrium concentration atmosphere  (GtC)    /588  /
    mueq  = 360 # mueq Equilibrium concentration in upper strata (GtC) /360  /
    mleq = 1720 # mleq Equilibrium concentration in lower strata (GtC) /1720 /
    
    #* Flow paramaters, denoted by Phi_ij in the model
    b12  = 0.12 #    Carbon cycle transition matrix                     /.12  /
    b23  = 0.007 #   Carbon cycle transition matrix                    /0.007/
    #* These are for declaration and are defined later
    b11  = None   # Carbon cycle transition matrix
    b21  = None  # Carbon cycle transition matrix
    b22  = None  # Carbon cycle transition matrix
    b32  = None  # Carbon cycle transition matrix
    b33  = None  # Carbon cycle transition matrix
    sig0  = None  # Carbon intensity 2010 (kgCO2 per output 2005 USD 2010)
    
    #** Climate model parameters
    # t2xco2  = 3.1 # Equilibrium temp impact (oC per doubling CO2)    / 3.1 / * Climate Sensitivity *
    fex0  = 0.5 #   2015 forcings of non-CO2 GHG (Wm-2)              / 0.5 /
    fex1  = 1.0 #   2100 forcings of non-CO2 GHG (Wm-2)              / 1.0 /
    tocean0  = 0.0068 # Initial lower stratum temp change (C from 1900) /.0068/
    tatm0  = 0.85 #  Initial atmospheric temp change (C from 1900)    /0.85/
    c1  = 0.1005 #     Climate equation coefficient for upper level  /0.1005/
    c3  = 0.088 #     Transfer coefficient upper to lower stratum    /0.088/
    c4  = 0.025 #     Transfer coefficient for lower level           /0.025/
    fco22x  = 3.6813 # eta in the model; Eq.22 : Forcings of equilibrium CO2 doubling (Wm-2)   /3.6813 /
    
    #** Climate damage parameters
    a10  = 0 #     Initial damage intercept                         /0   /
    a20  = None #     Initial damage quadratic term
    a1  = 0 #      Damage intercept                                 /0   /
    a2  = 0.00236 #      Damage quadratic term                     /0.00236/
    a3  = 2.00 #      Damage exponent                              /2.00   /
    
    #** Abatement cost
    expcost2 = 2.6 # Theta2 in the model, Eq. 10 Exponent of control cost function             / 2.6  /
    pback  = 550 #   Cost of backstop 2010$ per tCO2 2015          / 550  /
    gback  = 0.025 #   Initial cost decline backstop cost per period / .025/
    limmiu  = 1.2 #  Upper limit on control rate after 2150        / 1.2 /
    tnopol  = 45 #  Period before which no emissions controls base  / 45   /
    cprice0  = 2 # Initial base carbon price (2010$ per tCO2)      / 2    /
    gcprice  = 0.02 # Growth rate of base carbon price per year     /.02  /
    
    #** Scaling and inessential parameters
    #* Note that these are unnecessary for the calculations
    #* They ensure that MU of first period's consumption =1 and PV cons = PV utilty
    scale1  = 0.0302455265681763 #    Multiplicative scaling coefficient           /0.0302455265681763 /
    scale2  = -10993.704 #    Additive scaling coefficient       /-10993.704/;
    
    #* Parameters for long-run consistency of carbon cycle 
    #(Question)
    b11 = 1 - b12
    b21 = b12*mateq/mueq
    b22 = 1 - b21 - b23
    b32 = b23*mueq/mleq
    b33 = 1 - b32
    
    #* Further definitions of parameters
    a20 = a2
    
    lam = fco22x/ t2xco2 #From Eq. 25
    
    ol = np.zeros(oNT)
    ol[0] = ipop0 #Labor force
    oal = np.zeros(oNT) 
    oal[0] = ia0
    ogsig = np.zeros(oNT) 
    ogsig[0] = igsigma1
    osigma = np.zeros(oNT)
    osigma[0]= isig0
    
    iga = ga0 * np.exp(-dela*5*(ot-1+year_shift)) # shifted
    
    pbacktime = pback * (1-gback)**(ot-1+year_shift) #Backstop price
    
    oetree = eland0*(1-deland)**(ot-1+year_shift) #Emissions from deforestration    TEST
    
    rr = 1/((1+prstp)**(tstep*(ot-1+year_shift))) #Eq. 3
    
    #The following three equations define the exogenous radiative forcing; used in Eq. 23  
    forcoth = np.full(oNT,fex0)
    forcoth[0:18-year_shift] = forcoth[0:18-year_shift] + (1/17)*(fex1-fex0)*(ot[0:18-year_shift]-1)
    forcoth[18-year_shift:oNT] = forcoth[18-year_shift:oNT] + (fex1-fex0)
    
    optlrsav = (dk + .004)/(dk + .004*elasmu + prstp)*gama #Optimal long-run savings rate used for transversality (Question)
    
    ocost1 = np.zeros(oNT)
    ocumetree = np.zeros(oNT)
    
    #ocumetree[0] = 100      # FUNCTION PARAMETER
    ocumetree[0] = icumetree0   #TEST
    
    cpricebase = cprice0*(1+gcprice)**(5*(ot-1+year_shift))
    
    @njit('(float64[:], int32)')
    def oInitializeLabor(iol,ioNT):
        for i in range(1,ioNT):
            iol[i] = iol[i-1]*(popasym / iol[i-1])**popadj
    
    @njit('(float64[:], int32)')        
    def oInitializeTFP(ioal,ioNT):
        for i in range(1,ioNT):
            ioal[i] = ioal[i-1]/(1-iga[i-1])
            
    @njit('(float64[:], int32)')        
    def oInitializeGrowthSigma(iogsig,ioNT):
        for i in range(1,ioNT):
            iogsig[i] = iogsig[i-1]*((1+dsig)**tstep)
            
    @njit('(float64[:], float64[:],float64[:],int32)')        
    def oInitializeSigma(iosigma,iogsig,iocost1,ioNT):
        for i in range(1,ioNT):
            iosigma[i] =  iosigma[i-1] * np.exp(iogsig[i-1] * tstep)
            iocost1[i] = pbacktime[i] * iosigma[i]  / expcost2 /1000
            
    @njit('(float64[:], int32)')        
    def oInitializeCarbonTree(iocumetree,ioNT):
        for i in range(1,ioNT):
            iocumetree[i] = iocumetree[i-1] + oetree[i-1]*(5/3.666)
    
    """
    Functions of the model
    """
    
    """
    First: Functions related to emissions of carbon and weather damages
    """
    
    # Retuns the total carbon emissions; Eq. 18
    @njit('float64(float64[:],int32)') 
    def ofE(ioEIND,index):
        return ioEIND[index] + oetree[index]
    
    #Eq.14: Determines the emission of carbon by industry EIND
    @njit('float64(float64[:],float64[:],float64[:],int32)') 
    def ofEIND(ioYGROSS, ioMIU, iosigma,index):
        return iosigma[index] * ioYGROSS[index] * (1 - ioMIU[index])
    
    #Cumulative industrial emission of carbon
    @njit('float64(float64[:],float64[:],int32)') 
    def ofCCA(ioCCA,ioEIND,index):
        return ioCCA[index-1] + ioEIND[index-1] * 5 / 3.666
    
    #Cumulative total carbon emission
    @njit('float64(float64[:],float64[:],int32)')
    def ofCCATOT(ioCCA,iocumetree,index):
        return ioCCA[index] + iocumetree[index]
    
    #Eq. 22: the dynamics of the radiative forcing
    @njit('float64(float64[:],int32)')
    def ofFORC(ioMAT,index):
        return fco22x * np.log(ioMAT[index]/588.000)/np.log(2) + forcoth[index]
    
    # Dynamics of Omega; Eq.9
    @njit('float64(float64[:],int32)')
    def ofDAMFRAC(ioTATM,index):
        return a1*ioTATM[index] + a2*ioTATM[index]**a3
    
    #Calculate damages as a function of Gross industrial production; Eq.8 
    @njit('float64(float64[:],float64[:],int32)')
    def ofDAMAGES(ioYGROSS,ioDAMFRAC,index):
        return ioYGROSS[index] * ioDAMFRAC[index]
    
    #Dynamics of Lambda; Eq. 10 - cost of the reudction of carbon emission (Abatement cost)
    @njit('float64(float64[:],float64[:],float64[:],int32)') 
    def ofABATECOST(ioYGROSS,ioMIU,iocost1,index):
        return ioYGROSS[index] * iocost1[index] * ioMIU[index]**expcost2
    
    #Marginal Abatement cost
    @njit('float64(float64[:],int32)')
    def ofMCABATE(ioMIU,index):
        return pbacktime[index] * ioMIU[index]**(expcost2-1)
    
    #Price of carbon reduction
    @njit('float64(float64[:],int32)')
    def ofCPRICE(ioMIU,index):
        return pbacktime[index] * (ioMIU[index])**(expcost2-1)
    
    #Eq. 19: Dynamics of the carbon concentration in the atmosphere
    @njit('float64(float64[:],float64[:],float64[:],int32)')
    def ofMAT(ioMAT,ioMU,ioE,index):
        if(index == 0):
            return mat0
        else:
            return ioMAT[index-1]*b11 + ioMU[index-1]*b21 + ioE[index-1] * 5 / 3.666
    
    #Eq. 21: Dynamics of the carbon concentration in the ocean LOW level
    @njit('float64(float64[:],float64[:],int32)') 
    def ofML(ioML,ioMU,index):
        if(index == 0):
            return ml0
        else:
            return ioML[index-1] * b33  + ioMU[index-1] * b23
    
    #Eq. 20: Dynamics of the carbon concentration in the ocean UP level
    @njit('float64(float64[:],float64[:],float64[:],int32)') 
    def ofMU(ioMAT,ioMU,ioML,index):
        if(index == 0):
            return mu0
        else:
            return ioMAT[index-1]*b12 + ioMU[index-1]*b22 + ioML[index-1]*b32
    
    #Eq. 23: Dynamics of the atmospheric temperature
    @njit('float64(float64[:],float64[:],float64[:],int32)') 
    def ofTATM(ioTATM,ioFORC,ioTOCEAN,index):
        if(index == 0):
            return tatm0
        else:
            return ioTATM[index-1] + c1 * (ioFORC[index] - (fco22x/t2xco2) * ioTATM[index-1] - c3 * (ioTATM[index-1] - ioTOCEAN[index-1]))
    
    #Eq. 24: Dynamics of the ocean temperature
    @njit('float64(float64[:],float64[:],int32)')
    def ofTOCEAN(ioTATM,ioTOCEAN,index):
        if(index == 0):
            return tocean0
        else:
            return ioTOCEAN[index-1] + c4 * (ioTATM[index-1] - ioTOCEAN[index-1])
    
    """
    Second: Function related to economic variables
    """
    
    #The total production without climate losses denoted previously by YGROSS
    @njit('float64(float64[:],float64[:],float64[:],int32)')
    def ofYGROSS(ioal,iol,ioK,index):
        return ioal[index] * ((iol[index]/1000)**(1-gama)) * ioK[index]**gama
    
    #The production under the climate damages cost
    @njit('float64(float64[:],float64[:],int32)')
    def ofYNET(ioYGROSS, ioDAMFRAC, index):
        return ioYGROSS[index] * (1 - ioDAMFRAC[index])
    
    #Production after abatement cost
    @njit('float64(float64[:],float64[:],int32)')
    def ofY(ioYNET,ioABATECOST,index):
        return ioYNET[index] - ioABATECOST[index]
    
    #Consumption Eq. 11
    @njit('float64(float64[:],float64[:],int32)')
    def ofC(ioY,ioI,index):
        return ioY[index] - ioI[index]
    
    #Per capita consumption, Eq. 12
    @njit('float64(float64[:],float64[:],int32)')
    def ofCPC(ioC,iol,index):
        return 1000 * ioC[index] / iol[index]
    
    #Saving policy: investment
    @njit('float64(float64[:],float64[:],int32)')
    def ofI(ioS,ioY,index):
        return ioS[index] * ioY[index] 
    
    #Capital dynamics Eq. 13
    @njit('float64(float64[:],float64[:],int32)')
    def ofK(ioK,ioI,index):
        if(index == 0):
            return k0
        else:
            return (1-dk)**tstep * ioK[index-1] + tstep * ioI[index-1]
    
    #Interest rate equation; Eq. 26 added in personal notes
    @njit('float64(float64[:],int32)')
    def ofRI(ioCPC,index):
        return (1 + prstp) * (ioCPC[index+1]/ioCPC[index])**(elasmu/tstep) - 1
    
    #Periodic utility: A form of Eq. 2
    @njit('float64(float64[:],float64[:],int32)')
    def ofCEMUTOTPER(ioPERIODU,iol,index):
        return ioPERIODU[index] * iol[index] * rr[index]
    
    #The term between brackets in Eq. 2
    @njit('float64(float64[:],float64[:],int32)')
    def ofPERIODU(ioC,iol,index):
        return ((ioC[index]*1000/iol[index])**(1-elasmu) - 1) / (1 - elasmu) - 1
    
    #utility function
    @guvectorize([(float64[:], float64[:])], '(n), (m)')
    def ofUTILITY(ioCEMUTOTPER, resUtility):
        resUtility[0] = tstep * scale1 * np.sum(ioCEMUTOTPER) + scale2
    
    """
    In this part we implement the objective function
    """
    
    # * Control rate limits
    oMIU_lo = np.full(oNT,0.01)
    oMIU_up = np.full(oNT,limmiu)
    #oMIU_up[0:29] = 1
    oMIU_up[0:29+year_shift] = 1
    oMIU_lo[0] = miu0
    oMIU_up[0] = miu0
    oMIU_lo[oMIU_lo==oMIU_up] = 0.99999*oMIU_lo[oMIU_lo==oMIU_up]
    bnds1=[]
    for i in range(oNT):
        bnds1.append((oMIU_lo[i],oMIU_up[i]))
    # * Control variables
    lag10 = ot > oNT - 10
    oS_lo = np.full(oNT,1e-1)
    oS_lo[lag10] = optlrsav
    oS_up = np.full(oNT,0.9)
    oS_up[lag10] = optlrsav
    oS_lo[oS_lo==oS_up] = 0.99999*oS_lo[oS_lo==oS_up]
    bnds2=[]
    for i in range(oNT):
        bnds2.append((oS_lo[i],oS_up[i]))
        
    # Arbitrary starting values for the control variables:
    oS_start = np.full(oNT,0.2)
    oS_start[oS_start < oS_lo] = oS_lo[oS_start < oS_lo]
    oS_start[oS_start > oS_up] = oS_lo[oS_start > oS_up]
    oMIU_start = 0.99*oMIU_up
    oMIU_start[oMIU_start < oMIU_lo] = oMIU_lo[oMIU_start < oMIU_lo]
    oMIU_start[oMIU_start > oMIU_up] = oMIU_up[oMIU_start > oMIU_up]
    
    oK = np.zeros(oNT)
    oYGROSS = np.zeros(oNT)
    oEIND = np.zeros(oNT)
    oE = np.zeros(oNT)
    oCCA = np.zeros(oNT)
    oCCATOT = np.zeros(oNT)
    oMAT = np.zeros(oNT)
    oML = np.zeros(oNT)
    oMU = np.zeros(oNT)
    oFORC = np.zeros(oNT)
    oTATM = np.zeros(oNT)
    oTOCEAN = np.zeros(oNT)
    oDAMFRAC = np.zeros(oNT)
    oDAMAGES = np.zeros(oNT)
    oABATECOST = np.zeros(oNT)
    oMCABATE = np.zeros(oNT)
    oCPRICE = np.zeros(oNT)
    oYNET = np.zeros(oNT)
    oY = np.zeros(oNT)
    oI = np.zeros(oNT)
    oC = np.zeros(oNT)
    oCPC = np.zeros(oNT)
    oRI = np.zeros(oNT)
    oPERIODU = np.zeros(oNT)
    oCEMUTOTPER = np.zeros(oNT)
    
    #The objective function
    #It returns the utility as scalar
    def fOBJ(x,sign,ioI,ioK,ioal,iol,ioYGROSS,iosigma,ioEIND,ioE,ioCCA,ioCCATOT,iocumetree,ioMAT,ioMU,ioML,ioFORC,ioTATM,ioTOCEAN,ioDAMFRAC,ioDAMAGES,ioABATECOST,iocost1,ioMCABATE,
             ioCPRICE,ioYNET,ioY,ioC,ioCPC,ioPERIODU,ioCEMUTOTPER,ioRI,ioNT):
        
        ioMIU = x[0:oNT]
        ioS = x[oNT:(2*oNT)]
        
        for i in range(ioNT):
            
            if(i == 0):
            
                ioK[0] = K0
                ioYGROSS[0] = YGROSS0
                ioEIND[0] = EIND0
                ioE[0] = E0
                ioCCA[0] = CCA0
                ioCCATOT[0] = CCATOT0
                ioMAT[0] = MAT0
                ioML[0] = ML0
                ioMU[0] = MU0
                ioFORC[0] = FOC0
                ioTATM[0] = TATM0
                ioTOCEAN[0] = TOCEAN0
                ioDAMFRAC[0] = DAMFRAC0
                ioDAMAGES[0] = DAMAGES0
                ioABATECOST[0] = ABATECOST0
                ioMCABATE[0] = MCABATE0
                ioCPRICE[0] = CPRICE0
                ioYNET[0] = YNET0
                ioY[0] = Y0
                ioI[0] = I0
                ioC[0] = C0
                ioCPC[0] = CPC0
                ioRI[0] = RI0
                ioPERIODU[0] = PERIODU0
                ioCEMUTOTPER[0] = CEMUTOTPER0
                
            else:
                
                ioK[i] = ofK(ioK,ioI,i)
                ioYGROSS[i] = ofYGROSS(ioal,iol,ioK,i)
                ioEIND[i] = ofEIND(ioYGROSS, ioMIU, iosigma,i)
                ioE[i] = ofE(ioEIND,i)
                ioCCA[i] = ofCCA(ioCCA,ioEIND,i)
                ioCCATOT[i] = ofCCATOT(ioCCA,iocumetree,i)
                ioMAT[i] = ofMAT(ioMAT,ioMU,ioE,i)
                ioML[i] = ofML(ioML,ioMU,i)
                ioMU[i] = ofMU(ioMAT,ioMU,ioML,i)
                ioFORC[i] = ofFORC(ioMAT,i)
                ioTATM[i] = ofTATM(ioTATM,ioFORC,ioTOCEAN,i)
                ioTOCEAN[i] = ofTOCEAN(ioTATM,ioTOCEAN,i)
                ioDAMFRAC[i] = ofDAMFRAC(ioTATM,i)
                ioDAMAGES[i] = ofDAMAGES(ioYGROSS,ioDAMFRAC,i)
                ioABATECOST[i] = ofABATECOST(ioYGROSS,ioMIU,iocost1,i)
                ioMCABATE[i] = ofMCABATE(ioMIU,i)
                ioCPRICE[i] = ofCPRICE(ioMIU,i)
                ioYNET[i] = ofYNET(ioYGROSS, ioDAMFRAC, i)
                ioY[i] = ofY(ioYNET,ioABATECOST,i)
                ioI[i] = ofI(ioS,ioY,i)
                ioC[i] = ofC(ioY,ioI,i)
                ioCPC[i] = ofCPC(ioC,iol,i)
                ioPERIODU[i] = ofPERIODU(ioC,iol,i)
                ioCEMUTOTPER[i] = ofCEMUTOTPER(ioPERIODU,iol,i)
                ioRI = ofRI(ioCPC,i)
            
        resUtility = np.zeros(1)
        ofUTILITY(ioCEMUTOTPER, resUtility)
        
        return sign*resUtility[0]
    
    #For the optimal allocation of x, calculates the whole system variables
    def Optimality(x,ioI,ioK,ioal,iol,ioYGROSS,iosigma,ioEIND,ioE,ioCCA,ioCCATOT,iocumetree,ioMAT,ioMU,ioML,ioFORC,ioTATM,ioTOCEAN,ioDAMFRAC,ioDAMAGES,ioABATECOST,iocost1,ioMCABATE,
             ioCPRICE,ioYNET,ioY,ioC,ioCPC,ioPERIODU,ioCEMUTOTPER,ioRI,ioNT):
        
        ioMIU = x[0:oNT]
        ioS = x[oNT:(2*oNT)]
        
        for i in range(ioNT):
            
            if(i == 0):
            
                ioK[0] = K0
                ioYGROSS[0] = YGROSS0
                ioEIND[0] = EIND0
                ioE[0] = E0
                ioCCA[0] = CCA0
                ioCCATOT[0] = CCATOT0
                ioMAT[0] = MAT0
                ioML[0] = ML0
                ioMU[0] = MU0
                ioFORC[0] = FOC0
                ioTATM[0] = TATM0
                ioTOCEAN[0] = TOCEAN0
                ioDAMFRAC[0] = DAMFRAC0
                ioDAMAGES[0] = DAMAGES0
                ioABATECOST[0] = ABATECOST0
                ioMCABATE[0] = MCABATE0
                ioCPRICE[0] = CPRICE0
                ioYNET[0] = YNET0
                ioY[0] = Y0
                ioI[0] = I0
                ioC[0] = C0
                ioCPC[0] = CPC0
                ioRI[0] = RI0
                ioPERIODU[0] = PERIODU0
                ioCEMUTOTPER[0] = CEMUTOTPER0
                
            else:
                
                ioK[i] = ofK(ioK,ioI,i)
                ioYGROSS[i] = ofYGROSS(ioal,iol,ioK,i)
                ioEIND[i] = ofEIND(ioYGROSS, ioMIU, iosigma,i)
                ioE[i] = ofE(ioEIND,i)
                ioCCA[i] = ofCCA(ioCCA,ioEIND,i)
                ioCCATOT[i] = ofCCATOT(ioCCA,iocumetree,i)
                ioMAT[i] = ofMAT(ioMAT,ioMU,ioE,i)
                ioML[i] = ofML(ioML,ioMU,i)
                ioMU[i] = ofMU(ioMAT,ioMU,ioML,i)
                ioFORC[i] = ofFORC(ioMAT,i)
                ioTATM[i] = ofTATM(ioTATM,ioFORC,ioTOCEAN,i)
                ioTOCEAN[i] = ofTOCEAN(ioTATM,ioTOCEAN,i)
                ioDAMFRAC[i] = ofDAMFRAC(ioTATM,i)
                ioDAMAGES[i] = ofDAMAGES(ioYGROSS,ioDAMFRAC,i)
                ioABATECOST[i] = ofABATECOST(ioYGROSS,ioMIU,iocost1,i)
                ioMCABATE[i] = ofMCABATE(ioMIU,i)
                ioCPRICE[i] = ofCPRICE(ioMIU,i)
                ioYNET[i] = ofYNET(ioYGROSS, ioDAMFRAC, i)
                ioY[i] = ofY(ioYNET,ioABATECOST,i)
                ioI[i] = ofI(ioS,ioY,i)
                ioC[i] = ofC(ioY,ioI,i)
                ioCPC[i] = ofCPC(ioC,iol,i)
                ioPERIODU[i] = ofPERIODU(ioC,iol,i)
                ioCEMUTOTPER[i] = ofCEMUTOTPER(ioPERIODU,iol,i)
                ioRI[i] = ofRI(ioCPC,i)
            
        resUtility = np.zeros(1)
        ofUTILITY(ioCEMUTOTPER, resUtility)
        
        return (resUtility[0],ioI,ioK,ioal,iol,ioYGROSS,iosigma,ioEIND,ioE,ioCCA,ioCCATOT,iocumetree,ioMAT,ioMU,ioML,ioFORC,ioTATM,ioTOCEAN,ioDAMFRAC,ioDAMAGES,ioABATECOST,iocost1,ioMCABATE,
             ioCPRICE,ioYNET,ioY,ioC,ioCPC,ioPERIODU,ioCEMUTOTPER,ioRI)

        
    oInitializeLabor(ol,oNT)
    oInitializeTFP(oal,oNT)
    oInitializeGrowthSigma(ogsig,oNT)
    oInitializeSigma(osigma,ogsig,ocost1,oNT)
    oInitializeCarbonTree(ocumetree,oNT)
    
    x_start = np.concatenate([oMIU_start,oS_start])
    bnds = bnds1 + bnds2
    
    result = opt.minimize(fOBJ, x_start, args=(-1.0,oI,oK,oal,ol,oYGROSS,osigma,oEIND,oE,oCCA,oCCATOT,ocumetree,oMAT,oMU,oML,oFORC,oTATM,oTOCEAN,oDAMFRAC,oDAMAGES,oABATECOST,ocost1,oMCABATE,
         oCPRICE,oYNET,oY,oC,oCPC,oPERIODU,oCEMUTOTPER,oRI,oNT), method='SLSQP',bounds = tuple(bnds),options={'disp': True})
    FOptimal = Optimality(result.x,oI,oK,oal,ol,oYGROSS,osigma,oEIND,oE,oCCA,oCCATOT,ocumetree,oMAT,oMU,oML,oFORC,oTATM,oTOCEAN,oDAMFRAC,oDAMAGES,oABATECOST,ocost1,oMCABATE,
         oCPRICE,oYNET,oY,oC,oCPC,oPERIODU,oCEMUTOTPER,oRI,oNT)
    
    peak_TATM = np.max(oTATM)
    optimal_abatement = result.x[1]
    
    #np.save("TATM_cs%d_i%d].npy" % (t2xco2,year_shift), oTATM)
    
    #oPlotFigures100()
    #end = time.time()
    #print(end - start)
    
    return optimal_abatement, peak_TATM
    
    
    





