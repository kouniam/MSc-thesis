import numpy as np
from numba import njit
from numba import guvectorize
from numba import float64

from matplotlib import pyplot as plt
from pack_unpack import unpack_matrix

from DICE_optimizer import DICE_optimizer

from datetime import datetime
startTime = datetime.now()

#** Set **
t = np.arange(1, 22)
NT = len(t)

#** Parameters **

fosslim = 6000   # Maximum cumulative extraction fossil fuels (GtC); CCum
tstep  = 5       # Years per Period
ifopt  = 0       # Indicator where optimized is 1 and base is 0

#** Preferences **

alpha = 1.45    # Elasticity of marginal utility of consumption
rho = 0.015     # Initial rate of social time preference per year 

#** Population and technology **

gamma = 0.300      # Capital elasticity in production function           /.300/
pop0 = 7403        # Initial world population 2015 (millions)            /7403/
popadj = 0.134     # Growth rate to calibrate to 2050 pop projection    /0.134/
popasym = 11500    # Asymptotic population (millions)                   /11500/
delk  = 0.100      # Depreciation rate on capital (per year)             /.100/
q0 = 105.5         # Initial world gross output 2015 (trill 2010 USD)   /105.5/
k0 = 223           # Initial capital value 2015 (trill 2010 USD)          /223/
a0 = 5.115         # Initial level of total factor productivity         /5.115/
ga0 = 0.076        # Initial growth rate for TFP per 5 years            /0.076/
dela = 0.005       # Decline rate of TFP per 5 years                    /0.005/

#** Emissions parameters **

gsigma1 = -0.0152  # Initial growth of sigma (per year)               /-0.0152/
delsig = -0.001    # Decline rate of decarbonization (per period)      /-0.001/
eland0 = 2.6       # Carbon emissions from land 2015 (GtCO2 per year)     /2.6/
deland = 0.115     # Decline rate of land emissions (per period)         /.115/
eind0 = 35.85      # Industrial emissions 2015 (GtCO2 per year)         /35.85/
miu0 = 0.03        # Initial emissions control rate for base case 2015    /.03/

#** Carbon cycle **
#* Initial Conditions *

mat0 = 851      # Initial Concentration in atmosphere 2015 (GtC)          /851/
mup0 = 460      # Initial Concentration in upper strata 2015 (GtC)        /460/
mlo0 = 1740     # Initial Concentration in lower strata 2015 (GtC)       /1740/
mateq = 588     # mateq Equilibrium concentration atmosphere  (GtC)       /588/
mupeq = 360     # mupeq Equilibrium concentration in upper strata (GtC)   /360/
mloeq = 1720    # mloeq Equilibrium concentration in lower strata (GtC)  /1720/

#* Flow parameters, denoted by Phi_ij in the model *
b12 = 0.12   # Carbon cycle transition matrix                             /.12/
b23 = 0.007  # Carbon cycle transition matrix                           /0.007/

#* These are for declaration and are defined later *
b11 = None   # Carbon cycle transition matrix
b21 = None   # Carbon cycle transition matrix
b22 = None   # Carbon cycle transition matrix
b32 = None   # Carbon cycle transition matrix
b33 = None   # Carbon cycle transition matrix
sig0 = None  # Carbon intensity 2010 (kgCO2 per output 2005 USD 2010)

#** Climate model parameters **

t2xco2 = 3.1       # Equilibrium temp impact (oC per doubling CO2)        /3.1/
fex0 = 0.5         # 2015 forcings of non-CO2 GHG (Wm-2)                  /0.5/
fex1 = 1.0         # 2100 forcings of non-CO2 GHG (Wm-2)                  /1.0/
tocean0 = 0.0068   # Initial lower stratum temp change (C from 1900)    /.0068/
tatm0 = 0.85       # Initial atmospheric temp change (C from 1900)       /0.85/
c1 = 0.1005        # Climate equation coefficient for upper level      /0.1005/
c3 = 0.088         # Transfer coefficient upper to lower stratum        /0.088/
c4 = 0.025         # Transfer coefficient for lower level               /0.025/
fco22x = 3.6813    # eta: Forcings of equilibrium CO2 doubling (Wm-2)  /3.6813/

#** Climate damage parameters **

a10 = 0        # Initial damage intercept                                   /0/
a20 = None     # Initial damage quadratic term
a1 = 0         # Damage intercept                                           /0/
a2 = 0.00236   # Damage quadratic term                                /0.00236/
a3 = 2.00      # Damage exponent                                         /2.00/

#** Abatement cost **

expcost2 = 2.6   # Theta2 in the model, Eq. 10 Exponent of control cost function   /2.6/
pback = 550      # Cost of backstop 2010$ per tCO2 2015                   /550/
gback = 0.025    # Initial cost decline backstop cost per period         /.025/
limmiu = 1.2     # Upper limit on control rate after 2150                 /1.2/
tnopol = 45      # Period before which no emissions controls base          /45/
cprice0 = 2      # Initial base carbon price (2010$ per tCO2)               /2/
gcprice = 0.02   # Growth rate of base carbon price per year              /.02/

#** Scaling and inessential parameters
#* Note that these are unnecessary for the calculations
#* They ensure that MU of first period's consumption =1 and PV cons = PV utilty
scale1 = 0.0302455265681763  # Multiplicative scaling coefficient /0.0302455265681763/
scale2 = -10993.704          # Additive scaling coefficient       /-10993.704/;

#* Parameters for long-run consistency of carbon cycle 
#(Question)
b11 = 1 - b12
b21 = b12*mateq/mupeq
b22 = 1 - b21 - b23
b32 = b23*mupeq/mloeq
b33 = 1 - b32

#* Further definitions of parameters
a20 = a2
sig0 = eind0/(q0*(1-miu0)) # derived from industrial emissions equation at t=0

lam = fco22x/ t2xco2 # equilibrium climate sensitivity (?)

#** Initialization **

# Labor/Population
l = np.zeros(NT)
l[0] = pop0

# Total Factor Productivity (TFP)
al = np.zeros(NT) 
al[0] = a0

# Baseline carbon intensity growth
gsig = np.zeros(NT) 
gsig[0] = gsigma1

# Baseline carbon intensity
sigma = np.zeros(NT)
sigma[0]= sig0

# TFP growth rate dynamics
ga = ga0 * np.exp(-dela*5*(t-1))

# Backstop price
pbacktime = pback * (1-gback)**(t-1)

# Emissions from deforestration
etree = eland0*(1-deland)**(t-1)

# Discount factor arising from pure time preference (R)
rr = 1/((1+rho)**(tstep*(t-1)))

# Parametrization of exogenous radiative forcing (Fex);
forcoth = np.full(NT,fex0)
forcoth[0:18] = forcoth[0:18] + (1/17)*(fex1-fex0)*(t[0:18]-1)
forcoth[18:NT] = forcoth[18:NT] + (fex1-fex0)

# Optimal long-run savings rate used for transversality (Question) (?)
optlrsav = (delk + .004)/(delk + .004*alpha + rho)*gamma

cost1 = np.zeros(NT)
cumetree = np.zeros(NT)
cumetree[0] = 100
cpricebase = cprice0*(1+gcprice)**(5*(t-1)) 

@njit('(float64[:], int32)')
def InitializeLabor(il,iNT):
    for i in range(1,iNT):
        il[i] = il[i-1]*(popasym / il[i-1])**popadj

@njit('(float64[:], int32)')        
def InitializeTFP(ial,iNT):
    for i in range(1,iNT):
        ial[i] = ial[i-1]/(1-ga[i-1])
        
@njit('(float64[:], int32)')        
def InitializeGrowthSigma(igsig,iNT):
    for i in range(1,iNT):
        igsig[i] = igsig[i-1]*((1+delsig)**tstep)
        
@njit('(float64[:], float64[:],float64[:],int32)')        
def InitializeSigma(isigma,igsig,icost1,iNT):
    for i in range(1,iNT):
        isigma[i] =  isigma[i-1] * np.exp(igsig[i-1] * tstep)
        icost1[i] = pbacktime[i] * isigma[i]  / expcost2 /1000
        
@njit('(float64[:], int32)')        
def InitializeCarbonTree(icumetree,iNT):
    for i in range(1,iNT):
        icumetree[i] = icumetree[i-1] + etree[i-1]*(5/3.666)
        
        
"""
Functions of the model
"""

"""
First: Functions related to emissions of carbon and weather damages
"""        

# Determines the emission of carbon by industry;
@njit('float64(float64[:],float64[:],float64[:],int32)') 
def fEIND(iYGROSS, iMIU, isigma,index):
    return isigma[index] * iYGROSS[index] * (1 - iMIU[index])

# Retuns the total carbon emissions;
@njit('float64(float64[:],int32)') 
def fE(iEIND,index):
    return iEIND[index] + etree[index]        

# Cumulative industrial emission of carbon;
@njit('float64(float64[:],float64[:],int32)') 
def fCCA(iCCA,iEIND,index):
    return iCCA[index-1] + iEIND[index-1] * 5 / 3.666

# Cumulative total carbon emission;
@njit('float64(float64[:],float64[:],int32)')
def fCCATOT(iCCA,icumetree,index):
    return iCCA[index] + icumetree[index]

# Dynamics of the radiative forcing;
@njit('float64(float64[:],int32)')
def fFORC(iMAT,index):
    return fco22x * np.log(iMAT[index]/588.000)/np.log(2) + forcoth[index]

# Dynamics of Omega (Damage function);
@njit('float64(float64[:],int32)')
def fDAMFRAC(iTATM,index):
    return a1*iTATM[index] + a2*iTATM[index]**a3

# Calculate damages as a function of Gross industrial production;
@njit('float64(float64[:],float64[:],int32)')
def fDAMAGES(iYGROSS,iDAMFRAC,index):
    return iYGROSS[index] * iDAMFRAC[index]
    
# Dynamics of Lambda - cost of the reduction of carbon emission (Abatement cost);
@njit('float64(float64[:],float64[:],float64[:],int32)') 
def fABATECOST(iYGROSS,iMIU,icost1,index):
    return iYGROSS[index] * icost1[index] * iMIU[index]**expcost2 

# Marginal Abatement cost;
@njit('float64(float64[:],int32)')
def fMCABATE(iMIU,index):
    return pbacktime[index] * iMIU[index]**(expcost2-1)

# Price of carbon reduction;
@njit('float64(float64[:],int32)')
def fCPRICE(iMIU,index):
    return pbacktime[index] * (iMIU[index])**(expcost2-1)

# Dynamics of the carbon concentration in the atmosphere;
@njit('float64(float64[:],float64[:],float64[:],int32)') 
def fMAT(iMAT,iMUP,iE,index):
    if(index == 0):
        return mat0
    else:
        return iMAT[index-1]*b11 + iMUP[index-1]*b21 + iE[index-1] * 5 / 3.666

# Dynamics of the carbon concentration in the ocean LO level;
@njit('float64(float64[:],float64[:],int32)') 
def fMLO(iMLO,iMUP,index):
    if(index == 0):
        return mlo0
    else:
        return iMLO[index-1] * b33  + iMUP[index-1] * b23

# Dynamics of the carbon concentration in the ocean UP level;
@njit('float64(float64[:],float64[:],float64[:],int32)') 
def fMUP(iMAT,iMUP,iMLO,index):
    if(index == 0):
        return mup0
    else:
        return iMAT[index-1]*b12 + iMUP[index-1]*b22 + iMLO[index-1]*b32

# Dynamics of the atmospheric temperature;
@njit('float64(float64[:],float64[:],float64[:],int32)') 
def fTATM(iTATM,iFORC,iTOCEAN,index):
    if(index == 0):
        return tatm0
    else:
        return iTATM[index-1] + c1 * (iFORC[index] - lam * iTATM[index-1] 
            - c3 * (iTATM[index-1] - iTOCEAN[index-1]))

# Dynamics of the ocean temperature;
@njit('float64(float64[:],float64[:],int32)')
def fTOCEAN(iTATM,iTOCEAN,index):
    if(index == 0):
        return tocean0
    else:
        return iTOCEAN[index-1] + c4 * (iTATM[index-1] - iTOCEAN[index-1])

"""
Second: Function related to economic variables
"""

# Total production without climate losses (YGROSS);
@njit('float64(float64[:],float64[:],float64[:],int32)')
def fYGROSS(ial,il,iK,index):
    return ial[index] * ((il[index]/1000)**(1-gamma)) * iK[index]**gamma

# Production under the climate damages cost;
@njit('float64(float64[:],float64[:],int32)')
def fYNET(iYGROSS, iDAMFRAC, index):
    return iYGROSS[index] * (1 - iDAMFRAC[index])

# Production after abatement cost;
@njit('float64(float64[:],float64[:],int32)')
def fY(iYNET,iABATECOST,index):
    return iYNET[index] - iABATECOST[index]

# Consumption;
@njit('float64(float64[:],float64[:],int32)')
def fC(iY,iI,index):
    return iY[index] - iI[index]

# Per capita consumption;
@njit('float64(float64[:],float64[:],int32)')
def fCPC(iC,il,index):
    return 1000 * iC[index] / il[index]

# Saving policy: investment
@njit('float64(float64[:],float64[:],int32)')
def fI(iSAV,iY,index):
    return iSAV[index] * iY[index]

# Capital dynamics Eq. 13
@njit('float64(float64[:],float64[:],int32)')
def fK(iK,iI,index):
    if(index == 0):
        return k0
    else:
        return (1-delk)**tstep * iK[index-1] + tstep * iI[index-1]

# Interest rate equation;
@njit('float64(float64[:],int32)')
def fRI(iCPC,index):
    return (1 + rho) * (iCPC[index+1]/iCPC[index])**(alpha/tstep) - 1

# Periodic utility: A form of Eq. 2
@njit('float64(float64[:],float64[:],int32)')
def fCEMUTOTPER(iPERIODU,il,index):
    return iPERIODU[index] * il[index] * rr[index]

# The term between brackets in Eq. 2
@njit('float64(float64[:],float64[:],int32)')
def fPERIODU(iC,il,index):
    return ((iC[index]*1000/il[index])**(1-alpha) - 1) / (1 - alpha) - 1

# Utility function
@guvectorize([(float64[:], float64[:])], '(n), (m)')
def fUTILITY(iCEMUTOTPER, resUtility):
    resUtility[0] = tstep * scale1 * np.sum(iCEMUTOTPER) + scale2


#Ising Opinion Mechanism Function

def Ising(T,Iex,mc_steps):
       
    #breaking the Aij and Sij matrices into local groups
    uS = unpack_matrix(Sij,Lgx,Lgx)
    uA = unpack_matrix(Aij,Lgx,Lgx)
    
    for tt in range(mc_steps):
        
        #local field contribution
        for i in range(L):
            for j in range(L):
                
                hij = 0
                iisum = 0
                losum = 0
    
                neigh1 = list(G.neighbors((i,j)))
                for z in range(len(neigh1)):
                    
                    #index selection
                    tup1 = neigh1[z]
                    z1 = tup1[0]
                    z2 = tup1[1]
                    
                    #contribution of interpersonal connections
                    iisum += Aij[z1,z2]*Sij[z1,z2]
                    
                #local group identification
                ix = np.floor(i/Lg)
                jx = np.floor(j/Lg)
                
                #contribution of local group          
                losum = np.sum(uS[(ix, jx)]*uA[(ix, jx)])
                    
                hij = (1/kij[i,j])*(iisum+(losum/Ng))+Iex
                
                #opinion switch mechanism
                if T == 0:
                    
                    if hij*Sij[i,j] >= 0:
                    
                        pij = 0
                    
                    else:
                    
                        pij = (1-Aij[i,j])
                        Proll = np.random.random()
                    
                        if Proll <= pij:
                        
                            Sij[i,j] = (-1)*Sij[i,j]
                            
                else:
                    
                    if hij*Sij[i,j] > 0:
                    
                        pij = (1-Aij[i,j])*np.exp((-hij*Sij[i,j])/T)
                        Proll = np.random.random()
                    
                        if Proll <= pij:
                        
                            Sij[i,j] = (-1)*Sij[i,j]
                    
                    else:
                        
                        pij = (1-Aij[i,j])*(1-np.exp((hij*Sij[i,j])/T))
                        Proll = np.random.random()
                    
                        if Proll <= pij:
                        
                            Sij[i,j] = (-1)*Sij[i,j]


    return np.sum(Sij)/N         


# Variable list initialization

OPINION = np.zeros(NT)  # opinion results of network
COUPLING = np.zeros(NT)  # coupling equation

K = np.zeros(NT)
YGROSS = np.zeros(NT)
EIND = np.zeros(NT)
E = np.zeros(NT)
CCA = np.zeros(NT)
CCATOT = np.zeros(NT)
MAT = np.zeros(NT)
MLO = np.zeros(NT)
MUP = np.zeros(NT)
FORC = np.zeros(NT)
TATM = np.zeros(NT)
TOCEAN = np.zeros(NT)
DAMFRAC = np.zeros(NT)
DAMAGES = np.zeros(NT)
ABATECOST = np.zeros(NT)
MCABATE = np.zeros(NT)
CPRICE = np.zeros(NT)
YNET = np.zeros(NT)
Y = np.zeros(NT)
I = np.zeros(NT)
C = np.zeros(NT)
CPC = np.zeros(NT)
RI = np.zeros(NT)
PERIODU = np.zeros(NT)
CEMUTOTPER = np.zeros(NT)

InitializeLabor(l,NT)
InitializeTFP(al,NT)
InitializeGrowthSigma(gsig,NT)
InitializeSigma(sigma,gsig,cost1,NT)
InitializeCarbonTree(cumetree,NT)

#Network parameters

T = 0.15        #temperature
Iex = 0.0       #external stimulation
mc_steps = 100  #MC steps

#Model equations with fixed control variables

MIU = np.zeros(NT)      #abatement array
peak_T = np.zeros(NT)      #peak temperature array
SAV = 0.23*np.ones(NT)     #saving rate fixed at 0.23

MIU[0] = miu0
peak_T[0] = 2.5  # initialization peak temperature - Paris Agreement target

#Party-specific optimization target variables
MIU_g = np.zeros(NT)      # abatement array for Green party
MIU_r = np.zeros(NT)      # abatement array for Lukewarmer party

peak_T_g = np.zeros(NT)      # Green party peak temperature projection
peak_T_r = np.zeros(NT)      # Lukewarmer party peak temperature projection

MIU_g[0] = 0
MIU_r[0] = 0

peak_T_g[0] = 0
peak_T_r[0] = 0

for i in range(NT):

        K[i] = fK(K,I,i)
        YGROSS[i] = fYGROSS(al,l,K,i)
        EIND[i] = fEIND(YGROSS, MIU, sigma,i)
        E[i] = fE(EIND,i)
        CCA[i] = fCCA(CCA,EIND,i)
        CCATOT[i] = fCCATOT(CCA,cumetree,i)
        MAT[i] = fMAT(MAT,MUP,E,i)
        MLO[i] = fMLO(MLO,MUP,i)
        MUP[i] = fMUP(MAT,MUP,MLO,i)
        FORC[i] = fFORC(MAT,i)
        TATM[i] = fTATM(TATM,FORC,TOCEAN,i)
        TOCEAN[i] = fTOCEAN(TATM,TOCEAN,i)
        DAMFRAC[i] = fDAMFRAC(TATM,i)
        DAMAGES[i] = fDAMAGES(YGROSS,DAMFRAC,i)
        ABATECOST[i] = fABATECOST(YGROSS,MIU,cost1,i)
        MCABATE[i] = fMCABATE(MIU,i)
        CPRICE[i] = fCPRICE(MIU,i)
        YNET[i] = fYNET(YGROSS,DAMFRAC,i)
        Y[i] = fY(YNET,ABATECOST,i)
        I[i] = fI(SAV,Y,i)
        C[i] = fC(Y,I,i)
        CPC[i] = fCPC(C,l,i)
        PERIODU[i] = fPERIODU(C,l,i)
        CEMUTOTPER[i] = fCEMUTOTPER(PERIODU,l,i)
        RI[i] = fRI(CPC,i)    
        
        #Coupling Equation       
        
        tax_t = (MCABATE[i]*E[i]/l[i])*(1/1000)   # USD per person
        dam_t = DAMAGES[i]/l[i]*(1/1000000)       # USD per person
        
        COUPLING[i] = (1)*dam_t - tax_t + (0.001)*np.sqrt(peak_T[i])
        
        Iex = 100*COUPLING[i] # free, uncalibrated external influence parameter coupling
    
        #Network Opinion Check
        OPINION[i] = Ising(T,Iex,mc_steps)
        
        if i == NT-1:
            break
        
        else:
            
            #Optimization Schemes (for both parties)
        
            #GREENS
            MIU_g[i+1],peak_T_g[i+1] = DICE_optimizer(i,5.0,l[i],al[i],gsig[i],sigma[i],
                cumetree[i],MIU[i],K[i],YGROSS[i],EIND[i],E[i],CCA[i],CCATOT[i],MAT[i],
                MLO[i],MUP[i],FORC[i],TATM[i],TOCEAN[i],DAMFRAC[i],DAMAGES[i],ABATECOST[i],
                MCABATE[i],CPRICE[i],YNET[i],Y[i],I[i],C[i],CPC[i],RI[i],PERIODU[i],CEMUTOTPER[i])
            
            #LUKEWARMERS
            MIU_r[i+1],peak_T_r[i+1] = DICE_optimizer(i,2.0,l[i],al[i],gsig[i],sigma[i],
                cumetree[i],MIU[i],K[i],YGROSS[i],EIND[i],E[i],CCA[i],CCATOT[i],MAT[i],
                MLO[i],MUP[i],FORC[i],TATM[i],TOCEAN[i],DAMFRAC[i],DAMAGES[i],ABATECOST[i],
                MCABATE[i],CPRICE[i],YNET[i],Y[i],I[i],C[i],CPC[i],RI[i],PERIODU[i],CEMUTOTPER[i])
      
            #Peak temperature projection
            
            peak_T[i+1] = 0.5*peak_T_g[i+1]+0.5*peak_T_r[i+1]
            
            
            #Political party selection (parameters)
            if OPINION[i] > 0:
                
                MIU[i+1] = MIU_g[i+1]
                
            else:
                
                MIU[i+1] = MIU_r[i+1]
                

TT = np.linspace(2015, 2115, 21, dtype = np.int32)

def PlotFigures():
    
    pos_OPINION = OPINION.copy()
    neg_OPINION = OPINION.copy()

    pos_OPINION[pos_OPINION <= 0] = np.nan
    neg_OPINION[neg_OPINION > 0] = np.nan
    
    figOPINION = plt.figure(figsize=(8,6))
    plt.plot(TT,OPINION,'k--',alpha=0.5)
    plt.plot(TT,pos_OPINION,'g.')
    plt.plot(TT,neg_OPINION,'r.')
    figOPINION.suptitle('Opinion average', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('$<S>$', fontsize=12)
    
    figTATM = plt.figure(figsize=(8,6))
    plt.plot(TT,TATM,'b.')
    plt.plot(TT,TATM,'b-',alpha=0.5)
    figTATM.suptitle('Increase temperature of the atmosphere (TATM)', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Degrees C from 1900', fontsize=12)
    
    figTOCEAN = plt.figure(figsize=(8,6))
    plt.plot(TT,TOCEAN,'b.')
    plt.plot(TT,TOCEAN,'b-',alpha=0.5)
    figTOCEAN.suptitle('Increase temperature of the ocean (TOCEAN)', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Degrees C from 1900', fontsize=12)
    
    figMU = plt.figure(figsize=(8,6))
    plt.plot(TT,MUP,'b.')
    plt.plot(TT,MUP,'b-',alpha=0.5)
    figMU.suptitle('Carbon concentration increase in shallow oceans (MU)', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('GtC from 1750', fontsize=12)
    
    figML = plt.figure(figsize=(8,6))
    plt.plot(TT,MLO,'b.')
    plt.plot(TT,MLO,'b-',alpha=0.5)
    figML.suptitle('Carbon concentration increase in lower oceans (ML)', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('GtC from 1750', fontsize=12)

    figDAM = plt.figure(figsize=(8,6))
    plt.plot(TT,DAMAGES,'b.')
    plt.plot(TT,DAMAGES,'b-',alpha=0.5)
    figDAM.suptitle('Damages', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('trillions 2010 USD per year', fontsize=12)
    
    figDAMFRAC = plt.figure(figsize=(8,6))
    plt.plot(TT,DAMFRAC,'b.')
    plt.plot(TT,DAMFRAC,'b-',alpha=0.5)
    figDAMFRAC.suptitle('Damages as fraction of gross output', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('', fontsize=12)
    
    figCOSTRED = plt.figure(figsize=(8,6))
    plt.plot(TT,ABATECOST,'b.')
    plt.plot(TT,ABATECOST,'b-',alpha=0.5)
    figCOSTRED.suptitle('Cost of emissions reductions', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('trillions 2010 USD per year', fontsize=12)
    
    figMarg = plt.figure(figsize=(8,6))
    plt.plot(TT,MCABATE,'b.')
    plt.plot(TT,MCABATE,'b-',alpha=0.5)
    figMarg.suptitle('Marginal abatement cost', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('2010 USD per ton CO2', fontsize=12)
    
    figMIU = plt.figure(figsize=(8,6))
    plt.plot(TT,MIU,'b.')
    plt.plot(TT,MIU,'b-',alpha=0.5)
    figMIU.suptitle('Carbon emission control rate', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    
    figE = plt.figure(figsize=(8,6))
    plt.plot(TT,E,'b.')
    plt.plot(TT,E,'b-',alpha=0.5)
    figE.suptitle('Total CO2 emission', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('GtCO2 per year', fontsize=12)
    
    figMAT = plt.figure(figsize=(8,6))
    plt.plot(TT,MAT,'b.')
    plt.plot(TT,MAT,'b-',alpha=0.5)
    figMAT.suptitle('Carbon concentration increase in the atmosphere', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('GtC from 1750', fontsize=12)
    
    figFORC = plt.figure(figsize=(8,6))
    plt.plot(TT,FORC,'b.')
    plt.plot(TT,FORC,'b-',alpha=0.5)
    figFORC.suptitle('Increase in radiative forcing', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('watts per m2 from 1900', fontsize=12)
    
    figC = plt.figure(figsize=(8,6))
    plt.plot(TT,C,'b.')
    plt.plot(TT,C,'b-',alpha=0.5)
    figC.suptitle('Consumption', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('trillions 2010 USD per year', fontsize=12)
    
    figY = plt.figure(figsize=(8,6))
    plt.plot(TT,Y,'b.')
    plt.plot(TT,Y,'b-',alpha=0.5)
    figY.suptitle('Gross product net of abatement and damages', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('trillions 2010 USD per year', fontsize=12)
    
    figYGROSS = plt.figure(figsize=(8,6))
    plt.plot(TT,YGROSS,'b.')
    plt.plot(TT,YGROSS,'b-',alpha=0.5)
    figYGROSS.suptitle('World gross product', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('trillions 2010 USD per year', fontsize=12)
    
    figYGROSSbyY = plt.figure(figsize=(8,6))
    plt.plot(TT,YGROSS-Y,'b.')
    plt.plot(TT,YGROSS-Y,'b-',alpha=0.5)
    figYGROSSbyY.suptitle('Abatement and damages costs', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('trillions 2010 USD per year', fontsize=12)
    
    figSAV = plt.figure(figsize=(8,6))
    plt.plot(TT,SAV,'b.')
    plt.plot(TT,SAV,'b-',alpha=0.5)
    figSAV.suptitle('Saving rate', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('rate', fontsize=12)
    
    figI = plt.figure(figsize=(8,6))
    plt.plot(TT,I,'b.')
    plt.plot(TT,I,'b-',alpha=0.5)
    figI.suptitle('Investment (I)', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('trillions 2010 USD per year', fontsize=12)
    
    figpT = plt.figure(figsize=(8,6))
    plt.plot(TT,peak_T,'b.')
    plt.plot(TT,peak_T,'b-',alpha=0.5)
    figpT.suptitle('Peak Temperature Projection', fontsize=14)
    plt.grid(alpha=0.5)
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Degrees C from 1900', fontsize=12)
    
    plt.show()
    

def DataSave():
    
    np.save("OPINION.npy", OPINION)
    np.save("l.npy", l)
    np.save("al.npy", al)
    np.save("gsig.npy", gsig)
    np.save("TATM.npy", TATM)
    np.save("TOCEAN.npy", TOCEAN)
    np.save("MUP.npy", MUP)
    np.save("MLO.npy", MLO)
    np.save("DAMAGES.npy", DAMAGES)
    np.save("DAMFRAC.npy", DAMFRAC)
    np.save("ABATECOST.npy", ABATECOST)
    np.save("MCABATE.npy", MCABATE)
    np.save("MIU.npy", MIU)
    np.save("E.npy", E)
    np.save("MAT.npy", MAT)
    np.save("FORC.npy", FORC)
    np.save("C.npy", C)
    np.save("Y.npy", Y)
    np.save("YGROSS.npy", YGROSS)
    np.save("SAV.npy", SAV)
    np.save("I.npy", I)
    np.save("COUPLING.npy", COUPLING)
    np.save("MIU_g.npy", MIU_g)
    np.save("MIU_r.npy", MIU_r)
    np.save("peak_T_g.npy", peak_T_g)
    np.save("peak_T_r.npy", peak_T_r)


figLAB = plt.figure(figsize=(8,6))
plt.plot(TT,l,'b.')
plt.plot(TT,l,'b-',alpha=0.5)
figLAB.suptitle('Population growth over time ($L$)', fontsize=14)
plt.grid(alpha=0.5)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Population (million)', fontsize=12)

figTFP = plt.figure(figsize=(8,6))
plt.plot(TT,al,'b.')
plt.plot(TT,al,'b-',alpha=0.5)
figTFP.suptitle('Total factor productivity ($A$)', fontsize=14)
plt.grid(alpha=0.5)
plt.xlabel('Years', fontsize=12)
plt.ylabel('TFP', fontsize=12)

figGSIG = plt.figure(figsize=(8,6))
plt.plot(TT,gsig,'b.')
plt.plot(TT,gsig,'b-',alpha=0.5)
figGSIG.suptitle('Growth of sigma', fontsize=14)
plt.grid(alpha=0.5)
plt.xlabel('Years', fontsize=12)
plt.ylabel('$g_sigma$', fontsize=12)


PlotFigures()
#DataSave()

print(f'Computation time = {datetime.now() - startTime}')
