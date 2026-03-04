"""A phenomenological-based dynamic model of ethanol steam 
reforming for hydrogen production"""
# https://doi.org/10.1016/j.ijhydene.2026.154199
""""Mateo Arcila-Osorio"""


"""REMARK: Due to a typographical error, the variables here appear incorrectly 
labeled with respect to the process system (PS) to which they actually belong. 
Specifically, variables denoted with II actually correspond to I, those labeled 
III correspond to II, and so on. This is purely a nomenclature issue and does 
not affect the model results."""


import numpy as np
import math as m
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.close('all')
from scipy.io import loadmat
from timeit import default_timer as timer

# Order of reactions: [dehydrogenation, decomposition, WGSR, reforming]
global n
# Number of partitions (each stage)
n = 50
# Number of variables
nv = 20

# Order of variables
"""
T_S_II = 0         Solid temperature in PS II
N_G_II = 1         Moles of gas in PS II
y_E_II = 2         Mole fraction of ethanol in PS II
y_A_II = 3         Mole fraction of acetaldehyde in PS II
y_H_II = 4         Mole fraction of hydrogen in PS II
y_W_II = 5         Mole fraction of water in PS II
y_CD_II = 6        Mole fraction of carbon dioxide in PS II
y_CM_II = 7        Mole fraction of Carbon monoxide in PS II
y_M_II = 8         Mole fraction of Methane in PS II
T_G_II = 9         Gas temperature in PS II

T_S_III = 10         Solid temperature in PS III
N_G_III = 11         Moles of gas in PS III
y_E_III = 12         Mole fraction of ethanol in PS III
y_A_III = 13         Mole fraction of acetaldehyde in PS III
y_H_III = 14         Mole fraction of hydrogen in PS III
y_W_III = 15         Mole fraction of water in PS III
y_CD_III = 16        Mole fraction of carbon dioxide in PS III
y_CM_III = 17        Mole fraction of Carbon monoxide in PS III
y_M_III = 18         Mole fraction of Methane in PS III
T_G_III = 19         Gas temperature in PS III

"""
# Initial conditions: all zeros
X0 = np.ones(nv*n)

data = loadmat('initial_conditions50.mat')
T_S_II_i = data['T_S_II_in'].squeeze()  
N_G_II_i = data['N_G_II_in'].squeeze()
y_E_II_i = data['y_E_II_in'].squeeze()
y_A_II_i = data['y_A_II_in'].squeeze()
y_H_II_i = data['y_H_II_in'].squeeze()
y_W_II_i = data['y_W_II_in'].squeeze()
y_CD_II_i = data['y_CD_II_in'].squeeze()
y_CM_II_i = data['y_CM_II_in'].squeeze()
y_M_II_i = data['y_M_II_in'].squeeze()
T_G_II_i = data['T_G_II_in'].squeeze()

T_S_III_i = data['T_S_III_in'].squeeze() 
N_G_III_i = data['N_G_III_in'].squeeze()*0.1
y_E_III_i = data['y_E_III_in'].squeeze()
y_A_III_i = data['y_A_III_in'].squeeze()
y_H_III_i = data['y_H_III_in'].squeeze()
y_W_III_i = data['y_W_III_in'].squeeze()
y_CD_III_i = data['y_CD_III_in'].squeeze()
y_CM_III_i = data['y_CM_III_in'].squeeze()
y_M_III_i = data['y_M_III_in'].squeeze()
T_G_III_i = data['T_G_III_in'].squeeze()

# Initial conditions adjustments 
T_S_II_i = np.insert(T_S_II_i[:-1], 0, 773.15 + 0)
N_G_II_i = np.insert(N_G_II_i[:-1], 0, 1e-7) 
y_E_II_i = np.insert(y_E_II_i[:-1], 0, 0.1428) 
y_A_II_i = np.insert(y_A_II_i[:-1], 0, 0)
y_H_II_i = np.insert(y_H_II_i[:-1], 0, 0)  
y_W_II_i = np.insert(y_W_II_i[:-1], 0, 0.8572) 
y_CD_II_i = np.insert(y_CD_II_i[:-1], 0, 0) 
y_CM_II_i = np.insert(y_CM_II_i[:-1], 0, 0)
y_M_II_i = np.insert(y_M_II_i[:-1], 0, 0)  
T_G_II_i = np.insert(T_G_II_i[:-1], 0, 773.15) 

X0[n*0:n*1] = T_S_II_i
X0[n*1:n*2] = N_G_II_i 
X0[n*2:n*3] = y_E_II_i
X0[n*3:n*4] = y_A_II_i
X0[n*4:n*5] = y_H_II_i
X0[n*5:n*6] = y_W_II_i
X0[n*6:n*7] = y_CD_II_i
X0[n*7:n*8] = y_CM_II_i
X0[n*8:n*9] = y_M_II_i
X0[n*9:n*10] = T_G_II_i

X0[n*10:n*11] = T_S_III_i
X0[n*11:n*12] = N_G_III_i
X0[n*12:n*13] = y_E_III_i
X0[n*13:n*14] = y_A_III_i
X0[n*14:n*15] = y_H_III_i
X0[n*15:n*16] = y_W_III_i
X0[n*16:n*17] = y_CD_III_i
X0[n*17:n*18] = y_CM_III_i
X0[n*18:n*19] = y_M_III_i
X0[n*19:n*20] = T_G_III_i

pure_H2 = []
t_prev = -np.inf      # track last recorded time
step_applied = False  # global

def ODEs(time, X):
    
    global ndot_IV, ndot_III, ndot_II, t_prev, step_applied
    
    ndot_II  = []
    ndot_III = []
    ndot_IV  = []
    ndot_m   = []
            
    # Constants and constants parameters 
    
    # Universal gas constant
    R = 8.3145 # J/(mol-K)
    # Channel diameter
    d = 1.2e-3 # m
    # Channel cross-sectional area
    A = m.pi*(d/2)**2 # m2
    # Total device length
    L = 0.23 # m
    # Length of reforming stage
    L1 = 0.154 # m
    # Length of separation stage
    L2 = L - L1  # m
    # Total volume of reforming stage
    V1 = A*L1 # m3
    # Volume of partition in reforming stage
    V_p1 = (V1/n) # m3
    # Membrane diameter
    Dm = (1/8)*0.0254 # m
    # Reference temperature for kinetic parameters
    Tref1 = 773.15 # K
    # Reference temperature for enthalpy of reactions
    Tref2 = 298.15 # K
    # Heat transfer area in one partition in reforming stage
    A_HT = (2*m.pi*(d/2)*L1)/(n) # m2
    # Heat capacity of cordierite
    C_S = 0.4133 # J/(g-K)
    # Mass of solid in one partition
    M_S = 1.05 # g
    # Heat flow from Heater to Solid
    Q_HS = 0.36 # W
    # Pre-exponential factor of kinetic parameters
    kinf = [2.1e4, 2.0e3, 1.9e4, 2.0e5] #
    # Activation energy of reactions
    Ea = [7.0e4, 1.30e5, 7.0e4, 9.8e4] # J/mol
    # Standard enthalpy of reactions
    deltaH_std = [64600, 49875, -41166, 109136] # J/mol
    # "Activation energy" of membrane
    Eam = 8.8e3 # J/mol
    # Membrane thickness 
    delta = 3.0e-5 # m
    # Membrane surface area
    alpha = m.pi*Dm*(L2/n) # m2
    # Permeability pre-exponential factor
    Pe0 = (2.25e-8)*60 # mol/(min-Pa^1/2-m)
    # Inlet ethanol molar flow 
    ndot_E_in = 0.0021/236 # mol/min
    # Inlet water molar flow
    ndot_W_in = 0.0099/236 # mol/min 
    # Total inlet molar flow
    ndot_in = ndot_E_in + ndot_W_in # mol/min
       

    # Inlet conditions variables   
    y_E_in = ndot_E_in/ndot_in        
    y_A_in = 0        
    y_H_in = 0        
    y_W_in = ndot_W_in/ndot_in          
    y_CD_in = 0       
    y_CM_in = 0       
    y_M_in = 0 
    P_in = 4 * 100000 # Pa 
    T_G_in = 773.15 + 0 # K 
        
    dXdt = np.zeros_like(X)
    
    for z in range(2*n):
        
        # Correction factors (function of P_in)
        c1 = -1e-24*P_in**4 + 3e-18*P_in**3 - 2e-12*P_in**2 + 4e-7*P_in + 0.1095
        c2 = 6e-25*P_in**4 - 2e-18*P_in**3 + 3e-12*P_in**2 - 1e-6*P_in + 1.8041
        c3 = -2e-10*P_in + 1.0
        
        if z < n:
            
            idx = z
            T_S_II,N_G_II,y_E_II,y_A_II,y_H_II,y_W_II,y_CD_II,y_CM_II,y_M_II,T_G_II = X[idx:10*n:n]
            
            # Upstream values of variables
            y_E_II_up = y_E_in        if z == 0 else X[2*n+idx-1]
            y_A_II_up = y_A_in        if z == 0 else X[3*n+idx-1]
            y_H_II_up = y_H_in        if z == 0 else X[4*n+idx-1]
            y_W_II_up = y_W_in        if z == 0 else X[5*n+idx-1]
            y_CD_II_up = y_CD_in      if z == 0 else X[6*n+idx-1]
            y_CM_II_up = y_CM_in      if z == 0 else X[7*n+idx-1]
            y_M_II_up = y_M_in        if z == 0 else X[8*n+idx-1]
            T_G_II_up = T_G_in        if z == 0 else X[9*n+idx-1]
            
            
            # Heat transfer coefficient
            U = c1*(25)*60 # J/(m2-min-K)
            
            # Equations
            Q_SG_II = 0 if z == 0 else U*A_HT*(T_S_II - T_G_II)
            
            dT_S_II = 0*(1/(M_S*C_S))*(Q_HS - Q_SG_II)
            
            P_II = P_in #if z == 0 else (N_G_II*T_G_II*R)/(V_p1)
            
            P_E = (y_E_II*P_II)/(1e5)   # bar
            P_A = (y_A_II*P_II)/(1e5)   # bar
            P_H = (y_H_II*P_II)/(1e5)   # bar
            P_W = (y_W_II*P_II)/(1e5)   # bar
            P_CD = (y_CD_II*P_II)/(1e5) # bar
            P_CM = (y_CM_II*P_II)/(1e5) # bar
            
            k_1 = kinf[0]*m.exp(-Ea[0]*((1/(R*T_G_II))-(1/(R*Tref1))))
            k_2 = kinf[1]*m.exp(-Ea[1]*((1/(R*T_G_II))-(1/(R*Tref1))))
            k_3 = kinf[2]*m.exp(-Ea[2]*((1/(R*T_G_II))-(1/(R*Tref1))))
            k_4 = kinf[3]*m.exp(-Ea[3]*((1/(R*T_G_II))-(1/(R*Tref1))))
            k_WGS = m.exp((4577.8/T_G_II) - 4.33)
            
            # Volume of reaction
            V_rx = c2*V_p1 # m3
            
            r1 = 0 if z == 0 else ((k_1*P_E)/(7.96 + 5.82*((P_II/1e5)-1)))*V_rx
           
            r2 = 0 if z == 0 else (k_2*P_E)*V_rx
            
            r3 = 0 if z == 0 else (k_3*(P_CM*P_W - (P_CD*P_H/k_WGS)))*V_rx
            
            r4 = 0 if z == 0 else (k_4*P_A*P_W**3)*V_rx
        
            if z == 0:
                ndot_II.append(ndot_in) 
            else:
                ndot_II.append(ndot_II[z-1] + r1 + 2*r2 + 3*r4)
            
            dN_G_II = ndot_II[z-1] + r1 + 2*r2 + 3*r4 - ndot_II[z]
            
            # Ethanol
            dy_E_II = (1/N_G_II)*(y_E_II_up*ndot_II[z-1]-r1-r2-y_E_II*ndot_II[z]-y_E_II*dN_G_II)
            
            # Acetaldehyde
            dy_A_II = (1/N_G_II)*(y_A_II_up*ndot_II[z-1]+r1-r4-y_A_II*ndot_II[z]-y_A_II*dN_G_II)
            
            # Hydrogen
            dy_H_II = (1/N_G_II)*(y_H_II_up*ndot_II[z-1]+r1+r2+r3+5*r4-y_H_II*ndot_II[z]-y_H_II*dN_G_II)
            
            # Water
            dy_W_II = (1/N_G_II)*(y_W_II_up*ndot_II[z-1]-r3-3*r4-y_W_II*ndot_II[z]-y_W_II*dN_G_II)
            
            # Carbon dioxide
            dy_CD_II = (1/N_G_II)*(y_CD_II_up*ndot_II[z-1]+r3+2*r4-y_CD_II*ndot_II[z]-y_CD_II*dN_G_II)
            
            # Carbon monoxide
            dy_CM_II = (1/N_G_II)*(y_CM_II_up*ndot_II[z-1]+r2-r3-y_CM_II*ndot_II[z]-y_CM_II*dN_G_II)
            
            # Methane 
            dy_M_II = (1/N_G_II)*(y_M_II_up*ndot_II[z-1]+r2-y_M_II*ndot_II[z]-y_M_II*dN_G_II)
            
            Cp_E = 1.7690e1 + 1.4953e-1*T_G_II + 8.9481e-5*T_G_II**2 - 1.9738e-7*T_G_II**3 + 8.3175e-11*T_G_II**4 
            Cp_A = 2.4528e1 + 7.6013e-2*T_G_II + 1.3625e-4*T_G_II**2 - 1.9994e-7*T_G_II**3 + 7.5955e-11*T_G_II**4
            Cp_H = 1.7639e1 + 6.7005e-2*T_G_II - 1.3148e-4*T_G_II**2 + 1.0588e-7*T_G_II**3 - 2.9180e-11*T_G_II**4
            Cp_W = 3.4047e1 - 9.6506e-3*T_G_II + 3.2998e-5*T_G_II**2 - 2.0447e-8*T_G_II**3 + 4.3023e-12*T_G_II**4
            Cp_CD = 1.9022e1 + 7.9629e-2*T_G_II - 7.3706e-5*T_G_II**2 + 3.7457e-8*T_G_II**3 - 8.1330e-12*T_G_II**4
            Cp_CM = 2.9006e1 + 2.4923e-3*T_G_II - 1.8644e-5*T_G_II**2 + 4.7989e-8*T_G_II**3 - 2.8726e-11*T_G_II**4
            Cp_M = 3.8387e1 - 7.3663e-2*T_G_II + 2.9098e-4*T_G_II**2 - 2.6384e-7*T_G_II**3 + 8.0067e-11*T_G_II**4
            
            Cp = y_E_II*Cp_E + y_A_II*Cp_A + y_H_II*Cp_H + y_W_II*Cp_W + y_CD_II*Cp_CD + y_CM_II*Cp_CM + y_M_II*Cp_M
            
            Cv = Cp - R
            
            # Energy balance
            deltaCp_r1 = (1/1)*Cp_H + (1/1)*Cp_A - (1)*Cp_E
            deltaCp_r2 = (1/1)*Cp_H + (1/1)*Cp_M + (1/1)*Cp_CM - (1)*Cp_E
            deltaCp_r3 = (1/1)*Cp_H + (1/1)*Cp_CD - (1/1)*Cp_W - (1)*Cp_CM
            deltaCp_r4 = (5/1)*Cp_H + (2/1)*Cp_CD - (3/1)*Cp_W - (1)*Cp_A
            
            deltaH_1 = deltaH_std[0] + deltaCp_r1*(T_G_II - Tref2)
            deltaH_2 = deltaH_std[1] + deltaCp_r2*(T_G_II - Tref2)
            deltaH_3 = deltaH_std[2] + deltaCp_r3*(T_G_II - Tref2)
            deltaH_4 = deltaH_std[3] + deltaCp_r4*(T_G_II - Tref2)
            
            Sum_Qrxn = -(r1*deltaH_1 + r2*deltaH_2 + r3*deltaH_3 + r4*deltaH_4)
            
            dT_G_II = (1/(Cv*N_G_II))*(ndot_II[z-1]*Cp*(T_G_II_up - Tref2) + Sum_Qrxn\
                   - ndot_II[z]*Cp*(T_G_II - Tref2) + Q_SG_II) - (T_G_II/N_G_II)*dN_G_II
            
            dXdt[idx:10*n:n] = [dT_S_II,dN_G_II,dy_E_II,dy_A_II,dy_H_II,dy_W_II,dy_CD_II,dy_CM_II,dy_M_II,dT_G_II]
        
        else:
            
            # Separation stage
            j = z - n
            T_S_III,N_G_III,y_E_III,y_A_III,y_H_III,y_W_III,y_CD_III,y_CM_III,y_M_III,T_G_III = X[10*n+j:10*n+10*n:n]

            # Upstream values of variables
            y_E_III_up = X[3*n-1]        if z == n else X[12*n+j-1]
            y_A_III_up = X[4*n-1]        if z == n else X[13*n+j-1]
            y_H_III_up = X[5*n-1]        if z == n else X[14*n+j-1]
            y_W_III_up = X[6*n-1]        if z == n else X[15*n+j-1]
            y_CD_III_up = X[7*n-1]       if z == n else X[16*n+j-1]
            y_CM_III_up = X[8*n-1]       if z == n else X[17*n+j-1]
            y_M_III_up = X[9*n-1]        if z == n else X[18*n+j-1]
            T_G_III_up = X[10*n-1]       if z == n else X[19*n+j-1]
            
            # Heat transfer coefficient
            U = c2*(25)*60 # J/(m2-min-K)
            
            # Equations
            Q_SG_III = U*A_HT*(T_S_III - T_G_III)
            
            dT_S_III = 0*(1/(M_S*C_S))*(Q_HS - Q_SG_III)
            
            Pe = 0.32*Pe0*m.exp(-Eam/(R*T_G_III))
            
            P_III = P_in #(N_G_III*T_G_III*R)/(V_p2)
            
            P_H_r = y_H_III*P_III
            
            P_H_p = 101325 # Pa
            
            ndot_m.append((Pe/delta)*(alpha)*(m.sqrt(P_H_r)-m.sqrt(P_H_p)))
            
            if ndot_m[j] < 0:
                ndot_m[j] = 0
                
            if P_in == 100000:
                ndot_m[j] = 0
            
            if z == n:
                
                ndot_III_in = ndot_II[n-1]*236 # Number of channels
                
                ndot_III.append(ndot_III_in - ndot_m[j]) 
                
                dN_G_III = ndot_III_in - ndot_m[j] - ndot_III[j]

                # Ethanol
                dy_E_III = (1/N_G_III)*(y_E_III_up*ndot_III_in - y_E_III*ndot_III[j] - y_E_III*dN_G_III)
                
                # Acetaldehyde 
                dy_A_III = (1/N_G_III)*(y_A_III_up*ndot_III_in - y_A_III*ndot_III[j] - y_A_III*dN_G_III)
                
                # Hydrogen
                dy_H_III = (1/N_G_III)*(y_H_III_up*ndot_III_in - ndot_m[j] - y_H_III*ndot_III[j] - y_H_III*dN_G_III)

                # Water
                dy_W_III = (1/N_G_III)*(y_W_III_up*ndot_III_in - y_W_III*ndot_III[j] - y_W_III*dN_G_III)
                
                # Carbon dioxide
                dy_CD_III = (1/N_G_III)*(y_CD_III_up*ndot_III_in - y_CD_III*ndot_III[j] - y_CD_III*dN_G_III)
                
                # Carbon monoxide
                dy_CM_III = (1/N_G_III)*(y_CM_III_up*ndot_III_in - y_CM_III*ndot_III[j] - y_CM_III*dN_G_III)
                
                # Methane
                dy_M_III = (1/N_G_III)*(y_M_III_up*ndot_III_in - y_M_III*ndot_III[j] - y_M_III*dN_G_III)
                
                # Energy balance
                Cp_E = 1.7690e1 + 1.4953e-1*T_G_III + 8.9481e-5*T_G_III**2 - 1.9738e-7*T_G_III**3 + 8.3175e-11*T_G_III**4 
                Cp_A = 2.4528e1 + 7.6013e-2*T_G_III + 1.3625e-4*T_G_III**2 - 1.9994e-7*T_G_III**3 + 7.5955e-11*T_G_III**4
                Cp_H = 1.7639e1 + 6.7005e-2*T_G_III - 1.3148e-4*T_G_III**2 + 1.0588e-7*T_G_III**3 - 2.9180e-11*T_G_III**4
                Cp_W = 3.4047e1 - 9.6506e-3*T_G_III + 3.2998e-5*T_G_III**2 - 2.0447e-8*T_G_III**3 + 4.3023e-12*T_G_III**4
                Cp_CD = 1.9022e1 + 7.9629e-2*T_G_III - 7.3706e-5*T_G_III**2 + 3.7457e-8*T_G_III**3 - 8.1330e-12*T_G_III**4
                Cp_CM = 2.9006e1 + 2.4923e-3*T_G_III - 1.8644e-5*T_G_III**2 + 4.7989e-8*T_G_III**3 - 2.8726e-11*T_G_III**4
                Cp_M = 3.8387e1 - 7.3663e-2*T_G_III + 2.9098e-4*T_G_III**2 - 2.6384e-7*T_G_III**3 + 8.0067e-11*T_G_III**4
                
                Cp = y_E_III*Cp_E + y_A_III*Cp_A + y_H_III*Cp_H + y_W_III*Cp_W + y_CD_III*Cp_CD + y_CM_III*Cp_CM + y_M_III*Cp_M
                
                Cv = Cp - R
                
                deltaH_H = Cp_H*(T_G_III - Tref2)
                
                dT_G_III = ((1/(Cv*N_G_III))*(c3*ndot_III_in*Cp*(T_G_III_up - Tref2) - deltaH_H*ndot_m[j]\
                            - ndot_III[j]*Cp*(T_G_III - Tref2) + Q_SG_III) - (T_G_III/N_G_III)*dN_G_III)
            
            else:
                
                ndot_III.append(ndot_III[j-1] - ndot_m[j])
            
                dN_G_III = ndot_III[j-1] - ndot_m[j] - ndot_III[j]
    
                # Ethanol
                dy_E_III = (1/N_G_III)*(y_E_III_up*ndot_III[j-1] - y_E_III*ndot_III[j] - y_E_III*dN_G_III)
                
                # Acetaldehyde 
                dy_A_III = (1/N_G_III)*(y_A_III_up*ndot_III[j-1] - y_A_III*ndot_III[j] - y_A_III*dN_G_III)
                
                # Hydrogen
                dy_H_III = (1/N_G_III)*(y_H_III_up*ndot_III[j-1] - ndot_m[j] - y_H_III*ndot_III[j] - y_H_III*dN_G_III)
    
                # Water
                dy_W_III = (1/N_G_III)*(y_W_III_up*ndot_III[j-1] - y_W_III*ndot_III[j] - y_W_III*dN_G_III)
                
                # Carbon dioxide
                dy_CD_III = (1/N_G_III)*(y_CD_III_up*ndot_III[j-1] - y_CD_III*ndot_III[j] - y_CD_III*dN_G_III)
                
                # Carbon monoxide
                dy_CM_III = (1/N_G_III)*(y_CM_III_up*ndot_III[j-1] - y_CM_III*ndot_III[j] - y_CM_III*dN_G_III)
                
                # Methane
                dy_M_III = (1/N_G_III)*(y_M_III_up*ndot_III[j-1] - y_M_III*ndot_III[j] - y_M_III*dN_G_III)
                
                # Energy balance
                Cp_E = 1.7690e1 + 1.4953e-1*T_G_III + 8.9481e-5*T_G_III**2 - 1.9738e-7*T_G_III**3 + 8.3175e-11*T_G_III**4 
                Cp_A = 2.4528e1 + 7.6013e-2*T_G_III + 1.3625e-4*T_G_III**2 - 1.9994e-7*T_G_III**3 + 7.5955e-11*T_G_III**4
                Cp_H = 1.7639e1 + 6.7005e-2*T_G_III - 1.3148e-4*T_G_III**2 + 1.0588e-7*T_G_III**3 - 2.9180e-11*T_G_III**4
                Cp_W = 3.4047e1 - 9.6506e-3*T_G_III + 3.2998e-5*T_G_III**2 - 2.0447e-8*T_G_III**3 + 4.3023e-12*T_G_III**4
                Cp_CD = 1.9022e1 + 7.9629e-2*T_G_III - 7.3706e-5*T_G_III**2 + 3.7457e-8*T_G_III**3 - 8.1330e-12*T_G_III**4
                Cp_CM = 2.9006e1 + 2.4923e-3*T_G_III - 1.8644e-5*T_G_III**2 + 4.7989e-8*T_G_III**3 - 2.8726e-11*T_G_III**4
                Cp_M = 3.8387e1 - 7.3663e-2*T_G_III + 2.9098e-4*T_G_III**2 - 2.6384e-7*T_G_III**3 + 8.0067e-11*T_G_III**4
                
                Cp = y_E_III*Cp_E + y_A_III*Cp_A + y_H_III*Cp_H + y_W_III*Cp_W + y_CD_III*Cp_CD + y_CM_III*Cp_CM + y_M_III*Cp_M
                
                Cv = Cp - R
                
                deltaH_H = Cp_H*(T_G_III - Tref2)
                
                dT_G_III = ((1/(Cv*N_G_III))*(c3*ndot_III[j-1]*Cp*(T_G_III_up - Tref2) - deltaH_H*ndot_m[j]\
                            - ndot_III[j]*Cp*(T_G_III - Tref2) + Q_SG_III) - (T_G_III/N_G_III)*dN_G_III)
             
            if z == n:
                ndot_IV.append(ndot_m[j])
            else:
                ndot_IV.append(ndot_IV[j-1] + ndot_m[j])
 
            dXdt[10*n+j:10*n+10*n:n] = [dT_S_III,dN_G_III,dy_E_III,dy_A_III,dy_H_III,dy_W_III,dy_CD_III,dy_CM_III,dy_M_III,dT_G_III]
            
    # Record only when solver moves to a new output step
    if time > t_prev:
        pure_H2.append((time, ndot_IV[-1]))
        t_prev = time
            
    return dXdt

# Time span 
t_span = (0,3)         

start = timer()

# Solve ODE system
X_sol = solve_ivp(ODEs, t_span, X0, method='BDF')

end = timer()

print(f"Computation time: {end - start:.4f} seconds")

#%%

t = X_sol.t
X = X_sol.y.T

axial = np.linspace(0, 2*n, 2*n)
y_E  = np.concatenate((X[-1, 2*n:3*n], X[-1, 12*n:13*n]))
y_A  = np.concatenate((X[-1, 3*n:4*n], X[-1, 13*n:14*n])) 
y_H  = np.concatenate((X[-1, 4*n:5*n], X[-1, 14*n:15*n]))
y_W  = np.concatenate((X[-1, 5*n:6*n], X[-1, 15*n:16*n]))
y_CD = np.concatenate((X[-1, 6*n:7*n], X[-1, 16*n:17*n]))
y_CM = np.concatenate((X[-1, 7*n:8*n], X[-1, 17*n:18*n]))
y_M  = np.concatenate((X[-1, 8*n:9*n], X[-1, 18*n:19*n]))
T_G  = np.concatenate((X[-1, 9*n:10*n], X[-1, 19*n:20*n]))
ndot = np.concatenate((np.multiply(236,ndot_II),ndot_III)) 

plt.figure(1)
plt.title('Molar flow rate - axial profiles')
plt.plot(axial,np.multiply(ndot,y_E)*1000,linewidth=2,label='Ethanol')
plt.plot(axial,np.multiply(ndot,y_A)*1000,linewidth=2,label='Acetaldehyde')
plt.plot(axial,np.multiply(ndot,y_H)*1000,linewidth=2,label='Hydrogen')
plt.plot(axial,np.multiply(ndot,y_W)*1000,linewidth=2,label='Water')
plt.plot(axial,np.multiply(ndot,y_CD)*1000,linewidth=2,label='Carbon Dioxide')
plt.plot(axial,np.multiply(ndot,y_CM)*1000,linewidth=2,label='Carbon Monoxide')
plt.plot(axial,np.multiply(ndot,y_M)*1000,linewidth=2,label='Methane')
plt.grid(True)
plt.ylabel('Molar flow rate (mmol/min)')
plt.xlabel('Axial position (-)')
plt.legend(loc='upper right')

plt.figure(2)
plt.title('Temperature - axial profile')
plt.plot(axial,T_G,linewidth=2,label='Temperature')
plt.grid(True)
plt.ylabel('Temperature (K)')
plt.xlabel('Axial position (-)')
plt.legend(loc='best')

time_plot, pure_H2_plot = zip(*pure_H2)

plt.figure(3)
plt.title('Pure Hydrogen - dynamic molar flow')
plt.plot(time_plot,np.array(pure_H2_plot)*1000,linewidth=2,label='Pure Hydrogen')
plt.text(0.6, 0.235, 
         '*Small transient at the beginning due to steady-state \n mismatch with initial conditions',
         fontsize=10, color='blue')
plt.grid(True)
plt.ylabel('Molar flow rate (mmol/min)')
plt.xlabel('Time (min)')
plt.legend(loc='best')


