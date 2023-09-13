import matplotlib.pyplot as plt
import scipy.integrate as spi
import numpy as np

#Define constants

p = {
    "E_Na" : 55, #all voltages given in mV
    "g_bar_Na" : 120, #mmho/cm^2
    "E_K" : -72,
    "g_bar_K" : 20,
    "E_A" : -75,
    "g_bar_A" : 47.7, #in some computations the authors used 33
    "C" : 1,
    "E_L" : -17, 
    "G_L" : 0.3
}

#define the functions for the first-order kinetics (like in the appendix of the given paper)

def Falpha_m(V):
    MSHFT = -5.3
    alpha_m = -0.1*(V + 35 + MSHFT) / (np.exp( -(V + 35 + MSHFT)/10 -1))
    return alpha_m

def Fbeta_m(V):
    MSHFT = -5.3
    beta_m = 4 * np.exp(-(V+ 60 + MSHFT)/18)
    return beta_m

def Falpha_h(V):
    HSHFT = -12
    alpha_h = 0.07 * np.exp(-(V + 60 + HSHFT)/20)
    return alpha_h

def Fbeta_h(V):
    HSHFT = -12
    beta_h = 1/(np.exp(-(V + 30 + HSHFT)/10 + 1))
    return beta_h
  
def Falpha_n(V):
    NSHFT = -4.3
    alpha_n = -0.01*(V + 50 + NSHFT)/ (np.exp(-(V + 50 + NSHFT)/10) -1)
    return alpha_n

def Fbeta_n(V):
    NSHFT = -4.3
    beta_n = 0.125 * np.exp(-(V + 60 + NSHFT)/80)
    return beta_n

def FA_inf(V):
    A_inf = ( 0.0761 *  np.exp((V + 94.22)/31.84)/(1+ np.exp((V+1.17)/28.93)) )**(1/3)
    return A_inf

def FB_inf(V):
    B_inf = 1/(1+ np.exp((V + 53.3)/14.54))**4
    return B_inf

def Ftau_A(V):
    tau_A = 0.3632 + 1.158 / (1 + np.exp((V + 55.96)/20.12))
    return tau_A

def Ftau_B(V):
    tau_B = 1.24 + 2.678 / (1+ np.exp((V + 50)/16.027))
    return tau_B

def Fm_inf(V):
    m_inf = Falpha_m(V) / ( Falpha_m(V) + Fbeta_m(V) )
    return m_inf

def Fh_inf(V):
    h_inf = Falpha_h(V) / ( Falpha_h(V) + Fbeta_h(V) )
    return h_inf

def Fn_inf(V):
    n_inf = Falpha_n(V) / ( Falpha_n(V) + Fbeta_n(V) )
    return n_inf

def I(t):
    I = 8 #mycro Ampere / cm^2
    return I



#Define the ODE

def f_ConnorStevens(y, t, p):
        
    #define solution vector. 
    V, m, h, n = y
    
    #include/discretize the functions with the current parameters
    alpha_m = Falpha_m(V)
    beta_m = Fbeta_m(V)
    alpha_h = Falpha_h(V)
    beta_h = Fbeta_h(V)
    alpha_n = Falpha_n(V)
    beta_n = Fbeta_n(V)
    A_inf = FA_inf(V)
    B_inf = FB_inf(V)
    tau_A = Ftau_A(V)
    tau_B = Ftau_B(V)
    
    dV = 1/p["C"] * (     I(t) - p["g_bar_Na"] * m**3 * h *(V - p["E_Na"]) - p["g_bar_K"] * n**4 *(V - p["E_K"]) - p["g_bar_A"] * (  A_inf * (1 - np.exp(-t/tau_A))  )**3 * B_inf * (np.exp(-t/tau_B)) * (V - p["E_A"]) - p["G_L"] * (V - p["E_L"])    ) 
    dm = alpha_m * (1-m) - beta_m * m
    dh = alpha_h * (1-h) - beta_h * h
    dn = alpha_n * (1-n) - beta_n * n
        
    return [dV, dm, dh, dn]

#set initial potential
V_0 = -68

#Calculate the resting steady state values of m,n,h at V_0. They are the values of fraction of open channels that will be reached after infinite time for a certain membrane potential
m_inf_V0 = Fm_inf(V_0)
n_inf_V0 = Fn_inf(V_0)
h_inf_V0 = Fh_inf(V_0)

#Initial Condition
y0 = np.array([V_0, m_inf_V0, h_inf_V0, n_inf_V0])         

#Define the Simulation Time
t0 = 0      #start time
t_end = 50  #end time
step = 0.0001 #time step
tps = np.arange(t0,t_end,step)      #create an array for the time steps from t0 to T with step size T

#Run the Simulation 
y = spi.odeint(f_ConnorStevens, y0, tps, args=(p,)) 





#Plot
plt.close('all')
plt.figure(1)
plt.clf()
plt.subplot(211)
#plt.ylim([85,105])
#plt.ylim([32,40])
plt.plot(tps,y[:,0],'r', label = "V")    #plot y1
#plt.plot(tps,y[:,1],'b', label = "p_o")    #plot y2
plt.xlabel("Time in miliseconds")
plt.ylabel("Membrane Potential")
plt.legend()
plt.grid()
#plt.subplot(212)
#plt.plot(y[:,0],y[:,1],'r') #phase portrait
#plt.xlabel("Phase space")
#plt.ylim([87,89.5])
#x_min,x_max = plt.xlim()    #get limits of the phase portrait
#y_min,y_max = plt.ylim()

#N_arrow = 40                #chose to plot 20 arrows
#xx = np.arange(x_min,x_max+(x_max-x_min)/(N_arrow-1),(x_max-x_min)/(N_arrow-1)) #create x coordinates for the positions of the arrows. they are set with step size x_max-x_min / N 
#yy = np.arange(y_min,y_max+(y_max-y_min)/(N_arrow-1),(y_max-y_min)/(N_arrow-1))
#X,Y = np.meshgrid(xx,yy)    #create an x,y field 
#FX = 0*X
#FY = 0*Y
#n_shape,p_shape = np.shape(X)
#for i in range(n_shape):
#    for j in range(p_shape):
#        Z = f_O2_Model1a([X[i,j],Y[i,j]],1,p)    #compute the field of change. you can insert any time, e.g. t = 0 as it does not depend on t
#        FX[i,j] = Z[0]
#        FY[i,j] = Z[1]
#plt.quiver(X,Y,FX,FY,angles='xy',scale_units='xy',width = 0.001)

plt.show()

# #calculate the currents:
# V = y[:,0]
# m = y[:,1]
# h = y[:,2]
# n = y[:,3]
# #V_inf = V[-1]
# I_Na = p["g_bar_Na"] * m**3 * h *(V - p["E_Na"])
# I_K = p["g_bar_K"] * n**4 *(V - p["E_K"])
# #I_A = p["g_bar_A"] * (  FA_inf(V) * (1 - np.exp(-t/Ftau_A(V)))  )**3 * FB_inf(V) * (np.exp(-t/Ftau_B(V))) * (V - p["E_A"])
# plt.figure(2)
# plt.plot(tps,I_Na,'r', label = "I_Na")