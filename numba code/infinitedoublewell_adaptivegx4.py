from numba import jit,njit,vectorize, float64, int32
import numba as nb
import numpy as np

import matplotlib.pyplot as plt

## ---------------- Mathplotlib settings ----------------
SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 25

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def U(x):
    """
    potential of -the infinite double well
    """

    return (1/(2*x**2)+x*x)
    
# define the gradV function 
@njit(float64(float64))
def minusdU(x):
    """
    Compute the potential of the infinite double well:
    x: float 
    """

    return -1/(x*x*x)+2*x

# define the adaptive function 
@njit(float64[:](float64,float64,float64[:]))
def g4(x,h,dtbounds):
    """
    Compute the value of the adaptive function choosen:
    x: float 
    """
    dtmin=dtbounds[0]
    dtmax=dtbounds[1]

    M=h/dtmin
    m=h/dtmax

    x3=np.power(x,3)

    # value of function f, f' and f^2
    f=(1/x3-2*x)
    fprime=-(3/(x3*x)+2)
    f2 = f*f

    #compute the value of phi(f(x)) = \sqrt{f(x)^2}
    phif = np.sqrt(f2)
    phif2 = f2*f2

    # value of m^2
    m2 = m*m

    #compute gx
    gx_den=np.sqrt(phif2+m2)
    gx_num = gx_den/M + 1 
    gx=gx_num/gx_den

    #compute gx prime 
    gxp_num= -f*fprime
    gxp_den = gx_den*gx_den*gx_den
    gxprime= gxp_num/gxp_den

    #round number to avoid having too large number 
    gx =gx
    gxprime = gxprime

    #return
    re=np.array([gx,gxprime])
    return re

@njit(float64[:](float64,float64,float64,float64,float64[:]))
def e_m_ada4(y0,s,b1,dt,dtbounds):
    """
    The Euler-Maruyama scheme applied to the infinite double well
    y0: float
        value of y at t_n
    tau: float
        value of the temperature 
    b1: float
        brownian increment 
    dt: float
        time increment
    """
    re=g4(y0,dt,dtbounds)
    gy=re[0]
    nablag=re[1]
    newdt = np.round(gy*dt,7)
    y1=y0-minusdU(y0)*newdt+nablag*dt+s*b1*np.sqrt(gy)
    yt=np.array([y1,newdt])
    return yt    


# @njit(nb.types.UniTuple(nb.float64,2)(float64,float64,float64,float64,float64))
# def run_num(N,dt,s,T,n_esc):

@njit(float64[:](float64,float64,float64,float64[:]))
def run_num_ada4(T,dt,s,dtbounds):
    """
    Run the simulation for one sample path
    Input
    -----
    Ntot: int
        Number of steps to take to get to Tf with dt
    dt: float 
        Value of time increment. Is 1/N.
    s: float 
        Is sqrt(2 \tau dt). 
    Return
    ------
    yf: float
        Value of X(T) as approximated by the numerical scheme chosen
    """

    y0 = 1
    t=0
    t_count=0.0
    while t<T:
    # for jj in range(Ntot): # Run until T= Tsec
        b1 = np.random.normal(0,1,1)[0]
        yt = e_m_ada4(y0,s,b1,dt,dtbounds)
        y1=yt[0]
        newdt=yt[1]
        y0=y1 
        t=np.round(t+newdt,7)
        t_count=t_count+1
    y_t = np.array([y0,t_count])
    return (y_t)



@njit(parallel=True)
def IDW_nsample_ada4(n_samples,T,dt,dtbounds): # Function is compiled and runs in machine code

    """
    Input
    -------
    n_samples: int
        Number of sample to draw
    T: int 
        Final time
    dt: float
        Size of the time discretization 
    tau: float
        Value of the temperature of the DW SDE (+ sqrt(2*tau)*dW)
    method: function
        Numerical scheme used for the DW SDE
    Return
    -------
    y_final: np.array
        Array of shape (M,). Sample of numerical approximation of the DW SDE at time T
    
    """
    # N = int(np.round(1/dt,6))  #size of the time steps
    # Ntot = N*T #total number of steps to take to arrive at T in steps of dt 
    # y_final = [] #np.zeros(n_samples)
    # t_tot=[] #number of force evaluation per run
    yt_list=np.zeros((n_samples,2)) #vector to save solutions and number of run
    s = np.sqrt(2*dt)
    for i in range(n_samples):
        yf_t =run_num_ada4(T,dt,s,dtbounds)
        yt_list[i,::]=yf_t
    return yt_list






# #%time ytest=y_compile = DW_sde_fast(1000,3,10,0.01,20) # compile the function
# def plot_dist(y,tau,dt,n_samples,T,title,ax):

#     ax.set_title(str(title)+", $\\tau$="+str(tau)+", h="+str(dt)+", \n N="+str(n_samples)+", T="+str(T))

#     #Plot 1
#     histogram,bins = np.histogram(y,bins=100,range=[-3,3], density=True)

#     midx = (bins[0:-1]+bins[1:])/2
#     histogram=(histogram/np.sum(histogram))
#     ax.plot(midx,histogram,label='q-Experiment')

#     rho = np.exp(- (U(midx)/tau))
#     rho = rho / ( np.sum(rho) * (midx[1]-midx[0]) ) 
#     rho=(rho/np.sum(rho))*2
#     rho=[rho[i] if i>50 else 0 for i in range(len(rho))]
#     ax.plot(midx,rho,'--',label='Truth') 
#     ax.legend()

## Parameters 
# print("prout")
# ## compile
# ytest= IDW_nsample_ada4(10,1,0.1,np.array([0.01,1])) # compile the function
# print("prout")

# n_samples=10**4
# T=10
# tau=0.15
# dt=1
# dtbounds = np.array([0.001,0.1])

# y_adag4= IDW_nsample_ada4(n_samples,T,dt,dtbounds) 
# fig, (ax1)= plt.subplots(1,1,figsize=(18,10))# plt.figure(figsize=(4,4))
# fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# plot_dist(y_adag4,tau,dt,n_samples,T,"adaptive g4",ax1)
# plt.show()