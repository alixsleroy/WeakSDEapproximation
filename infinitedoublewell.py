from numba import jit,njit,vectorize, float64, int32
import numba as nb
import numpy as np
import time



# define the gradV function 
@njit(float64(float64))
def dU(x):
    """
    Compute the potential of the infinite double well:
    x: float 
    """

    return -1/(x*x*x)+2*x

@njit(float64(float64,float64,float64,float64))
def e_m_fast(y0,s,b1,dt):
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
    y1=y0 - dU(y0)*dt+s*b1
    return y1    
    



@njit() #float64[:](float64,float64,float64,float64))
def DW_sde_fast(N,dt,tau): # Function is compiled and runs in machine code
    """
    Input
    -------
    n_samples: int
        Number of sample to draw
    T: int 
        Final time
    N: int
        Number of time steps 
    tau: float
        Value of the temperature of the DW SDE (+ sqrt(2*tau)*dW)
    method: function
        Numerical scheme used for the DW SDE
    Return
    -------
    y_final: np.array
        Array of shape (M,). Sample of numerical approximation of the DW SDE at time T
    
    """
    y_final = [] #np.zeros(n_samples)
    s = np.sqrt(2*tau*dt)
    n_esc=0
    y0 = 2 #initial condition
    for jj in range(N): # Run until T= Tsec
        b1 = np.random.normal(0,1,1)[0]
        y1 = e_m_fast(y0,s,b1,dt)
        y0=y1 
        y_final.append(y1)
    y_final=np.array(y_final)
    return y_final


## Infinite double well system
    
def U(x):
    """
    potential of the infinite double well
    """

    return (1/(2*x**2)+x*x)

def pinf(x,tau):
    return np.exp(-U(x)/tau)

def dU(x):
    """
    potential of the double well
    """

    return -(-x**(-3)+2*x)
#print(DW_sde_fast(10**2,10**(-6),0.001)) # compile the function
