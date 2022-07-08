from numba import jit,njit,vectorize, float64, int32
import numba as nb
import numpy as np
import time




@njit(float64(float64,float64,float64,float64))
def em_lin_plus(y0,s,b1,dt):
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
    a=-8
    b=np.power(2,11/4)
    y1=y0 + (a*y0+b)*dt+s*b1
    return y1
    
@njit(float64(float64,float64,float64,float64))
def em_lin_minus(y0,s,b1,dt):
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
    a=4
    b=np.power(2,3/4)
    y1=y0 + (a*y0+b)*dt+s*b1
    return y1 


@njit() #float64[:](float64,float64,float64,float64))
def DW_sde_fast_lin(N,dt,tau,em_method,y0): # Function is compiled and runs in machine code
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
    for jj in range(N): # Run until T= Tsec
        b1 = np.random.normal(0,1,1)[0]
        y1 = em_method(y0,s,b1,dt)
        y0=y1 
        y_final.append(y1)
    y_final=np.array(y_final)
    return y_final


    

#print(DW_sde_fast_lin(1000,0.1,0.1)) # compile the function

