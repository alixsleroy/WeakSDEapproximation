import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from numba import jit,njit,vectorize, float64, int32
import numba as nb
import time as time

def F(q):
    return q**2/2


@njit(float64(float64))
def nablaU(q):
    return q


@njit(float64(float64,float64,float64[:]))
def g(x,h,dtbounds):
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
    f=np.abs(nablaU(x))
    f2 = f*f

    #compute the value of phi(f(x)) = \sqrt{f(x)^2}
    phif2 = f2*f2

    # value of m^2
    m2 = m*m

    #compute gx
    gx_den=np.sqrt(phif2+m2)
    gx_num = gx_den/M + 1 
    gx=gx_num/gx_den

    #return
    re=gx
    return re


@njit(float64[:](float64[:],float64,float64[:]))
def A_ada(qp,h,dtbounds):
    q=qp[0]
    p=qp[1]
    gq = qp[2]
    ## fixed point method for g((qn+1+qn)/2)
    g_half = g(q+0.5*h*p*gq,h,dtbounds)
    g_half = g(q+0.5*h*p*g_half,h,dtbounds)
    g_half = g(q+0.5*h*p*g_half,h,dtbounds)
    g_half = g(q+0.5*h*p*g_half,h,dtbounds)
    gq = g_half
    q = q+p*gq*h
    qp=np.array([q,p])
    return (qp)

@njit(float64[:](float64[:],float64,float64[:]))
def B_ada(qp,h,dtbounds):
    q=qp[0]
    p=qp[1]
    gq=g(q,h,dtbounds)
    p = p-gq*nablaU(q)*h
    qp_gq=np.array([q,p,gq])
    return (qp_gq)

@njit(float64[:](float64[:],float64,float64[:],float64,float64))
def O_ada(qp,h,dtbounds,gamma,beta):
    q=qp[0]
    p=qp[1]
    dB = np.random.normal(0,1,1)[0]
    gq=g(q,h,dtbounds)
    alpha =np.exp(-gamma*h*gq)
    p = alpha*p+ np.sqrt((1-alpha*alpha)/beta)*dB
    qp_gq=np.array([q,p,gq])
    return (qp_gq)

@njit(float64[:](float64[:],float64,float64,float64[:],float64,float64))
def one_traj_ada(qp,T,h,dtbounds,gamma,beta):
    t=0
    h_half=h/2
    tcount=0
    while t<T:
        qp_gq=B_ada(qp,h_half,dtbounds)
        qp=A_ada(qp_gq,h_half,dtbounds)
        qp_gq=O_ada(qp,h_half,dtbounds,gamma,beta)
        qp=A_ada(qp_gq,h_half,dtbounds)
        qp_gq=B_ada(qp,h_half,dtbounds)
        qp=qp_gq[:2]
        gq=qp_gq[2]
        t=np.round(t+gq*h,7)
        tcount=tcount+1
    qp_t=np.append(qp,tcount)
    return (qp_t)
    

@njit(parallel=True)
def method_baoab_ada2(n_samples,T,gamma,beta,h,dtbounds):
    qpt_list=np.zeros((n_samples,3))
    qipi = np.array([1.0,1.0]) #np.random.normal(0,1,2) #initial conditions
    for j in nb.prange(n_samples):
        qfpftf = one_traj_ada(qipi,T,h,dtbounds,gamma,beta)
        qpt_list[j,::]=qfpftf
    return(qpt_list)

#compile the method
print(method_baoab_ada2(10,10,0.1,0.5,0.1,np.array([0.01,0.1])))



