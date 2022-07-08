
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from numba import jit,njit,vectorize, float64, int32
import numba as nb
import time as time

def F(q):
    return q**2/2


@njit(float64(float64))
def nablaF(q):
    return q

# @njit(float64(float64))
# def g(q):
#     dtmin=0.001
#     dtmax=0.1
#     R=1
#     y= (dtmax-dtmin)*(1-np.exp(-q*R))+dtmin
#     return y

@njit(float64(float64))
def g(q):
    dtmin=0.01
    dtmax=0.1
    R=1
    y= (dtmax-dtmin)*(1-np.exp(-q*R))+dtmin
    return y


@njit(float64[:](float64[:],float64))
def A_ada(qp,h):
    q=qp[0]
    p=qp[1]
    gq = qp[2]
    q = q+p*gq*h
    qp=np.array([q,p])
    return (qp)

@njit(float64[:](float64[:],float64))
def B_ada(qp,h):
    q=qp[0]
    p=qp[1]
    gq=g(q)
    p = p+gq*nablaF(q)*h
    qp_gq=np.array([q,p,gq])
    return (qp_gq)

@njit(float64[:](float64[:],float64,float64,float64))
def O_ada(qp,h,gamma,beta):
    q=qp[0]
    p=qp[1]
    dB = np.random.normal(0,1,1)[0]
    gq=g(q)
    alpha =np.exp(-gamma*h*gq)
    p = alpha*p+ np.sqrt((1-alpha*alpha)/beta)*dB
    qp_gq=np.array([q,p,gq])
    return (qp_gq)

@njit(float64[:](float64[:],float64,float64,float64,float64))
def one_traj_ada(qp,T,h,gamma,beta):
    for i in range(int(T/h)):
        qp_gq=B_ada(qp,h)
        qp=A_ada(qp_gq,h)
        qp_gq=O_ada(qp,h,gamma,beta)
        qp=A_ada(qp_gq,h)
        qp_gq=B_ada(qp,h)
        qp=qp_gq[:2]
    return (qp)

@njit(parallel=True)
def method_baoab_ada(T,gamma,beta,h,N):
    qp_list=np.zeros((N,2))
    qipi = np.array([1.0,1.0]) #np.random.normal(0,1,2) #initial conditions
    for j in nb.prange(N):
        qfpf = one_traj_ada(qipi,T,h,gamma,beta)
        qp_list[j,::]=qfpf
    return(qp_list)