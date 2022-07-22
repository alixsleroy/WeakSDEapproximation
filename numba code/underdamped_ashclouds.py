import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from numba import jit,njit,vectorize, float64, int32
import numba as nb
import time as time


@njit(float64(float64,float64))
def sigmaU(X,H):
    kappa_sigma = 1.3
    ustar=0.2
    sigmaU = kappa_sigma*ustar*np.power((1-X/H),3/4)
    eps=0.1
    sigmaU_eps = kappa_sigma*ustar*np.power(1-eps/H,3/4)
    sigmaU_Heps = kappa_sigma*ustar*np.power(1-(H-eps)/H,3/4)
    final_s = (X<eps)*sigmaU_eps+(X>(H-eps))*sigmaU_Heps+ (X<=(H-eps))*(X>=eps)*sigmaU
    return final_s

@njit(float64(float64,float64))
def sigmaUprime(X,H):
    kappa_sigma = 1.3
    ustar=0.2
    sigmaUprimeX = (-3*kappa_sigma*ustar)/(4*H)*np.power((1-X/H),-1/4)
    eps=0.1
    final_v = (X<eps)*0+(X>(H-eps))*0+ (X<=(H-eps))*(X>=eps)*sigmaUprimeX
    return final_v


@njit(float64(float64,float64))
def nablaV(X,U):
    H=1
    sigmaUX = sigmaU(X,H)
    re = - sigmaUprime(X,H)*(sigmaU(X,H)+U*U/sigmaUX)
    return re

@njit(float64(float64,float64))
def lambda_U(X,H):
    kappa_tau=0.5
    tau = kappa_tau*X/sigmaU(X,H) 
    eps=0.1
    tau_eps= kappa_tau*eps/sigmaU(eps,H) 
    tau_Heps = kappa_tau*(H-eps)/sigmaU(H-eps,H) 
    final_t = (X<eps)*tau_eps+(X>(H-eps))*tau_Heps+ (X<=(H-eps))*(X>=eps)*tau
    return 1/final_t

@njit(float64(float64,float64,float64[:]))
def g(X,h,dtbounds):
    return 1

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

@njit(float64[:](float64[:],float64,float64,float64[:],float64))
def B_ada(qp,n,h,dtbounds,H):
    q=qp[0]
    p=qp[1]
    gq=g(q,h,dtbounds)
    sigmaUq=sigmaU(q,1)
    sigmaUq2=sigmaUq*sigmaUq
    p = sigmaUq2*np.tan((n+1)*h*np.abs(gq)*np.abs(sigmaUq))
    p=(q<=0)*-p+(q>=H)*-p+(q>=0)*(q<=H)*p
    qp_gq=np.array([q,p,gq])
    return (qp_gq)
# B_ada(np.array([0.0,1.1]),1,0.1,np.array([0.0,0.1]),1)


@njit(float64[:](float64[:],float64,float64[:],float64))
def O_ada(qp,h,dtbounds,H):
    q=qp[0]
    p=qp[1]
    dB = np.random.normal(0,1,1)[0]
    gq=g(q,h,dtbounds)
    lambdap=lambda_U(q,H)
    alpha =np.exp(-lambdap*h*gq)
    sigmaUq=sigmaU(q,H)
    s = sigmaUq*sigmaU(q,H)/(2*lambdap)
    p = alpha*p+ np.sqrt((1-alpha*alpha)*s)*dB
    qp_gq=np.array([q,p,gq])
    return (qp_gq)

@njit(float64[:](float64[:],float64,float64,float64[:],float64))
def one_traj_ada(qp,T,h,dtbounds,H):
    t=0
    h_half=h/2
    tcount=0
    while t<T:
        tcount=tcount+1
        qp_gq=B_ada(qp,tcount,h_half,dtbounds,H)
        qp=A_ada(qp_gq,h_half,dtbounds)
        qp_gq=O_ada(qp,h_half,dtbounds,H)
        qp=A_ada(qp_gq,h_half,dtbounds)
        qp_gq=B_ada(qp,tcount,h_half,dtbounds,H)
        qp=qp_gq[:2]
        gq=qp_gq[2]
        t=np.round(t+gq*h,7)
    qp_t=np.append(qp,tcount)
    return (qp_t)
    

@njit(parallel=True)
def method_baoab_ada2(n_samples,T,H,h,dtbounds):
    qpt_list=np.zeros((n_samples,3))
    qipi = np.array([0.5,0.5]) #np.random.normal(0,1,2) #initial conditions
    for j in nb.prange(n_samples):
        qfpftf = one_traj_ada(qipi,T,h,dtbounds,H)
        qpt_list[j,::]=qfpftf
    return(qpt_list)

# #compile the method
# qpt_list=method_baoab_ada2(1,10,1.0,0.1,np.array([0.01,0.1]))
# print(qpt_list)
n_samples=1
T=1
h=0.1
dtbounds=np.array([0.1,0.1])
H=1
# qpt_list=np.zeros((n_samples,3))
qipi = np.array([0.5,0.5]) #np.random.normal(0,1,2) #initial conditions
t=0
h_half=h/2
tcount=0
qp = np.array([0.5,0.5]) #np.random.normal(0,1,2) #initial conditions

while t<T:
    tcount=tcount+1
    qp_gq=B_ada(qp,tcount,h_half,dtbounds,H)
    qp=A_ada(qp_gq,h_half,dtbounds)
    qp_gq=O_ada(qp,h_half,dtbounds,H)
    qp=A_ada(qp_gq,h_half,dtbounds)
    qp_gq=B_ada(qp,tcount,h_half,dtbounds,H)
    qp=qp_gq[:2]
    gq=qp_gq[2]
    t=np.round(t+gq*h,7)
    print(t)
    qp_t=np.append(qp,tcount)

plt.plot(qp_t)
plt.show()
print(qp_t)