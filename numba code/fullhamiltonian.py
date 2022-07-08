# full hamiltonian system
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from numba import jit,njit,vectorize, float64, int32
import numba as nb


def F(q):
    return q**2/2


@njit(float64(float64))
def nablaF(q):
    return -q

@njit(float64[:](float64[:],float64))
def A(qp,h):
    q=qp[0]
    p=qp[1]
    q = q+p*h
    qp=np.array([q,p])
    return (qp)

@njit(float64[:](float64[:],float64))
def B(qp,h):
    q=qp[0]
    p=qp[1]
    p = p+nablaF(q)*h
    qp=np.array([q,p])
    return (qp)

@njit(float64[:](float64[:],float64,float64,float64))
def O(qp,h,gamma,beta):
    q=qp[0]
    p=qp[1]
    dB = np.random.normal(0,1,1)[0]
    alpha =np.exp(-gamma*h)
    p = alpha*p+ np.sqrt((1-alpha*alpha)/beta)*dB
    qp=np.array([q,p])
    return (qp)

@njit(float64[:](float64[:],float64,float64,float64,float64))
def one_traj(qp,T,h,gamma,beta):
    for i in range(int(T/h)):
        qp=B(qp,h)
        qp=A(qp,h)
        qp=O(qp,h,gamma,beta)
        qp=A(qp,h)
        qp=B(qp,h)
    return (qp)

@njit(parallel=True)
def method_baoab(T,gamma,beta,h,N):
    qp_list=np.zeros((N,2))
    qipi = np.array([1.0,1.0]) #np.random.normal(0,1,2) #initial conditions
    for j in nb.prange(N):
        qfpf = one_traj(qipi,T,h,gamma,beta)
        qp_list[j,::]=qfpf
    return(qp_list)

    
# axis of the plot 
def plot_qp(qp,beta,gamma):
    fig, (ax1,ax2,ax3)= plt.subplots(1, 3,figsize=(16,6))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)

    fig.suptitle("$\\beta$="+str(beta)+", $\\gamma=$"+str(gamma)+", $N=$"+str(len(qp[::,0])))

    #Plot 1
    ## position q experiment
    histogram,bins = np.histogram(qp[::,0],bins=100,range=[-3,3], density=True)
    midx = (bins[0:-1]+bins[1:])/2
    ax1.plot(midx,histogram,label='q-Experiment')

    #Plot 2: 
    ### momentum p experiment
    histogram,bins = np.histogram(qp[::,1],bins=100,range=[-3,3], density=True)
    midx = (bins[0:-1]+bins[1:])/2
    # histogram=(histogram/np.sum(histogram)*(midx[1]-midx[0]) )
    ax2.plot(midx,histogram,label='p-Experiment')


    ### mementum p true
    rho = np.exp(- beta*(midx**2)/2)
    rho = rho / ( np.sum(rho)* (midx[1]-midx[0]) ) # Normalize rho by dividing by its approx. integral
    ax2.plot(midx,rho,'--',label='Truth')
    ax1.plot(midx,rho,'--',label='Truth')
    ax2.legend() 

    #Plot 3 
    ax3.set_ylim(-5,5)
    ax3.plot(qp[::,1],label="p")
    ax3.plot(qp[::,0],label="q")

    ax3.legend()