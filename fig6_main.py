import matplotlib.pyplot as plt
from os import environ
environ["OMP_NUM_THREADS"] = "1"
environ["OPENBLAS_NUM_THREADS"] = "1"
environ["MKL_NUM_THREADS"] = "1"
environ["VECLIB_MAXIMUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
from time import gmtime, strftime
import numpy as np
import math
import multiprocessing
from multiprocessing import Pool
import csv
from numpy import linalg as la
#from scipy.stats import kde
#from scipy.fftpack import fft, ifft
#from IPython import get_ipython;   
#from itertools import repeat, izip
#get_ipython().magic('reset -sf')
#plt.close('all')


rmin = 1.1
rmax = 4.0
numr = 20
rvec = np.linspace(rmin,rmax,numr, endpoint=True) 

eff0 = 0.001

EPS = 0.05
g = np.sqrt(1.5)

sigma0 = 0.1
x0 = 0.3

numtrial = 1

#mu = 0.6


from numpy.linalg import qr


##############################################################################
# Smooth projection onto [0,1]
##############################################################################



def H(z):
    """
    Smooth clipping function.

    H(z) ≈ 0 for z<0
    H(z) ≈ z for 0<z<1
    H(z) ≈ 1 for z>1
    """

    return (
        0.5
        - 0.5 * EPS * np.log(np.cosh((1.0 - z) / EPS))
        + 0.5 * EPS * np.log(np.cosh(z / EPS))
    )


def Hprime(z):
    """
    Exact derivative of H.
    """

    return (
        0.5 * np.tanh((1.0 - z) / EPS)
        + 0.5 * np.tanh(z / EPS)
    )


##############################################################################
# Network construction
##############################################################################

def random_couplings(N, seed=None):
    """
    Gaussian dense matrix

        a_ij ~ N(0, g^2/N)

    with a_ii = 0
    """

    rng = np.random.default_rng(seed)

    A = rng.normal(
        loc=0.0,
        scale=g / np.sqrt(N),
        size=(N, N)
    )

    np.fill_diagonal(A, 0.0)

    return A


##############################################################################
# Dynamics
##############################################################################

def step(r,x, A):

    u = r * x * (1.0 - x + A @ x)

    return H(u)


##############################################################################
# Jacobian
##############################################################################

def jacobian(r,x, A):
    """
    Jacobian of

    x' = H( x * (1 - x + A x) )
    """

    N = len(x)

    gvec = 1.0 - x + A @ x

    u = r * x * gvec

    Hp = Hprime(u)

    #
    # Off-diagonal part
    #
    J = r* x[:, None] * A

    #
    # Diagonal:
    #
    # d/dx_i [r x_i g_i]
    # = r g_i + r x_i(a_ii - 1)
    #
    # and a_ii = 0
    #
    diag = r* gvec - r * x

    idx = np.arange(N)

    J[idx, idx] = diag

    #
    # Chain rule
    #
    J *= Hp[:, None]

    return J


##############################################################################
# Lyapunov spectrum
##############################################################################

def trial(r,
        A,
        x0,
        transient=5000,
        steps=20000,
        qr_interval=1,
        verbose=True):
    """
    Full Lyapunov spectrum using QR decomposition.

    Returns exponents sorted largest -> smallest.
    """

    N = len(x0)

    print(r)

    x = x0.copy()

    #
    # Burn in
    #
    for _ in range(transient):
        x = step(r,x, A)

    #
    # Tangent basis
    #
    Q = np.eye(N)

    sums = np.zeros(N)

    total_time = 0

    for n in range(steps):

        M = np.eye(N)

        for _ in range(qr_interval):

            J = jacobian(r,x, A)

            M = J @ M

            x = step(r,x, A)

        Z = M @ Q

        Q, R = qr(Z)

        diagR = np.abs(np.diag(R))#should it just be the real part of the log below? It's the same actually

        diagR[diagR < 1e-300] = 1e-300

        sums += np.log(diagR)

        total_time += qr_interval

        if verbose and (n + 1) % 100 == 0:
            print(
                f"{n+1}/{steps} QR steps",
                flush=True
            )

    lyap = sums / total_time

    lyap = np.sort(lyap)[::-1]

    csum = np.cumsum(lyap)

    positive = np.where(csum >= 0)[0]

    if len(positive) == 0:
        return 0.0

    k = positive[-1]

    if k == len(lyap) - 1:
        return float(len(lyap))

    return (
        (k + 1)
        + csum[k] / abs(lyap[k + 1])
    )

def trial_star(args):
    return trial(*args)

if __name__ == '__main__':

    #print("effective process")

    #xeff = effproc()

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    


    #Calculate average leading eigenvalue
    DKYvec = np.zeros(numr)


    pool = Pool(processes = 5)

    for itrial in range(0,numtrial):
        print(itrial)

        N = 2048#1024#512#256


        seed = 1234+itrial

        A = random_couplings(
            N=N,
            seed=seed
        )

        rng = np.random.default_rng(seed)

        x0 = rng.uniform(
            low=0.1,
            high=0.9,
            size=N
        )


        myinput = list(range(numr))
        for rtrial in range(0,numr):
            myinput[rtrial]= [rvec[rtrial], A, x0, 1000, 1000, 1 , True]

        myx = pool.starmap(trial, myinput )




        for rtrial in range(0,numr):
            DKYvec[rtrial] = DKYvec[rtrial] + myx[rtrial]/float(numtrial) #myx[gammatrial][0]/float(numtrial)

                    
    pool.close()
    pool.join()

    print("Simulation complete...")
    mystr = "vsr_n2048_numtrial1_sigmasqrt1point5_eps0point05_tmax1000.csv"
    
    np.savetxt("DKY"+mystr, DKYvec, delimiter=",")
    np.savetxt("rvec"+mystr, rvec, delimiter=",")


    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))






