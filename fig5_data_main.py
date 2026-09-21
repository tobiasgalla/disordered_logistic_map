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

n = 4000
c = 1.0
gammax = 0.0
sigma = 0.2

rmin = 1.1
rmax = 4.0
numr = 20
rvec = np.linspace(rmin,rmax,numr, endpoint=True) 

eff0 = 0.001


sigma0 = 0.1
x0 = 0.3

mu = 0.6

omega = -0.8

tmax = 1000.0
tmin = 0.0
tsamp = 0.0

numtrial = 1


#monte-carlo run
def trial(j, r, rit):
    np.random.seed(j*numr + rit+25 )
    print([j, r])

    axx = np.zeros((n,n))
    for i in range(0,n):
        for k in range(0,i):
            rand2 = np.random.uniform(0.0,1.0, 1)[0]
            rand3 = np.random.uniform(0.0,1.0, 1)[0]
            if rand2<0.5:
                axx[i][k] = -mu/n-sigma/np.sqrt(n)
            if rand2>0.5:
                axx[i][k] = -mu/n+sigma/np.sqrt(n)
            if rand3<0.5:
                axx[k][i] = -mu/n-sigma/np.sqrt(n)
            if rand3>0.5:
                axx[k][i] = -mu/n+sigma/np.sqrt(n)




    #Initial condition
    '''x = np.zeros(n)

    for i in range(0, n):
        x[i] = x0

    x[0]= 2.0*x0'''

    x = np.random.uniform(0.1,0.9,n)


    step = 0
    t = 0
    #Numerical integration
    while t <= tmax:
        x = r*x*(1.0 - (1.0-mu)*x + axx.dot(x))*np.heaviside(r*x*(1.0 - (1.0-mu)*x + axx.dot(x)),0.0)*np.heaviside(1.0-r*x*(1.0 - (1.0-mu)*x + axx.dot(x)),0.0) + (1.0-np.heaviside(1.0-r*x*(1.0 - (1.0-mu)*x + axx.dot(x)),0.0)) + 0.00001
        step = step +1
        t = t + 1
        
    xstar = np.copy(x)


    nstar = np.copy(n)
    for k in range(0,n):
        if x[k] <eff0 or x[k]>1.0-eff0:
            axx = np.delete(axx, k-(n-nstar), 0)
            axx = np.delete(axx, k-(n-nstar), 1)
            xstar = np.delete(xstar, k-(n-nstar), 0)
            #xstar = np.delete(xstar, k-(n-nstar), 1)
            nstar = nstar - 1
	

    
    id = np.identity(nstar)
        
    jac = np.zeros((nstar,nstar))
    xdiag = np.zeros((nstar,nstar))
    zmat = axx+mu/float(n)

    modav = 0.0 
    resterm = 0.0 
    resterm2 = 0.0 

    for i in range(0,nstar):
        xdiag[i][i] = r*xstar[i]/(omega -1.0+ r*(1.0-mu)*xstar[i])
        for j in range(0, nstar):
            jac[i][j] = r*xstar[i]*( axx[i][j] -id[i][j]*(1.0-mu) ) + id[i][j]
            modav = modav + xstar[i]*axx[i][j]/nstar
            '''resterm = resterm + r**2*xstar[i]*(axx[i][j]+mu/float(n))*xstar[j]/nstar/(omega -1.0+ r*(1.0-mu)*xstar[i])/(omega-1.0 + r*(1.0-mu)*xstar[j])
            for k in range(0,nstar):
                resterm2 = resterm2 + r**3*xstar[i]*xstar[j]*xstar[k]*(axx[i][k]+mu/float(n))*(axx[k][j]+mu/float(n))/nstar/(omega -1.0+ r*(1.0-mu)*xstar[i])/(omega-1.0 + r*(1.0-mu)*xstar[j])/(omega-1.0 + r*(1.0-mu)*xstar[k])'''
            

    
    evaluesj, evectorsj = la.eig(jac)
    lambdamin = min(np.real(evaluesj))
    
    mul = np.sum(np.sum(axx))/nstar 
    resterm = sum(sum(xdiag.dot(zmat.dot(xdiag))))/float(nstar)
    resterm2 = sum(sum(xdiag.dot(zmat.dot(xdiag.dot(zmat.dot(xdiag))))))/float(nstar)
    

    '''axxred = axx - mul/nstar
    axxdash = axx - mu/float(n)
    gammal = np.trace(axxred.dot(axxred))/nstar
    sl = np.sum(axxred.dot(axxred))/nstar - gammal
    s3l = np.sum((axxred.dot(axxred)).dot(axxred))/nstar
    s3ldash = np.sum((axxdash.dot(axxdash)).dot(axxdash))/nstar
    s4l = np.sum(((axxred.dot(axxred)).dot(axxred)).dot(axxred))/nstar 
    s4ldash =  np.sum(((axxdash.dot(axxdash)).dot(axxdash)).dot(axxdash))/nstar'''
		

    return lambdamin,  mul,nstar, modav, resterm, resterm2;#, mul, gammal, sl, s3l, s4l, s4ldash, s3ldash;

def trial_star(args):
    return trial(*args)

if __name__ == '__main__':

    #print("effective process")

    #xeff = effproc()

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    


    #Calculate average leading eigenvalue
    lambdavec = np.zeros(numr)
    mulvec = np.zeros(numr)
    phivec = np.zeros(numr)
    modavvec = np.zeros(numr)
    restermvec = np.zeros(numr)
    resterm2vec = np.zeros(numr)
    '''slvec = np.zeros(numgamma)
    s3lvec = np.zeros(numgamma)
    s4lvec = np.zeros(numgamma)
    s4ldashvec = np.zeros(numgamma)
    s3ldashvec = np.zeros(numgamma)'''
    
    pool = Pool(processes = 5)

    for itrial in range(0,numtrial):
        print(itrial)
        myinput = list(range(numr))
        for rtrial in range(0,numr):
            myinput[rtrial]= [itrial, rvec[rtrial], rtrial]

        myx = pool.starmap(trial, myinput )




        for rtrial in range(0,numr):
            lambdavec[rtrial] = lambdavec[rtrial] + myx[rtrial][0]/float(numtrial) #myx[gammatrial][0]/float(numtrial)
            mulvec[rtrial] = mulvec[rtrial] + myx[rtrial][1]/float(numtrial)
            phivec[rtrial] = phivec[rtrial] + myx[rtrial][2]/float(numtrial*n)
            modavvec[rtrial] = modavvec[rtrial] + myx[rtrial][3]/float(numtrial)
            restermvec[rtrial] = restermvec[rtrial] + myx[rtrial][4]/float(numtrial)
            resterm2vec[rtrial] = resterm2vec[rtrial] + myx[rtrial][5]/float(numtrial)
            '''gammalvec[gammatrial] = gammalvec[gammatrial] + myx[gammatrial][2]/float(numtrial)
            slvec[gammatrial] = slvec[gammatrial] + myx[gammatrial][3]/float(numtrial)
            s3lvec[gammatrial] = s3lvec[gammatrial] + myx[gammatrial][4]/float(numtrial)
            s4lvec[gammatrial] = s4lvec[gammatrial] + myx[gammatrial][5]/float(numtrial)
            s4ldashvec[gammatrial] = s4ldashvec[gammatrial] + myx[gammatrial][6]/float(numtrial)
            s3ldashvec[gammatrial] = s3ldashvec[gammatrial] + myx[gammatrial][7]/float(numtrial)'''
                    
    pool.close()
    pool.join()

    print("Simulation complete...")
    mystr = "vsr_n4000_numtrial1_sigma0point2_mu0point6_effzero0point001_tmax1000_rmax4.csv"
    
    np.savetxt("lambda"+mystr, lambdavec, delimiter=",")
    np.savetxt("mul"+mystr, mulvec, delimiter=",")
    np.savetxt("phi"+mystr, phivec, delimiter=",")
    np.savetxt("rvec"+mystr, rvec, delimiter=",")
    np.savetxt("modav"+mystr, modavvec, delimiter=",")
    np.savetxt("resterm"+mystr, restermvec, delimiter=",")
    np.savetxt("resterm2"+mystr, resterm2vec, delimiter=",")

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))






