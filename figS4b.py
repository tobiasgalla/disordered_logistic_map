from os import environ
environ["OMP_NUM_THREADS"] = '1'
environ["OPENBLAS_NUM_THREADS"] = '1'
environ["MKL_NUM_THREADS"] = '1'
environ["VECLIB_MAXIMUM_THREADS"] = '1'
environ["NUMEXPR_NUM_THREADS"] = '1'
from time import gmtime, strftime
import matplotlib.pyplot as plt
import numpy as np
import math
import multiprocessing
from multiprocessing import Pool
import csv
from numpy import linalg as la
import networkx as nx


n = 4000
sigma = 0.59
mu = 0.0
gammax = 0.0
r = 1.8

a = 0.2

eff0 = 0.001


tmax = 10000


numtrial = 10
kbound = 1.0

numsample = 4000

sigmat = 0.001

omvec = np.linspace(0.0, 2.0*np.pi*(numsample-1.0)/numsample, numsample, endpoint = True)

xmin = 0
xmax = 1
numbins = 51
binedges = np.linspace(xmin, xmax, numbins+1, endpoint=True)
bins = np.linspace(xmin+float((xmax-xmin)/(numbins-1))*0.5, xmax-float((xmax-xmin)/(numbins-1))*0.5, numbins, endpoint=True)	

#monte-carlo run
def trial(tr):

    print(tr)
    trajectory = np.ones((numsample, n))
    transform = (1.0j)*np.ones((numsample, n))
    axx = np.zeros((n,n))
    powerspectrum = np.zeros(numsample)
    xhist = np.zeros(numbins)
    sampvec = np.zeros(numsample)
    sampvec2 = np.zeros(numsample)
                
    #construct random interaction matrix
    for i in range(0,n):
        for k in range(0,i):
            rand2 = np.random.uniform(0.0,1.0, 1)[0]
            rand3 = np.random.uniform(0.0,1.0, 1)[0]
            if rand2<0.5:
                axx[i][k] = -sigma/np.sqrt(n)
            if rand2>0.5:
                axx[i][k] = sigma/np.sqrt(n)
            if rand3<0.5:
                axx[k][i] = -sigma/np.sqrt(n)
            if rand3>0.5:
                axx[k][i] = sigma/np.sqrt(n)
        '''axx[i][k] = np.random.normal(0.0, sigma/np.sqrt(n), 1)[0]
            axx[k][i] = np.random.normal(0.0, sigma/np.sqrt(n), 1)[0]'''


        
    x = np.ones(n)*(1.0-1.0/r)
    xdet = np.ones(n)*(1.0-1.0/r)
    step = 0
    t = 0
    #Numerical integration
    while t <= tmax:

        xi = np.random.normal(0.0, sigmat, n)
        x = r*x*(1.0 - x + axx.dot(x) + xi)*np.heaviside(r*x*(1.0 - x + axx.dot(x)+ xi),0.0)*np.heaviside(kbound-r*x*(1.0 - x + axx.dot(x)+ xi),0.0) + kbound*(1.0-np.heaviside(kbound-r*x*(1.0 - x + axx.dot(x)+ xi),0.0)) + 0.0001
        xdet = r*xdet*(1.0 - xdet + axx.dot(xdet))*np.heaviside(r*xdet*(1.0 - xdet + axx.dot(xdet)),0.0)*np.heaviside(kbound-r*xdet*(1.0 - xdet + axx.dot(xdet)),0.0) + kbound*(1.0-np.heaviside(kbound-r*xdet*(1.0 - xdet + axx.dot(xdet)),0.0)) + 0.0001


        step = step +1
        t = t + 1
    flag = 0
    ind = 0
    while flag==0:
        if(x[ind]>eff0 and x[ind]<1.0-eff0):
            flag = 1
        ind = ind +1

    ind = ind-1

    xdet2 = r*xdet*(1.0 - xdet + axx.dot(xdet))*np.heaviside(r*xdet*(1.0 - xdet + axx.dot(xdet)),0.0)*np.heaviside(kbound-r*xdet*(1.0 - xdet + axx.dot(xdet)),0.0) + kbound*(1.0-np.heaviside(kbound-r*xdet*(1.0 - xdet + axx.dot(xdet)),0.0)) + 0.0001

    avdet = (xdet+xdet2)/2.0
    step = 0
    while t <= tmax + numsample:

        xi = np.random.normal(0.0, sigmat, n)

        x = r*x*(1.0 - x + axx.dot(x)+ xi)*np.heaviside(r*x*(1.0 - x + axx.dot(x)+ xi),0.0)*np.heaviside(kbound-r*x*(1.0 - x + axx.dot(x)+ xi),0.0) + kbound*(1.0-np.heaviside(kbound-r*x*(1.0 - x + axx.dot(x)+ xi),0.0)) + 0.0001
        xdet = r*xdet*(1.0 - xdet + axx.dot(xdet))*np.heaviside(r*xdet*(1.0 - xdet + axx.dot(xdet)),0.0)*np.heaviside(kbound-r*xdet*(1.0 - xdet + axx.dot(xdet)),0.0) + kbound*(1.0-np.heaviside(kbound-r*xdet*(1.0 - xdet + axx.dot(xdet)),0.0)) + 0.0001



        for j in range(0, n):
            trajectory[step][j] = np.copy(x[j])#np.exp((0.0+1.0j)*omvec[i]*(step))*(x[j]-avdet[j])



        sampvec[step] = np.copy(x[ind])
        sampvec2[step] = np.copy(xdet[ind])
        step = step +1
        t = t + 1

    avvec = np.zeros(n)
    for j in range(0,n):
        avvec[j] = sum((trajectory.T)[j])/float(numsample)

    for st in range(0,numsample):
        for om in range(0, numsample):
            for j in range(0,n):
                transform[om][j] = transform[om][j] + np.exp((0.0+1.0j)*omvec[om]*(st))*(trajectory[st][j]-avvec[j])


    for i in range(0, numsample):
        for j in range(0, n):
            if x[j]>eff0 and x[j]<1.0-eff0:
                powerspectrum[i] = powerspectrum[i] + abs(transform[i][j])**2/float(n)

	
    xhist = np.histogram(xdet, binedges)[0]/float(n)*numbins/(xmax-xmin)

    
    return xhist, powerspectrum;

def trial_star(args):
    return trial(*args)

if __name__ == '__main__':


    print([numtrial, n])
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    
    numproc = 5
    pool = Pool(processes = numproc)
    

    myxhist = np.zeros(numbins)
    mypowerspectrum = np.zeros(numsample)

    avimg = 0.0

    



    #myhistimg, myhistreg, myhistfakeimg, myhistfakereg, myhistimself, myhistreself, myhistimoff, myhistreoff, myhistfakeimoff, myhistfakereoff, avimg, avreg, avimself, avreself, avimoff, avreoff;

    
    for itrial in range(0,int(numtrial/numproc)):
        myinput = list(range(numproc))
        for it in range(0, numproc):
            myinput[it] = [numproc*itrial + it]

        print(myinput)
        myx = pool.starmap(trial, myinput)


        for it in range(0,numproc):   
            myxhist = myxhist + np.copy(myx[it][0])/float(numtrial)
            mypowerspectrum = mypowerspectrum + np.copy(myx[it][1])/float(numtrial)


            
                
                    
    pool.close()
    pool.join()
    

    print("Simulation complete...")
    
    

    mystr = "_n4000_numtrial10_numsample4000_r1point8_sigma0point59.csv"
    np.savetxt("xhist"+mystr, myxhist, delimiter=",")
    np.savetxt("powerspectrum"+mystr, mypowerspectrum, delimiter=",")

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    










