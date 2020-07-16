# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:11:25 2020

@author: Alex
"""
import numpy as np
from matplotlib import pyplot as plt
import frontend as fe
import backend as be
import matplotlib
matplotlib.rcParams['text.usetex'] = True

def Yapprox(n,x):
    """ The approximation
    """
    val1=1+np.pi*x-2*n*x*np.pi+2*(x**2)
    return -1+np.sqrt(val1)

def approxSolu(k0):
    """
    Returns the approximate solution for the lowest energy for a given k0
    """
    def onevalue(k0):
        """takes a scaler value and returns the respective root"""
        n=int(np.floor((k0/np.pi)+0.5))
        
        nn=0
        while (nn-0.5)*np.pi<=k0:
            nn+=1
        
        assert(nn-1)==n
        fac=(-2*np.pi*n*k0) + 2*(k0**2)+(np.pi*k0) +1
        assert fac>0
        return np.float64(-1+np.sqrt(fac))
    
    if isinstance(k0,(list,tuple)):
        k0=np.fromiter(k0,np.float64)
        
    if isinstance(k0,np.ndarray):
        k0=k0.flatten()
        k=np.zeros(len(k0),np.float64)
        for i in range(len(k)):
            k[i]=onevalue(k0[i])
            
    elif isinstance(k0,(float,int)):
        k=onevalue(k0)
    else:
        typeString=str(type(k0))
        raise TypeError("k0 is of non-supported type "+typeString)
    return k

def Py(n,y):
    """The probabilities based off the approximation
    """
    maxval=Yapprox(n,(n+0.5)*np.pi)
    minval=Yapprox(n,(n-0.5)*np.pi)
    
    def oneval(y):
        #Returns just one value of y for scaler y
        if y<minval or y>maxval:
            return 0
        else:
            numerator=2*(1+y)
            denom=np.pi* np.sqrt((np.pi**2) *((1 - 2*n)**2) + 8* y* (2 + y))
            return numerator/denom
        
    if isinstance(y,np.ndarray):
        out= np.zeros(np.shape(y))
        for i in range(len(out)):
            out[i]=oneval(y[i])
    else:
        out=oneval(y)
    return out

def ProbM(M,ys):
    """ The probabilities attained from summing the first M solutions.
    
    Parameters
    ------------
    M: int
        How many solution numbers to include, must be equal to or greater than
        one
    ys: ndarray
        The values at which you wish to know the probabilities
    
    Returns
    out: ndarray
        the probabilities at each y value
    """
    ys=ys.flatten()
    out= np.zeros(len(ys))
    for i in range(1,M+1):
        out+=Py(i,ys)
    
    out=out/M
    return out
    
def HistogramVsApprox(n):
    """Finds the probabilities via the histogram method and plots them vs 
    the probabilities from the analytical approximation
    
    Paramters
    --------------
    n: int
        How many solutions you want to go up to.
    """
    t1=time()
   
    dk0=0.001
    histdx=0.015
    windowSize=0.05#histogram window size
    
    k0Max=np.pi*(n+0.5)
    k0sNumer,ksNumer=fe.kRootVSk0(showPlot=False,k0Max=k0Max,dk0=dk0)

    values,counts=be.makeHist(ksNumer,dx=histdx,windowSize=windowSize)
    
    integral=histdx*np.sum(counts)
    prob=counts/integral
    
    ys=np.arange(np.min(ksNumer),np.max(ksNumer),step=dk0)#For the approximate solution
    approxProb=ProbM(n,ys)
    
    k0sApprox=np.arange(np.pi/2+0.01,k0Max,dk0)
    ksApprox=approxSolu(k0sApprox)
    
    matplotlib.rcParams['text.usetex'] = True
    fig, (ax0,ax1, ax2) = plt.subplots(1, 3,
                                     gridspec_kw={'width_ratios': [0.5,3, 1]})
    ax0.axis("off")
    ax0.text(-0.5,0.6,"Probability \n $P(\kappa)$",fontsize=10)
    
    
    ax1.scatter(values,prob,s=0.5,c="b",label="Numerical Probabiliy from histogram")
    ax1.set_xlabel("Observed least $\kappa$")
    ax1.plot(ys,approxProb,c="r",label="Approximate Solution Probability")
    
    h1=np.max(prob)#Scale the vertical axis to make it match up
    h2=np.max(approxProb)
    height=np.max([h1,h2])
    k0sApproxPlot=height*k0sApprox/np.max(k0sApprox)
    k0sNumerPlot=height*k0sNumer/np.max(k0sNumer)
    
    ax1.plot(ksNumer,k0sNumerPlot,alpha=0.3,c="blue",label="Numerical Solution")
    ax1.plot(ksApprox,k0sApproxPlot,alpha=0.3,c="red",label="Approximate Solution")
    ax1.legend()


    ax2.axis("off")
    noteString="\\underline{Values used} \n"
    noteString+="k0Max="+str(np.round(k0Max,4))+"\n"
    noteString+="dk0="+str(np.round(dk0,5))+"\n"
    noteString+="histogram dx="+str(histdx)+"\n"
    noteString+="histogram window size="+str(windowSize)
    ax2.text(-0.1,0.75,noteString,fontsize=10)
    
    _=fig.suptitle("Probability of the least root of $\kappa$ being a certain value",
                     fontsize=15)
    print(np.round(time()-t1,4))
def main():
    
    HistogramVsApprox(5)
    """
    n=3
    for n in range(1,5):
        ys=np.arange(Yapprox(n,np.pi*(n-0.5)),Yapprox(n,np.pi*(n+0.5)),0.05)
        probs=Py(n,ys)
        plt.scatter(ys,probs,s=0.5)
    plt.show()
    """
if __name__=="__main__":
    main()