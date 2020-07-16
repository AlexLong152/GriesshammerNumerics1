# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import backend as be
#from backend import r0 as r0
import frontend as fe
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
k0maxValue=(17*np.pi/2)

def approxSolu(k0):
    """
    Returns the approximate solution for the lowest energy for a given k0
    """
    def onevalue(k0):
        """takes a scaler value and returns the respective root"""
        n=int(np.floor((k0/np.pi)+0.5))
        
        nn=0
        while (nn-0.5)*np.pi<k0:
            nn+=1
        
        assert(nn-1)==n
        fac=(-2*np.pi*n*k0) + 2*(k0**2)+(np.pi*k0) +1
        if fac<0:
            raise ValueError("Negative value in sqrt")
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

def compareSolutions():
    """Compares the True solution to the approximate solution
    """
    k0s,ks=fe.kRootVSk0(showPlot=False,k0Max=(17*np.pi/2))
    approxKs=approxSolu(k0s)
    plt.plot(k0s,ks,label="True Solution")
    plt.plot(k0s,approxKs,label="Approximate Solution")
    plt.title("Approximate Solutions vs True Solutions for least $\kappa$")
    plt.xlabel("$\kappa_0$")
    plt.ylabel("$\kappa$")
    plt.legend()
    plt.show()
    
def probViaHistogram():
    """Finds the probabilities via the histogram method
    """
    k0Max=(17*np.pi/2)
    dk0=0.005
    k0s,ks=fe.kRootVSk0(showPlot=False,k0Max=k0Max,dk0=dk0)
    
    dx=0.04
    windowSize=0.2
    values,counts=be.makeHist(ks,dx=dx,windowSize=windowSize)
    
    integral=dx*np.sum(counts)
    prob=counts/integral
    fig, (ax0,ax1, ax2) = plt.subplots(1, 3,
                                     gridspec_kw={'width_ratios': [0.5,3, 1]})
    ax0.axis("off")
    ax0.text(-0.5,0.6,"Probability \n $P(\kappa)$",fontsize=10)
    
    
    ax1.scatter(values,prob,s=0.5)
    ax1.set_xlabel("Observed least $\kappa$")
    
    ax2.axis("off")
    noteString="\\underline{Values used} \n"
    noteString+="k0Max="+str(np.round(k0Max,4))+"\n"
    noteString+="dk0="+str(np.round(dk0,5))+"\n"
    noteString+="histogram dx="+str(dx)+"\n"
    noteString+="histogram window size="+str(windowSize)
    ax2.text(-0.1,0.75,noteString,fontsize=10)
    fig.suptitle("Probability of the least root of $\kappa$ being a certain value",
                     fontsize=15)
    
def main():
    compareSolutions()
    

if __name__=="__main__":
    main()
