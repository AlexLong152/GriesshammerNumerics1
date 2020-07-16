# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:25:19 2020
@author: alexl
"""

import backend as be
from backend import r0
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

def makeManyplots():
    for k in np.arange(5,8,1):  
        be.plot(k0=k)
        
def testRootFinding():
    def f(x):
        """
        x**2 + 2x -8
        roots at 2 and -4
        """
        return (x-2)*(x+4)
    def fPrime(x):
        return 2*x+2
    # __init__(self,f,x0,bounds=[None,None],fPrime=None,maxError=0.001):
    x0=None
    bounds=[0,None]
    rootInit=be.rootFind(f,x0,bounds=bounds)
    #x=rootInit.newtonRoot()
    x=rootInit.goldenSection()
    print("testRootFinding: [0,None], x0=None , Analytic Root is 2")
    print("testRootFinding: Found root was x=",np.round(x,decimals=5))
    
    x0=None
    bounds=[None,1]
    rootInit=be.rootFind(f,x0,bounds=bounds)
    #x=rootInit.newtonRoot()
    x=rootInit.goldenSection()
    print("testRootFinding: [None,1], x0=None , Analytic Root is -4")
    print("testRootFinding: Found root was x=",np.round(x,decimals=5))
    
def testRootsOfErrorFunc():
    """
    Tests to make sure we are actually finding the roots of the error function
    """
    k0s,ks=kRootVSk0(False)
    errors=np.zeros(len(ks))
    for i in range(len(k0s)):
        errors[i]=be.errorFunc(ks[i],k0s[i])
    plt.scatter(ks,errors,s=0.5)
    plt.show()
    
def threeDplot(func):
    """
    Makes a 3d plot of the error function, clipping the function above and 
    below 10
    """
    matplotlib.rcParams['text.usetex'] = True
    #func=be.errorFunc
    fig = plt.figure(figsize=(9.33,7))
    ks=np.arange(2,10*np.pi,0.2)
    k0s = np.arange(2,10*np.pi,0.2)
    
    KS,K0S = np.meshgrid(ks, k0s)
    Z=np.zeros(np.shape(KS))
    for i in range(len(k0s)):
        for j in range(len(ks)):
            Z[i,j]=func(ks[j],k0s[i])
            
    locs=np.where(np.isinf(Z))
    Z[locs]=0
    locs=np.where(Z>10)
    Z[locs]=10
    locs=np.where(Z<-10)
    Z[locs]=-10
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(KS, K0S, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    
   
    xText="$\kappa$"
    yText="$\kappa_0$"
    zText="$f(\kappa,\kappa_0)$"
    ax.set_xlabel(xText,fontsize=14)
    ax.set_ylabel(yText,fontsize=14)
    
    #ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(zText,fontsize=14)
    ax.set_title("Surface")
    
    ax.view_init(elev=27,azim=-176)
    fig.show()
    
    #matplotlib.rcParams['text.usetex'] = False
    #if you set this variable to zero at the end it spazes out for this one
    
def probDist(dx=0.005,dxHistrogramFactor=3,windowSizeFactor=240,numRoots=3,maxK0=50):
    """
    This is the thing we're actually after, the probability of finding k
    and a function of k
    
    Things are left in terms of a factor times dx because this way you don't
    have to adjust as many values if you, for example just change dx
    
    Paramters
    ----------
    dx: float
        The step in the root finding
    dxHistrogramFactor: float
        dxHistogram is the step in the histogram, and we have
        dxHistrogram=dxHistrogramFactor*dx
    windowSizeFactor: 
        the size of the bin in the histogram is windowSize and we have
         windowSize=dx*windowSizeFactor
    
    max
    """
    matplotlib.rcParams['text.usetex'] = True
    
    dxHistrogram=dxHistrogramFactor*dx
    windowSize=dx*windowSizeFactor

    n=numRoots-1
    tmp=(2*n+1)*np.pi/2
    if maxK0<tmp:
        print("probDist: maxK0 too low for numRoots, value was increased")
        maxK0=tmp
    fig, (ax0,ax1, ax2) = plt.subplots(1, 3,
                                     gridspec_kw={'width_ratios': [0.5,3, 1]})
    for rootNumber in range(numRoots):
        k0s,ys=be.sameRoot(maxK0,dx,rootNumber=rootNumber)
        locs=np.where(~np.isinf(ys))#where ys is not infinite
        ys=ys[locs]
        #bins,yCounts=be.makeHist(ys,dxHistrogram,np.min(ys)+dxHistrogram+windowSize,
        #                         np.max(ys)-dxHistrogram-windowSize,windowSize)
        bins,yCounts=be.makeHist(ys,dxHistrogram, windowSize=windowSize)
                                
        integral=dxHistrogram*np.sum(yCounts)
        prob=yCounts/integral
        label="Root Number "+str(rootNumber+1)
        ax1.scatter(bins, prob,label=label,s=0.5)
    ax1.legend()
    
    
    titleString="$\kappa $ probability vs measured $\kappa$ "
    titleString+="for $\kappa_0$ randomly distributed"
    fig.suptitle(titleString,fontsize=16)
        
   
    #ax1.scatter(bins, prob,s=1)
    ax1.set_xlabel("Measured $\kappa$")
    #ax1.set_ylabel("Probability\n$P(\kappa)$",rotation=0)
    ax0.axis('off')
    ax0.text(-1,0.5,"Relative\n Probability \n    $\;\;\;P(\kappa)$")
    #ax0.text(-1,0.5,"Counts")
    ax1.xaxis.labelpad=0
    ax2.axis('off')
    
    dataString="\\underline{Values used} \n$r_0$="+str(np.round(r0,4))
    dataString+="\n$\kappa_0   \in ["
    dataString+=str(np.round(np.pi/2 +0.0001,4))+","+str(np.round(maxK0,4))+"]$\n"
    dataString+="$dx$="+str(np.round(dx,4))+"\n"
    dataString+="dxHistrogram="+str(np.round(dxHistrogram,4))+"\n"
    dataString+="windowSize="+str(np.round(windowSize,4))
    #print(dataString)
    ax2.text(0,0.75,dataString,fontsize=13)
    fig.show()
    #fig.legend()
    matplotlib.rcParams['text.usetex'] = False
    
def plotHist1Over(dx=0.005,dxHistrogramFactor=3,windowSizeFactor=3,numRoots=3,maxK0=50):
    matplotlib.rcParams['text.usetex'] = True
    
    dxHistrogram=dxHistrogramFactor*dx
    windowSize=dx*windowSizeFactor

    n=numRoots-1
    tmp=(2*n+1)*np.pi/2
    if maxK0<tmp:
        print("probDist: maxK0 too low for numRoots, value was increased")
        maxK0=tmp
    fig, (ax0,ax1, ax2) = plt.subplots(1, 3,
                                     gridspec_kw={'width_ratios': [0.5,3, 1]})
    for rootNumber in range(numRoots):
        k0s,ys=be.sameRoot(maxK0,dx,rootNumber=rootNumber)
        locs=np.where(~np.isinf(ys))#where ys is not infinite
        ys=ys[locs]
        ys=1/ys
        #bins,yCounts=be.makeHist(ys,dxHistrogram,np.min(ys)+dxHistrogram+windowSize,
        #                         np.max(ys)-dxHistrogram-windowSize,windowSize)
        bins,yCounts=be.makeHist(ys,dxHistrogram, windowSize=windowSize)
                                
        integral=dxHistrogram*np.sum(yCounts)
        prob=yCounts/integral
        label="Root Number "+str(rootNumber+1)
        ax1.scatter(bins, prob,label=label,s=0.5)
    ax1.legend()
    
    
    titleString="$\kappa $ probability vs measured $\kappa$ "
    titleString+="for $\kappa_0$ randomly distributed"
    fig.suptitle(titleString,fontsize=16)
        
   
    #ax1.scatter(bins, prob,s=1)
    ax1.set_xlabel("Measured $\kappa$")
    #ax1.set_ylabel("Probability\n$P(\kappa)$",rotation=0)
    ax0.axis('off')
    ax0.text(-1,0.5,"Relative\n Probability \n    $\;\;\;P(\kappa)$")
    #ax0.text(-1,0.5,"Counts")
    ax1.xaxis.labelpad=0
    ax2.axis('off')
    
    dataString="\\underline{Values used} \n$r_0$="+str(np.round(r0,4))
    dataString+="\n$\kappa_0   \in ["
    dataString+=str(np.round(np.pi/2 +0.0001,4))+","+str(np.round(maxK0,4))+"]$\n"
    dataString+="$dx$="+str(np.round(dx,4))+"\n"
    dataString+="dxHistrogram="+str(np.round(dxHistrogram,4))+"\n"
    dataString+="windowSize="+str(np.round(windowSize,4))
    #print(dataString)
    ax2.text(0,0.75,dataString,fontsize=13)
    fig.show()
    #fig.legend()
    matplotlib.rcParams['text.usetex'] = False
    
def kRootVSk0(showPlot=True,k0Min=np.pi/2+0.001,k0Max=15,dk0=0.01):
    """
    Finds the first root greater than zero as a function of \kappa_0
    
    this function is unimportant and is of little interest
    
    Parameters
    ------------
    showPlot: boolean, optional
        True if you want the function to be plotted
    
    Returns
    ---------
    k0s: 1-d ndarray
        The \kappa_0 values used
    ks: 1-d ndarray
        the least value of \kappa such that the error function returns 
        zero.
    """
    matplotlib.rcParams['text.usetex'] = True
    """
    dk0=0.01
    k0Min=np.pi/2+0.001
    k0Max=15
    """
    k0s=np.arange(k0Min,k0Max,dk0)
    
  
    ks=np.zeros(len(k0s))
    for i in range(len(ks)):
        ks[i]=be.findRoot(k0s[i])
    if showPlot:
        fig, (ax0,ax1, ax2) = plt.subplots(1, 3,gridspec_kw={'width_ratios': [0.5,3, 1]})
        ax2.axis("off")
        
        dataString="\\underline{Values used} \n$r_0$="+str(np.round(r0,4))+"\n"
        dataString+="$d \kappa_0$="+str(np.round(dk0,4))+"\n"
        dataString+="note in code \n $d \kappa_0$ is dk0"
        ax2.text(0,0.7,dataString,fontsize=14)
        ax1.plot(k0s,ks)
        ax1.set_xlabel("$\kappa_0$",fontsize=16)
        ax0.axis('off')
        ax0.text(-1,0.5,"$\kappa$ such that \n $f(\kappa,\kappa_0)=0$",fontsize=14)
        fig.suptitle("$\kappa_0$ vs the least root of $f(\kappa,\kappa_0)$",
                     fontsize=19)
        fig.show()
    
    matplotlib.rcParams['text.usetex'] = False
    return k0s,ks

def continuousRoots(numRoots=5,maxval=45,step=0.05,rootStart=0):
    """
    Plots \kappa_\ell (\kappa_0) for \ell from rootStart to numRoots
    
    Parameters
    --------------
    numRoots: int, optional
        the max root number \ell
    maxval: float, optional,
        the max value of \kappa_0
    step: float, optional,
        the step in \kappa_0
    rootStart: int, optional
        Which value of \ell you wish to start at, for increased computational
        speed if you only care about higher value of \ell. rootStart=0 includes
        all the roots
        
    Returns
    ------------
    k0s: 1-d ndarray of floats
        the \kappa_0 values used
    root: 2-d ndarray of floats
        the function \kappa_\ell(\kappa_0) the element
        root[i][j] refers to root \ell=i+1 and the \kappa_0 corrisponding to 
        it can be found at k0s[j]
    """
    minval=np.pi/2 +0.001
    n=numRoots-1
    tmp=(2*n+1)*np.pi/2
    if maxval<tmp:
        print("continuousRoots: maxVal too low for numRoots, value was increased")
        maxval=tmp
        
    rootStart=0
    root=np.zeros((numRoots,len(np.arange(minval,maxval,step))))
    for i in range(rootStart,numRoots):
        k0s,root[i]=be.sameRoot(maxval,step,rootNumber=i+1)
        #k0s are always the same so it doesnt matter
    
    """
    tmp=deepcopy(root)
    tmp[np.isinf(root)]=10*maxval
    diffs=abs(tmp[:,1:]-tmp[:,:-1])
    condition=(diffs>0.75) & (diffs<maxval)
    locs=np.where(condition)
    assert len(locs[0])==0
    
    For values of numRoots up to about 6, we can use the above commented out 
    code to test if the roots finding algorithm didn't skip values,
    however after this point the derivative at the
    begining is very high, so it no longer works. This isn't a big deal 
    becuase one can immedietly tell by inspection of the graph if we 
    accidently jumped between roots
    """
    matplotlib.rcParams['text.usetex'] = True
    plt.plot(k0s,k0s,label="$\kappa = \kappa_0$")
    for i in range(rootStart,numRoots):
        plt.plot(k0s,root[i],label="Root number "+str(i+1))
    plt.xlabel("$\kappa_0$")
    plt.ylabel("The root $\kappa$")
    plt.title("$\kappa_0$ vs the root $\kappa$")
    plt.legend()
    plt.show()
    
    return k0s, root

def main():
    #probDist(dx=0.002,windowSizeFactor=200,numRoots=8)
    #plotHist1Over()
    #threeDplot(be.errorFunc)
    continuousRoots(10)
    
if __name__ =="__main__":
    main()