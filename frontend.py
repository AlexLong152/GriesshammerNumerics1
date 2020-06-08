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
    
def probDist():
    """
    This is the thing we're actually after, the probability of finding k
    and a function of k
    
    dx: The distance between samples
    dxHistrogram: the step between in the histogram
    windowSize: the size of the bin in the histogram
    """
    dx=0.001
    dxHistrogram=3*dx
    windowSize=dxHistrogram*30
    
    matplotlib.rcParams['text.usetex'] = True
    fig, (ax0,ax1, ax2) = plt.subplots(1, 3,
                                     gridspec_kw={'width_ratios': [0.5,3, 1]})
    
    
    k0s=np.arange(np.pi/2+0.01,(3*np.pi/2)-0.01,step=dx)
    ys=np.zeros(len(k0s))
    for i in range(len(k0s)):
        ys[i]=be.findRoot(k0s[i])
    
   
    bins,yCounts=be.makeHist(ys,dxHistrogram,0,np.max(ys)+dxHistrogram,windowSize)
    total=np.sum(yCounts)
    prob=yCounts/total
    titleString="$\kappa $ probability vs measured $\kappa$ "
    titleString+="for $\kappa_0$ randomly distributed"
    fig.suptitle(titleString,fontsize=16)
    
    ax1.plot(bins, prob)
    #ax1.scatter(bins, prob,s=1)
    ax1.set_xlabel("Measured $\kappa$")
    #ax1.set_ylabel("Probability\n$P(\kappa)$",rotation=0)
    ax0.axis('off')
    ax0.text(-1,0.5,"Probability\n    $\;\;\;P(\kappa)$")
    #ax0.text(-1,0.5,"Counts")
    ax1.xaxis.labelpad=0
    ax2.axis('off')
    
    dataString="\\underline{Values used} \n$r_0$="+str(np.round(r0,4))
    dataString+="\n$\kappa_0   \in ["
    dataString+=str(np.round(np.min(k0s),4))+","+str(np.round(np.max(k0s),4))+"]$\n"
    dataString+="$dx$="+str(np.round(dx,4))+"\n"
    dataString+="dxHistrogram="+str(np.round(dxHistrogram,4))+"\n"
    dataString+="windowSize="+str(np.round(windowSize,4))
    #print(dataString)
    ax2.text(0,0.75,dataString,fontsize=13)
    fig.show()
    
    matplotlib.rcParams['text.usetex'] = False
    
def kRootVSk0(showPlot=True):
    """
    Plot of the roots of the function  k vs k0
    """
    matplotlib.rcParams['text.usetex'] = True
    dk0=0.01
    k0Min=1.9
    k0Max=15
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

def main():
    probDist()
    #threeDplot(be.errorFunc)
    """
    k0s,ks=kRootVSk0(False)
    errors=np.zeros(len(ks))
    for i in range(len(k0s)):
        errors[i]=be.errorFunc(ks[i],k0s[i])
    plt.scatter(ks,errors,s=0.5)
    """
if __name__ =="__main__":
    main()