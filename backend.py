# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:25:05 2020

@author: alexl
"""
import numpy as np
from mpmath import cot
from matplotlib import pyplot as plt
from copy import deepcopy
from numba import jit
import math
r0=1

def plot(k0=3*np.pi):
    ks=np.arange(0,k0-0.01,step=0.05)
    ys=np.zeros(len(ks))
    for i in range(len(ks)):
        ys[i]=errorFunc(ks[i],k0)
    labelString="Error for k0="+str(np.round(k0,3))
    plt.scatter(ks,ys,label=labelString,s=0.5)
    plt.legend()
    plt.title("Error")
    plt.ylim(-10,10)
    plt.hlines(0,-2,k0+2,colors="black")
    plt.xlim(0,k0)
    
class rootFind():
    """
    Newton's Method implimentation for root finding
    
    Note for small k0 there is no root
    """
    def __init__(self,f,x0,bounds=[None,None],fPrime=None,maxError=0.00001):
        """
        Paramters
        -------------
        f: function
            The thing you're trying to find the root of. of the form f(x)
            that is, it must take only one argument
        x0: float-like, or None
            The intial guess for the root. If x0 is None, a value will be
            assigned for it using the bounds. For this to work both bounds 
            cannot be None.
        bounds: list
            of the form [lowerbound,upperbound]. if one or both bounds are
            none, then it will be found automatically such that the root
            is in the range
        """
        self.f=f
        self.maxError=maxError
        self.x0=x0
        
        if fPrime is not None:
            self.fPrime=fPrime
        else:
            self.fPrime=self.derivApprox
        
        self.bounds=bounds
        allBoundsSet=not(None in self.bounds)
        if self.x0 is None and allBoundsSet:
            #if all bounds are set and x0 is not set, then set it to be
            # the average of the bounds
            self.x0=(self.bounds[0]+self.bounds[1])/2
            
        if not allBoundsSet:
            """
            If at least one of the bounds are not given,Set the bounds
            automatically. Note that if self.x0 is None, then self.x0 will 
            be set in this process as well.
            """
            self.autoBounds()
            
        if isinstance(self.bounds,list):
            #cast to array once everything is done
            self.bounds=np.array(self.bounds)
        
    def autoBounds(self,step=0.1):
        """
        Finds bounds such that the root is in between the bounds.
        basically we look around x0, and while theyre f(lowerBound) and 
        f(upperBound) are the same sign, we continue to increase the bounds.
        
        This takes care of two cases; when we just have x0, and when we don't
        have x0, but we have one bound. The case when we have both bounds but 
        no x0 is taken care of in the init
        
        Paramters
        ----------
        step: float, optional
            The increment by which we go up each time
            
        """
        increment=np.zeros(2)
        for i in range(len(increment)):
            if self.bounds[i] is None:
                increment[i]=1#sets increment[i] to 1 if bounds[i] is None
        increment=increment*np.array([-1,1])*step
        #print("Autobounds: increment=",np.round(increment,5))
        #print("Autobounds: starting bounds are:", self.bounds)
        """
        The increment array gets added to bounds each time until the bounds
        truncate the root. The increment array is of the form 
        [-step if bounds[0] is None, 0 otherwise,
             +step if bounds[1] is None, 0 otherwise ]
        """
        #print("increment array is:", increment)
        if self.x0 is None:
            if np.array_equal(self.bounds,[None,None]):
                #Check to make sure the lower bound is defined
                raise ValueError("Bounds and x0 cannot all be None")
            if self.bounds[0] is not None:
                self.x0=self.bounds[0]
            else:
                self.x0=self.bounds[1]

        testBounds=increment+self.x0
        #print("First Test Bounds were", testBounds)
        fOfBounds=np.array([self.f(y) for y in testBounds])
        
        while (np.sign(fOfBounds[0])==np.sign(fOfBounds[1])):
            testBounds=testBounds+increment
            fOfBounds=np.array([self.f(y) for y in testBounds])
            
        for i in (0,1):
            if self.bounds[i] is None:
                self.bounds[i]=testBounds[i]
        #print("AutoBounds: Final bounds are", np.round(self.bounds,5),"\n\n")
        self.x0=np.mean(self.bounds)
        
    def derivApprox(self,x,dx=0.001):
        """
        approximates the derivative if there isn't an analytical one
        """
        num=self.f(x+dx)-self.f(x)
        return num/dx
    
    def newtonRoot(self):
        """
        calculates the root with newton's method
        
        This method is kind of unstable... I wouldnt recomend it unless you
        know what you are doing.
        """
        x=deepcopy(self.x0)
        error=self.f(x)
        #counter=1
        while abs(error)>self.maxError:
            x=x-(self.fPrime(x)/self.f(x))
            error=self.f(x)
            #counter+=1
        #print("Final count:", counter," iterations")
        return x
    
    def goldenSection(self):
        """
        I stole this code off the internet:
        https://en.wikipedia.org/wiki/Golden-section_search#Algorithm
        
        Note golden section is more stable the newton's method, but 
        its is actually a minimum finding algorithm, rather than a zero 
        finding one so you have to take the absolute value of f in order
        to find the root
        """
        gr = (math.sqrt(5) + 1) / 2
        def g(x):
            return abs(self.f(x))
        
        a=self.bounds[0]
        b=self.bounds[1]
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        while abs(c - d) > self.maxError:
            if g(c) < g(d):
                b = d
            else:
                a = c
    
            # We recompute both c and d here to avoid loss of precision which 
            #may lead to incorrect results or infinite loop
            c = b - (b - a) / gr
            d = a + (b - a) / gr

        return (b + a) / 2
    
def errorFunc(k,k0=10*np.pi):
    """
    The function we are attempting to find the roots of
    note that we cannot have k0>k
    
    TODO: things break when you call this with k as an array
    """
    """
    if not isinstance(k,np.ndarray):
        if k>=k0:
            return float('inf')
    
    else:
        locs=np.where(k>=k0)
    """
    if k>= k0:
        return float('inf')
    k0Square=k0**2
    kSquare=k**2
    arg=(k0Square-kSquare)**(1/2)
    val=arg*cot(r0*arg)+k
    return val

def findRoot(k0,plot=False):
    """
    Finds the first root
    
    TODO: Generalize to nth root
    """
    def f(x):
        return errorFunc(x,k0)
    x=0
    while f(x)>0:
        x+=0.1
    # __init__(self,f,x0,bounds=[None,None],fPrime=None,maxError=0.001):
    rootFinder=rootFind(f,None,bounds=[x,None])
    root=rootFinder.goldenSection()
    if plot:
        xs=np.arange(0,3*root,step=0.05)
        ys=np.zeros(len(xs))
        for i in range(len(xs)):
            ys[i]=f(xs[i])
        plt.plot(xs,ys,label="Error Function")
        plt.axvline(x=root,ymin=-10,ymax=10,c="r",label="Root")
        plt.axhline(y=0,xmin=0,xmax=np.max(xs),c="black")
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    return root

        
def makeHist(data,dx,lowerBound,upperBound,windowSize=None):
    """
    My own histogram function becuase numpy's sucks
    
    Maybe in the future we can look at overlaping bins?
    """
    if windowSize is None:
        windowSize=dx
    locs=np.arange(lowerBound,upperBound,dx)
    counts=np.zeros(len(locs),dtype=int)
    for i in range(len(locs)-1):
        lBound=locs[i]-(windowSize/2)
        uBound=locs[i]+(windowSize/2)
        values=data[(lBound<data)&(data<uBound)]
        counts[i]=len(values)
    return locs,counts

def main():
    plot()

    
    
    
    
    
    
    
    
if __name__ =="__main__":
    main()