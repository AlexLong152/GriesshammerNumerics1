import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from copy import deepcopy
r0=1
def plot(k0=3*np.pi):
    matplotlib.rcParams['text.usetex'] = True
    ks=np.arange(0,k0-0.05,step=0.05)
    ys=np.zeros(len(ks))
    for i in range(len(ks)):
        ys[i]=errorFunc(ks[i],k0)
    labelString="Error for $\kappa_0$="+str(np.round(k0,3))
    plt.scatter(ks,ys,label=labelString,s=0.5)
    plt.legend()
    plt.title("Error")
    plt.ylim(-10,10)
    plt.hlines(0,-2,k0+2,colors="black")
    plt.xlim(0,k0)
    
class rootFind():
    """
    Root finding implimentation, with functionality for newton's method and
    golden section root finding. Recomend use of golden section since it is 
    more .
    
    Note for small k0 there is no root
    """
    def __init__(self,f,x0=None,bounds=[None,None],fPrime=None,
                 maxError=0.00001):
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
        if self.x0 is None:
            self.x0=(self.bounds[0]+self.bounds[1])/2
        
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
          if self.bounds[0] is not None and self.bounds[1] is None:
              if self.f(self.bounds[0])>0:
                  raise ValueError("f(self.bounds[0])>0")
              deriv=self.fPrime(self.bounds[0])
              xDistToYAxis=-self.f(self.bounds[0])/deriv
              steptmp=np.abs(xDistToYAxis/10)
              if steptmp<step:
                  step=steptmp
                  
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
                
    def derivApprox(self,x,dx=1e-6):
        """
        approximates the derivative if there isn't an analytical one for use
        in newton's method
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
        sqrt5=5**(1/2)
        gr = np.float64( (sqrt5 + 1) / 2)
        """
        You would think that you don't need np.float64 around gr, and that
        would be reasonable. Unfortunately it would also be wrong, something 
        to do with mpf types.... im not sure what.
        """
        def g(x):
            """
            abs turns minima finding into root finding method
            
            Again we need np.float64 around this or it breaks for some reason
            """
            return np.float64(abs(self.f(x)))
        assert(not (None in self.bounds))
        
        a=self.bounds[0]
        b=self.bounds[1]
        
        if self.f(a)>=self.f(b):
            raise ValueError("goldenSection: Bounds messed up")
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
            a,b,c,d=np.array([a,b,c,d]).astype(np.float64)
        return (b + a) / 2
    
def errorFunc(k,k0=10*np.pi):
    """
    The function we are attempting to find the roots of
    note that we cannot have k0>k
    
    f(k,k0)=k+sqrt(k0**2 - k**2)*cot(r0*sqrt(k0**2 - k**2))
    
    Parameters
    ------------
    k: float, or ndarray
        the parameter k, or kappa
    k0: float, optional
        the parameter k0, or kappa_0. note this cannot be an array, maybe I 
        will add this functionality later, but it sounds like a pain
        
    Returns
    --------
    val: float or ndarray
        float if k is float, ndarray if k is ndarray
        this value is f(k,k0)=k+sqrt(k0**2 - k**2)*cot(r0*sqrt(k0**2 - k**2))
        
        or where k>= k0 it is float('inf')
    """
    def cot(x):
        return np.float64(np.cos(x)/np.sin(x))
    
    if not isinstance(k,np.ndarray):
        if k>=k0:
            return float('inf')
    else:
        locs=np.where(k>=k0)
    k0Square=k0**2
    kSquare=k**2
    arg=(k0Square-kSquare)**(1/2)
    val=arg*cot(r0*arg)+k
    if isinstance(k,np.ndarray):
        val[locs]=float('inf')
        
    val=np.float64(val)
    return val

def findRoot(k0,plot=False,start=0,step=0.05):
    """
    Finds the first root greater than start
    
    This function is unimportant
    
    Parameters
    -----------
    k0: float
        The input value
    plot: boolean, optional
        True if you want to plot the function
    rootNumber: int, optional
        Which root you want to take, rootNumber==1 leads to the least root
        greater than zero, rootNumber==2 leads to the second root greater
        than zero etc
    step: float, optional
         the step size when stepping through to find a lower bound for
         the root
    
    Returns
    ---------
    start: float
        the next starting location
    root: float
        The root of erroFunc specified
    """
    def f(x):
        return errorFunc(x,k0)
    x=start
    while f(x)>=0:
        """
        Looks where the sign changes from positive to negative and uses that
        as a lower bound
        """
        x+=step
        if x>k0:
            print("findRoot: No root in this region")
            return 0

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


def sameRoot(k0Max,dk0,rootNumber=1,k0Start=np.pi/2 +0.0001):
    """
    tracks the nth root while varying k, if the root is not defined for that 
    value of k, then returns float('inf') in that location of the array 
    instead.
    
    Parameters
    ------------
    k0Start: float
        The starting k0
    k0Max: float
        The ending k0
    dk0: float
        The step
    rootNumber: optional
        which root you are interested in
        
    Returns:
    --------
    k0s: ndarray
        the k0s used
    roots: ndarray
        The roots at each k0
    """
    k0s=np.arange(k0Start,k0Max,dk0)
    roots=np.zeros(len(k0s))
    n=rootNumber-1
    tmp=(2*n+1)*np.pi/2
    loc=np.where(k0s>tmp)[0]
    minloc=np.min(loc)
    roots[:minloc]=float('inf')
    
    step=dk0/20
    leftBound=0
    while errorFunc(leftBound,k0s[minloc])>0:
        leftBound+=step
        
    for i in range(minloc,len(k0s)):
        def f(x):
            return errorFunc(x,k0s[i])
        while f(leftBound)>0:
            leftBound+=step
            
        rightBound=deepcopy(leftBound)+dk0
        while f(rightBound)<0:
            rightBound+=step
            
        roots[i]=rootFind(f,None,bounds=[leftBound,rightBound]).goldenSection()
        #assert abs(f(roots[i]))<0.1
        if abs(f(roots[i]))>0.1:
            print("messed up")
        leftBound=deepcopy(roots[i])#The start for the next loop
        
    return k0s, roots

def makeHist(data,dx,lowerBound=None,upperBound=None,windowSize=None):
    """
    My own histogram function becuase numpy's sucks, with support 
    for overlapping bins
    
    The counts are the number of occurences within windowSize/2 from a central
    value say x, then we move to x+dx and repeat.
    
    Parameters
    -----------
    data: ndarray
        The incoming data, will be flattened
    dx: float
        The step size
    lowerBound: float, optional
        The minimum value to start at, will be set to np.min(data) if not
        assigned
    upperBound: float, optional
        The maximum value to start at, will be set to np.max(data) if not
        assigned
    windowSize: float, optional
        The size of the window that we are interested in. windowSize>dx will
        result in smoothing. 
        
    Returns
    -----------
    locs: 1-d ndarray
        The locations of the central values
    counts: 1-d ndarray
        The number of occurences of values within windowSize in the data
    """
    data=deepcopy(data.flatten())
    if windowSize is None:
        windowSize=dx
    if lowerBound is None:
        lowerBound=np.min(data)
    if upperBound is None:
        upperBound=np.max(data)
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