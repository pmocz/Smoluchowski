#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps


"""
Solve a Smoluchowski problem
Philip Mocz (2017)
Princeton University

based on FLFM method presented in http://arxiv.org/pdf/1312.7240v1.pdf

The equation is:
df/dt = 1/2 * int_0^x[K_A(x-y,y)f(x-y,t)f(y,t)]dy - int_0^inf[K_A(x,y)f(x,t)f(y,t)]dy
where 
K_A(x,y) is the aggregation kernel
Note that the integral of g := x*f is a conserved quantity

usage: python smoluchowski.py
"""

def main():
  """ Consider a problem with known analytic solution
  compare against numerical solution
  """
  
  # Example 1. K = 1
  Nx = 1000
  x = np.linspace(0.001,100,Nx);
  cc = 0;
  for t in np.array([0,0.5,1,3]):
    cc+=1
    f = (2./(2.+t))**2 * np.exp(-2./(2.+t)*x)
    g = x*f
    fig = plt.plot(x, f, linewidth=2, color=[0., 0., cc/5.], label='$t='+"%0.1f" % t +'$')
  #plt.xscale('log')
  plt.yscale('log')

  ## Example 2. K = xy
  #cc = 0;
  #for t in np.array([0,0.5,1,3]):
  #  cc+=1
  #  T = 1+t
  #  if (t>1):
  #    T = 2*np.sqrt(t)
  #  if t > 0:
  #    f = np.exp(-T*x)/x**2/np.sqrt(t) * sps.iv(1,2*x*np.sqrt(t))
  #  else:
	#  f = np.exp(-x)/x
  #  g = x*f
  #  fig = plt.plot(x, f, linewidth=1, color=[cc/5., 0., 0.])


  # compare with the numerical solutions
  t = 3.
  Nx = 200
  x = np.linspace(0.001,100,Nx);
  Nt = 30
  f0 = np.exp(-x)
  #f0 = x**-2
  K_A = '1'
  f = smolsolve(x, f0, t, K_A, Nt)
  fig = plt.plot(x, f, 'o', color=[0., 1., 0.])

  plt.legend(loc='lower left')
  plt.xlabel('$x$')
  plt.ylabel('$f(t,x)$')
  plt.axis([0, 12, 1e-6, 1e1])
  plt.savefig('solution.pdf', aspect = 'normal', bbox_inches='tight', pad_inches = 0)
  plt.close()
   
   
   
   
def smolsolve(x, f0, t, K_A, Nt):
  """ solve Smoluchowski equations
  Input: x, initial condition, time, kernel, # timestep
  Output: solution f(t,x)
  """
  dx = x[1] - x[0]
  Nx = x.size
  xMid = np.linspace(x[0]+0.5*dx,x[-1]-0.5*dx,Nx-1);
  dt = t / Nt
  g = x * f0
  for t in xrange(Nt):
    JL = 0*x;
    for i in xrange(1,Nx):
        for p in xrange(0,i):
            # K_A = 1
            # this is analytic expression for int_{x_j}^{x_j+1} K_A(x_mid(i),y)/y \, dy
            kernBndry = np.log(x[i-p]/xMid[i-p-1])
            kern = np.log(x[i-p+1:-1]/x[i-p:-2])
            JL[i] = JL[i] + dx*g[p] * (kernBndry*g[i-p-1] + np.sum(kern*g[i-p:-2]));
    
    JR = np.roll(JL,-1);
    JR[-1]= 0;
    g = g - dt / dx * ( JR - JL );
  
  f = g / x
  return f
     
   


if __name__ == "__main__":
  main()
