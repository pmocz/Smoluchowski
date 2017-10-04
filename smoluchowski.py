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
  
  # [Example 1.] K = 1
  x = np.linspace(0.001,100,1000);
  cc = 0;
  for t in np.array([0,0.5,1,3]):
    cc+=1
    f = (2./(2.+t))**2 * np.exp(-2./(2.+t)*x)
    g = x*f
    fig = plt.plot(x, f, linewidth=2, color=[0., 0., cc/5.], label='$t='+"%0.1f" % t +'$')
  plt.yscale('log')

  # compare with the numerical solutions
  t = 3.
  Nt = 30
  Nx = 100
  xBound = np.linspace(0.01,100,Nx);
  dx = xBound[1] - xBound[0]
  x = np.linspace(xBound[0]+0.5*dx,xBound[-1]-0.5*dx,Nx-1);
  f0 = np.exp(-x)
  #f0 = x**-2
  K_A = '1'
  f = smolsolve(x, xBound, f0, t, K_A, Nt)
  fig = plt.plot(x, f, 'o', color=[0., 1., 0.])

  plt.legend(loc='lower left')
  plt.xlabel('$x$')
  plt.ylabel('$f(t,x)$')
  plt.axis([0.001, 12, 1e-6, 1e1])
  plt.savefig('solution1.pdf', aspect = 'normal', bbox_inches='tight', pad_inches = 0)
  plt.close()
  


  # [Example 2.] K = x*y
  x = np.linspace(0.001,100,1000);
  cc = 0;
  for t in np.array([0,0.5,1,3]):
    cc+=1
    T = 1+t
    if (t>1):
      T = 2*np.sqrt(t)
    if t > 0:
      f = np.exp(-T*x)/x**2/np.sqrt(t) * sps.iv(1,2*x*np.sqrt(t))
    else:
	  f = np.exp(-x)/x
    g = x*f
    fig = plt.plot(x, f, linewidth=1, color=[cc/5., 0., 0.], label='$t='+"%0.1f" % t +'$')
  plt.yscale('log')
   
  # compare with the numerical solutions
  t = 3.
  Nt = 30
  Nx = 100
  xBound = np.linspace(0.01,100,Nx);
  dx = xBound[1] - xBound[0]
  x = np.linspace(xBound[0]+0.5*dx,xBound[-1]-0.5*dx,Nx-1);
  f0 = np.exp(-x)/x
  K_A = 'x*y'
  f = smolsolve(x, xBound, f0, t, K_A, Nt)
  fig = plt.plot(x, f, 'o', color=[0., 1., 0.])

  plt.legend(loc='lower left')
  plt.xlabel('$x$')
  plt.ylabel('$f(t,x)$')
  plt.axis([0.001, 12, 1e-6, 1e1])
  plt.savefig('solution2.pdf', aspect = 'normal', bbox_inches='tight', pad_inches = 0)
  plt.close()  
   
   
   
   
def smolsolve(x, xBound, f0, t, K_A, Nt):
  """ solve Smoluchowski equations
  Input: x, initial condition, time, kernel, # timestep
  Output: solution f(t,x)
  """
  dx = xBound[1] - xBound[0]
  Nx = x.size
  dt = t / Nt
  g = x * f0
  for t in range(Nt):
    JL = 0*x;
    for i in range(1,Nx):
        for p in range(0,i):
            # K_A = 1
            # this is analytic expression for int_{x_j}^{x_j+1} K_A(x_mid(i),y)/y \, dy
            if K_A == '1':
				kernBndry = np.log(xBound[i-p]/x[i-p-1])
				kern = np.log(xBound[i-p+1:-1]/xBound[i-p:-2])
            elif K_A == 'x*y':
				xA = x[i-p-1];
				xB = xBound[i-p];
				kernBndry = (xB - xA) * x[p];
				xA = xBound[i-p:-2];
				xB = xBound[i-p+1:-1];
				kern      = (xB - xA) * x[p];
            elif K_A == '2+(x/y)^2+(y/x)^2':
				xA = x[i-p-1];
				xB = xBound[i-p];
				kernBndry = (-xA**2 + xB**2 + x[p]**4 * (1./xA**2-1./xB**2)) / (2.*x[p]**2) + 2.*np.log(xB/xA);
				xA = xBound[i-p:-2];
				xB = xBound[i-p+1:-1];
				kern      = (-xA**2 + xB**2 + x[p]**4 * (1./xA**2-1./xB**2)) / (2.*x[p]**2) + 2.*np.log(xB/xA);
            elif K_A == '(x*y)^(15/14)*(x+y)^(9/14)':  # https://arxiv.org/pdf/astro-ph/0201102.pdf
				sys.exit("implement")
            else:
				sys.exit("kernel incorrectly specified") 
            
            JL[i] = JL[i] + dx*g[p] * (kernBndry*g[i-p-1] + np.sum(kern*g[i-p:-1]));
    
    JR = np.roll(JL,-1);
    JR[-1]= 0;
    g = g - dt / dx * ( JR - JL );

  f = g / x
  return f
     
   


if __name__ == "__main__":
  main()
