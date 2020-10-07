import numpy as np
from os.path import join

import time
start_time = time.time()

D=1.0

time0=0.0
time1=0.5

LL = 1.0

mx = 256

import sys
if(len(sys.argv)!=2):
  print("One argument needed: implicit or explicit")
  sys.exit()
else:
  method = sys.argv[1]
  if(not method in ["implicit","explicit"]):
    print("method must be implicit or explicit")
    sys.exit()
    

#OUTDIR
outdir="test1d-%s" % method
import os
if not os.path.exists(outdir):
  os.mkdir(outdir)




#define xx
start=0.0
end=LL
xx = np.linspace(start,end,mx)
dx = xx[1]-xx[0]

from math import pi
def anSol(t):
  return np.sin(pi * xx) * np.exp(-D * pi**2 * t)

  


def bc(yy):
  yy[0] = 0
  yy[mx+1] = 0



def getMaxDt():
  alpha = 0.5
  return alpha*dx**2/D




###define explicit and implicit update
def explicit(alpha,yy,yy1):
  for ii in range(1,mx+1):
    yy1[ii] = alpha*(yy[ii+1] + yy[ii-1]) + (1.0 - 2.0 * alpha) * yy[ii]


def implicit(alpha,yy,yy1):
  aa=np.empty((mx))
  cc=np.empty((mx+1))
  bb=np.empty((mx+1))  
  aa[:]=-alpha
  cc[:]=-alpha
  bb[:] = 1.0 + 2.0*alpha
  #left bc for bb and cc 
  cc[0] = 0.0
  bb[0] =1.0
  
  #thomas alg start
  #https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
  dd =np.zeros((mx+1))
  dd[:] =  yy[0:-1] 
  ##the forward part
  for ii in range(1,mx+1):
    ww = aa[ii-1]/bb[ii-1]
    bb[ii] = bb[ii] - ww * cc[ii-1]
    dd[ii] = dd[ii] - ww * dd[ii-1]
  ##the backwards part
  yy1[mx] = dd[mx]/bb[mx]
  for ii in range(mx-1,0,-1):
    yy1[ii] = (dd[ii] - cc[ii] * yy1[ii+1])/bb[ii]      
  #thomas alg end

###define explicit and implicit update END
    

if method=="implicit":
  update = implicit
  cfl=10.0 
else:
  update = explicit
  cfl=1.0  

yy = np.empty((mx+2), dtype="f")
yy1 = np.empty((mx+2), dtype="f")

#initial conditions
yy[1:-1]=np.sin(pi * xx)

niter_save = 100 
niter = 0
nsaved = 0
simTime = time0
simTimes = []
while(simTime<time1):
  ##aply bc
  bc(yy)
  if(niter%niter_save==0):
    print("Iter %d at time %f" % (niter,simTime))
    simTimes.append(simTime)
    ##output
    np.savetxt(join(outdir, ("test1d%04d.txt" % nsaved)), yy[1:-1])
    nsaved+=1

  ##new dt  
  dt = cfl * getMaxDt()
  
  ##update
  alpha = D * dt/dx**2
  update(alpha,yy,yy1)
 
  ###end  
  #next
  yy[1:-1] = yy1[1:-1]
  simTime +=dt
  niter+=1 

##save last time snapshot
print("Iter %d at time %f" % (niter,simTime))
np.savetxt(join(outdir, ("test1d%04d.txt" % nsaved)), yy[1:-1])
simTimes.append(simTime)
#save times file
np.savetxt(join(outdir, "times.txt"), np.array(simTimes))  

elapsed_time = time.time() - start_time
print("Time %f" % elapsed_time)
