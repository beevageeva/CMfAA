import numpy as np
from os.path import join

import time
start_time = time.time()

D=1.0

mx = 256
my = 256


outdir="test2d"
import os
if not os.path.exists(outdir):
  os.mkdir(outdir)

LL=1.0

xx = np.linspace(0.0,LL,mx)
yy = np.linspace(0.0,LL,my)

dx = xx[1]-xx[0]
dy = yy[1]-yy[0]


from math import pi




def getMaxDt():
  alpha = 0.5
  return alpha*dx**2/D


time0=0.0
time1=0.025
def bc(uu):
  uu[0,:] = 0.0
  uu[-1,:] = 0.0
  uu[:,0] = 0.0
  uu[:,-1] = 0.0




 
#yy1 of length mx +2, dd of length mx+1
def thomas(mx,dd,yy1,aa,bb,cc):
  ##the forward part
  for ii in range(1,mx+1):
    ww = aa[ii-1]/bb[ii-1]
    bb[ii] = bb[ii] - ww * cc[ii-1]
    dd[ii] = dd[ii] - ww * dd[ii-1]

  ##the backwards part
  yy1[mx] = dd[mx]/bb[mx]

  for ii in range(mx-1,0,-1):
    yy1[ii] = (dd[ii] - cc[ii] * yy1[ii+1])/bb[ii]          




##SOLVE start
uu = np.zeros((mx+2,my+2))
uu1 = np.zeros((mx+2,my+2))

##ini cond
from math import exp
for ii in range(1,mx+1):
  for jj in range(1,my+1):
    uu[ii,jj] = exp(-20.0 * (xx[ii-1]-0.5*LL)**2 -20.0 * (yy[jj-1]-0.5*LL)**2)
##bc
bc(uu)

niter_save = 1
niter = 0
nsaved = 0
simTime = time0
simTimes = []
while(simTime<time1):
  if(niter%niter_save==0):
    print("Iter %d at time %f" % (niter,simTime))
    simTimes.append(simTime)
    ##output
    np.savetxt(join(outdir, ("test2d%04d.txt" % nsaved)), uu[1:-1,1:-1])
    nsaved+=1

  ##new dt  
  cfl=50.0
  dt = cfl * getMaxDt()

  
  ##update
  alpha = 0.5 * D * dt/dx**2
  beta = 0.5 * D * dt/dy**2


  ##solve start

  #####fix second index
  aa=np.zeros((mx))
  bb=np.zeros((mx+1))  
  cc=np.zeros((mx+1))
  aa[:]=-alpha
  cc[:]=-alpha
  #left bc for cc
  cc[0] = 0.0
  dd =np.zeros((mx+1))
  xx1 =np.zeros((mx+2))
  for jj in range(1,my+1):
    bb[:] = 1.0 + 2.0*alpha
    #left bc for bb
    bb[0] = 1.0
    #the last index of uu is not needed
    dd[:] = beta * uu[:-1,jj+1] + (1.0 - 2.0*beta)  * uu[:-1,jj] + beta * uu[:-1,jj-1]
    thomas(mx,dd,xx1,aa,bb,cc)  
    uu1[1:-1,jj] =  xx1[1:-1] 
    #bc 
    bc(uu1)


  #####fix first index
  aa=np.zeros((my))
  bb=np.zeros((my+1))  
  cc=np.zeros((my+1))
  aa[:]=-beta
  cc[:]=-beta
  cc[0] = 0.0
  dd =np.zeros((my+1))
  xx1 =np.zeros((my+2))
  for ii in range(1,mx+1):
    bb[0] = 1.0
    bb[:] = 1.0 + 2.0*beta
    dd[:] = alpha * uu1[ii+1,:-1] + (1.0 - 2.0*alpha)  * uu1[ii,:-1] + alpha * uu1[ii-1,:-1]
    thomas(my,dd,xx1,aa,bb,cc)  
    uu[ii,1:-1] =  xx1[1:-1] 
 
  #bc 
  bc(uu)
  ##solve end
 
  #next
  simTime +=dt
  niter+=1 

##save last time snapshot
print("Last iter %d at time %f" % (niter,simTime))
np.savetxt(join(outdir, ("test2d%04d.txt" % nsaved)), uu[1:-1,1:-1])
simTimes.append(simTime)
#save times file
np.savetxt(join(outdir, "times.txt"), np.array(simTimes))  

elapsed_time = time.time() - start_time
print("Time %f" % elapsed_time)
