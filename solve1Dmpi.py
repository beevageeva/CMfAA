import numpy as np
from os.path import join
from mpi4py import MPI

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
    

##Mpi
comm = MPI.COMM_WORLD
nproc = comm.Get_size()

if(mx%nproc!=0):
  print("mx should be div to nproc" )
  sys.exit(0)

rank = comm.Get_rank()

#OUTDIR
outdir="test1dMpi-%s" % method
if(rank==0):
  import os
  if not os.path.exists(outdir):
    os.mkdir(outdir)


my_mx = int(mx/nproc)

#define xx
start=0.0
end=LL
xx_tot = np.linspace(start,end,mx)
xx = xx_tot[rank*my_mx:(rank+1)*my_mx]
dx = xx[1]-xx[0]

from math import pi
def anSol(t):
  return np.sin(pi * xx) * np.exp(-D * pi**2 * t)

  


def bc(yy):
  #only one ghost 
  data_send = np.empty((1), dtype='f')
  data_recv = np.empty((1), dtype='f')
  #send left and receive from right
  if(rank==0):
    dest = MPI.PROC_NULL
  else:
    data_send[0] = yy[1]
    dest = rank - 1

  if(rank==nproc-1):
    src = MPI.PROC_NULL
  else:
    src = rank + 1

  comm.Sendrecv(data_send, dest=dest, sendtag=0, recvbuf=data_recv, source=src, recvtag=0)

  if(rank==nproc-1):
    #set right bc
    yy[-1] = 0
  else:
    yy[-1] = data_recv[0]


  #send right and receive from left
  if(rank==nproc-1):
    dest = MPI.PROC_NULL
  else:
    data_send[0] = yy[my_mx]
    dest = rank + 1

  if(rank==0):
    src = MPI.PROC_NULL
  else:
    src = rank - 1

  comm.Sendrecv(data_send, dest=dest, sendtag=0, recvbuf=data_recv, source=src, recvtag=0)

  if(rank==0):
    #set left bc
    yy[0] = 0
  else:
    yy[0] = data_recv[0]



def getMaxDt():
  alpha = 0.5
  return alpha*dx**2/D




###define explicit and implicit update
def explicit(alpha,yy,yy1):
  for ii in range(1,my_mx+1):
    yy1[ii] = alpha*(yy[ii+1] + yy[ii-1]) + (1.0 - 2.0 * alpha) * yy[ii]


def implicit(alpha,yy,yy1):
  aa=np.empty((my_mx))
  bb=np.empty((my_mx+1))
  cc=np.empty((my_mx+1))
  aa[:]=-alpha
  cc[:]=-alpha
  bb[:] = 1.0 + 2.0*alpha
  #left bc for bb and cc 
  if(rank==0):
    cc[0] = 0.0
    bb[0] =1.0
  
  #thomas alg start
  #https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
  dd =np.zeros((my_mx+1))
  dd[:] =  yy[0:-1] 
  ##the forward part
  if(rank!=0):
    #receive from left
    data_recv = np.empty((3), dtype='f')
    comm.Recv([data_recv,3, MPI.DOUBLE], source=rank-1, tag=0)
    dd[0] = data_recv[0] 
    cc[0] = data_recv[1] 
    bb[0] = data_recv[2] 
  #else it is already set as yy[0]

  for ii in range(1,my_mx+1):
    ww = aa[ii-1]/bb[ii-1]
    bb[ii] = bb[ii] - ww * cc[ii-1]
    dd[ii] = dd[ii] - ww * dd[ii-1]

  if(rank!=nproc-1):
    data_send = np.empty((3), dtype='f')
    data_send[0] = dd[-1]
    data_send[1] = cc[-1]
    data_send[2] = bb[-1]
    comm.Send([data_send,3,MPI.DOUBLE], dest=rank+1, tag=0)

  ##the backwards part
  if(rank!=nproc-1):
    #receive from right
    data_recv = np.empty((1), dtype='f')
    comm.Recv([data_recv, 1,MPI.DOUBLE], source=rank+1, tag=0)
    yy1[my_mx+1] = data_recv[0] 
    yy1[my_mx] = (dd[my_mx] - cc[my_mx] * yy1[my_mx+1])/bb[my_mx]
  else:  
    yy1[my_mx] = dd[my_mx]/bb[my_mx]

  for ii in range(my_mx-1,0,-1):
    yy1[ii] = (dd[ii] - cc[ii] * yy1[ii+1])/bb[ii]       
   
  if(rank!=0):
    data_send = np.empty((1), dtype='f')
    data_send[0] = yy1[1]
    comm.Send([data_send,1,MPI.DOUBLE], dest=rank-1, tag=0)
  #thomas alg end

###define explicit and implicit update END


##write to file methods
#Write a single array collected by proc 0 from the others
def writeToFile1(my_uu,nsaved):
  data = np.empty((my_mx), dtype='f')
  if rank != 0:
    data[:] = my_uu[1:-1]
    comm.Send([data, MPI.DOUBLE], dest=0, tag=77)
  else:
    bigArray = np.empty((mx))
    bigArray[:my_mx] = my_uu[1:-1]
    for proc in range(1,nproc):
      comm.Recv([data, MPI.DOUBLE], source=proc, tag=77)
      bigArray[proc*my_mx:(proc+1)*my_mx] = data[:]
    np.savetxt(join(outdir, ("test1d%04d.txt" % nsaved)), bigArray)
    
def writeToFile2(my_uu,nsaved):
  np.savetxt(join(outdir, ("test1d%04d-%04d.txt" % (rank, nsaved))), my_uu[1:-1])
  
#define function to write to file
writeToFile = writeToFile1
#writeToFile = writeToFile2
##write to file methods END




if method=="implicit":
  update = implicit
  cfl=10.0 
else:
  update = explicit
  cfl=1.0  

yy = np.empty((my_mx+2), dtype="f")
yy1 = np.empty((my_mx+2), dtype="f")

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
    if(rank==0):
      print("Iter %d at time %f" % (niter,simTime))
    simTimes.append(simTime)
    ##output
    writeToFile(yy,nsaved)
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
writeToFile(yy, nsaved)
simTimes.append(simTime)
#save times file
if(rank==0):
  print("Iter %d at time %f" % (niter,simTime))
  np.savetxt(join(outdir, "times.txt"), np.array(simTimes))  
elapsed_time = time.time() - start_time
print("Time %f" % elapsed_time, " rank %d" % rank)
