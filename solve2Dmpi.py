import numpy as np
from os.path import join
from mpi4py import MPI

import time
start_time = time.time()

D=1.0

mx = 256
my = 256


import sys
if(len(sys.argv)!=3):
  print("Two arguments needed: nprocx nprocy")
  sys.exit()
else:
  try:
    nprocx = int(sys.argv[1])
  except ValueError:
    print("nprocx must be int")
    sys.exit()
  try:
    nprocy = int(sys.argv[2])
  except ValueError:
    print("nprocy must be int")
    sys.exit()

import sys

comm = MPI.COMM_WORLD
nproctot = comm.Get_size()

##mpi check 
if(nproctot != nprocx * nprocy):
  print(nprocx*nprocy, "(nprocx*nprocy) has to be equal to ", nproctot, "(nproctot)" )
  sys.exit(0)

if(mx%nprocx!=0):
  print("mx should be div to nprocx" )
  sys.exit(0)

if(my%nprocy!=0):
  print("my should be div to nprocy" )
  sys.exit(0)
##mpi check end 

my_mx = int(mx/nprocx)
my_my = int(my/nprocy)


rank = comm.Get_rank()
comm2d = comm.Create_cart(dims = [nprocy,nprocx],periods =[False,False],reorder=False)

coord2d = comm2d.Get_coords(rank)


left,right = comm2d.Shift(direction = 0,disp=1)
down,up = comm2d.Shift(direction = 1,disp=1)

#this defined because comm2d.Get_coords(rr) thows an error when rr==MPI.PROC_NULL
def getCoord2d(rr):
  if(rr!=MPI.PROC_NULL):
    return comm2d.Get_coords(rr)
  return -1

print("Processor ",rank, coord2d, "has his neighbour LR ", left, " coords : ",  getCoord2d(left), " and ",right, " coords : ",  getCoord2d(right))
print("Processor ",rank, coord2d,  "has his neighbour DU ", down, " coords ",  getCoord2d(down), " and ",up, " coords ", getCoord2d(up) )



outdir="test2dMpi"
if(rank==0):
  import os
  if not os.path.exists(outdir):
    os.mkdir(outdir)

LL=1.0

dx=LL/(mx-1)
dy=LL/(my-1)

my_startx = my_mx * (nprocx-1-coord2d[1]) * dx
my_endx = (my_mx *  (nprocx-coord2d[1]) - 1) * dx
my_starty = my_my * coord2d[0] * dy 
my_endy = (my_my * (coord2d[0] + 1) - 1) * dy 


xx = np.linspace(my_startx,my_endx,my_mx)
yy = np.linspace(my_starty,my_endy,my_my)


dx = xx[1]-xx[0]
dy = yy[1]-yy[0]

print("RANK ", rank, " dims X: ", xx[0], xx[-1], dx, " Y: ",  yy[0], yy[-1])

from math import pi




def getMaxDt():
  alpha = 0.5
  return alpha*dx**2/D


time0=0.0
time1=0.025





#notice that corners are not send/recv and are in the corners BC are not set.
def bc(uu):
  ##vertical dir x (first matrix index)
  data_x_send = np.empty(my_my, dtype='f')
  data_x_recv = np.empty(my_my, dtype='f')

  #send up the first row (in the physical domain) and receive from down
  if(up!=MPI.PROC_NULL):
    data_x_send[:] = uu[1,1:-1]

  comm2d.Sendrecv(data_x_send, dest=up, sendtag=0, recvbuf=data_x_recv, source=down, recvtag=0)

  if(down!=MPI.PROC_NULL):
    uu[-1,1:-1] = data_x_recv[:] 
  else:
    #set bottom b.c.
    uu[-1,1:-1] = 0.0

  #send down the last row and receive from up
  if(down!=MPI.PROC_NULL):
    data_x_send[:] = uu[my_mx,1:-1]

  comm2d.Sendrecv(data_x_send, dest=down, sendtag=0, recvbuf=data_x_recv, source=up, recvtag=0)

  if(up!=MPI.PROC_NULL):
    uu[0,1:-1] = data_x_recv[:] 
  else:
    #set top  b.c.
    uu[0,1:-1] = 0.0

  #for send/recv in the y dir (horizontal)
  data_y_send = np.empty(my_mx, dtype='f')
  data_y_recv = np.empty(my_mx, dtype='f')

  #send left the first col (in the physical domain) and receive from right
  if(left!=MPI.PROC_NULL):
    data_y_send[:] = uu[1:-1,1]

  comm2d.Sendrecv(data_y_send, dest=left, sendtag=0, recvbuf=data_y_recv, source=right, recvtag=0)

  if(right!=MPI.PROC_NULL):
    uu[1:-1,-1] = data_y_recv[:] 
  else:
    #set right b.c.
    uu[1:-1,-1] = 0.0

  #send right the last col and receive from left
  if(right!=MPI.PROC_NULL):
    data_y_send[:] = uu[1:-1,my_my]

  comm2d.Sendrecv(data_y_send, dest=right, sendtag=0, recvbuf=data_y_recv, source=left, recvtag=0)

  if(left!=MPI.PROC_NULL):
    uu[1:-1,0] = data_y_recv[:] 
  else:
    #set left  b.c.
    uu[1:-1,0] = 0.0



def writeToFile1(my_uu,nsaved):
  data = np.empty((my_mx*my_my), dtype='f')
  if rank != 0:
    data[:] = np.ndarray.flatten(my_uu[1:-1,1:-1])
    comm.Send([data, MPI.DOUBLE], dest=0, tag=77)
  else:
    bigArray = np.empty((mx,my))
    bigArray[mx-my_mx:mx,:my_my] = my_uu[1:-1,1:-1]
    for proc in range(1,nproctot):
      comm.Recv([data, MPI.DOUBLE], source=proc, tag=77)
      co  = comm2d.Get_coords(proc)
      bigArray[(nprocx-co[1]-1)*my_mx:(nprocx-co[1])*my_mx, co[0]*my_my:(co[0]+1)*my_my] = np.reshape(data,(my_mx,my_my))
    np.savetxt(join(outdir, ("test2d%04d.txt" % nsaved)), bigArray)
    
def writeToFile2(my_uu,nsaved):
  np.savetxt(join(outdir, ("test2d%04d-%04d.txt" % (rank, nsaved))), my_uu[1:-1,1:-1])
  
def writeToFile2_withGhosts(my_uu,nsaved):
  np.savetxt(join(outdir, ("test2d%04d-%04d.txt" % (rank, nsaved))), my_uu)


writeToFile = writeToFile1
#writeToFile = writeToFile2
#writeToFile = writeToFile2_withGhosts


#yy1 of length my_mx +2, dd of length my_mx+1
def thomas(my_mx,dd,yy1,aa,bb,cc,direction):
  if(direction == 0):
    notFirst = (up!=MPI.PROC_NULL)
    notLast = (down!=MPI.PROC_NULL)
    leftProc = up
    rightProc = down
  else:
    notFirst = (left!=MPI.PROC_NULL)
    notLast = (right!=MPI.PROC_NULL)
    leftProc = left
    rightProc = right

  ##the forward part
  if(notFirst):
    #receive from left, up
    data_recv = np.empty((3), dtype='f')
    comm2d.Recv([data_recv,3, MPI.DOUBLE], source=leftProc, tag=0)
    dd[0] = data_recv[0] 
    cc[0] = data_recv[1] 
    bb[0] = data_recv[2] 
  #else it is already set as yy[0]
  for ii in range(1,my_mx+1):
    ww = aa[ii-1]/bb[ii-1]
    bb[ii] = bb[ii] - ww * cc[ii-1]
    dd[ii] = dd[ii] - ww * dd[ii-1]
  if(notLast):
    data_send = np.empty((3), dtype='f')
    data_send[0] = dd[-1]
    data_send[1] = cc[-1]
    data_send[2] = bb[-1]
    comm2d.Send([data_send,3,MPI.DOUBLE], dest=rightProc, tag=0)


  ##the backwards part
  if(notLast):
    #receive from right, down
    data_recv = np.empty((1), dtype='f')
    comm2d.Recv([data_recv, 1,MPI.DOUBLE], source=rightProc, tag=0)
    yy1[my_mx+1] = data_recv[0] 
    yy1[my_mx] = (dd[my_mx] - cc[my_mx] * yy1[my_mx+1])/bb[my_mx]
  else:  
    yy1[my_mx] = dd[my_mx]/bb[my_mx]

  for ii in range(my_mx-1,0,-1):
    yy1[ii] = (dd[ii] - cc[ii] * yy1[ii+1])/bb[ii]          
  if(notFirst):
    data_send = np.empty((1), dtype='f')
    data_send[0] = yy1[1]
    comm2d.Send([data_send,1,MPI.DOUBLE], dest=leftProc, tag=0)


##SOLVE start
uu = np.zeros((my_mx+2,my_my+2))
uu1 = np.zeros((my_mx+2,my_my+2))

##ini cond
from math import exp
for ii in range(1,my_mx+1):
  for jj in range(1,my_my+1):
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
    ##output
    writeToFile(uu,nsaved)
    if(rank==0):
      print("Iter %d at time %f" % (niter,simTime))
      simTimes.append(simTime)
    nsaved+=1

  ##new dt  
  cfl=50.0
  dt = cfl * getMaxDt()

  
  ##update
  alpha = 0.5 * D * dt/dx**2
  beta = 0.5 * D * dt/dy**2


  ##solve start

  #####fix second index
  aa=np.empty((my_mx))
  bb=np.empty((my_mx+1))
  cc=np.empty((my_mx+1))
  aa[:]=-alpha
  cc[:]=-alpha
  #left bc for cc
  if(up==MPI.PROC_NULL):
    cc[0] = 0.0
  dd =np.zeros((my_mx+1))
  xx1 =np.zeros((my_mx+2))
  for jj in range(1,my_my+1):
    bb[:] = 1.0 + 2.0*alpha
    #left bc for bb
    if(up==MPI.PROC_NULL):
      bb[0] = 1.0
    #the last index of uu is not needed
    dd[:] = beta * uu[:-1,jj+1] + (1.0 - 2.0*beta)  * uu[:-1,jj] + beta * uu[:-1,jj-1]
    thomas(my_mx,dd,xx1,aa,bb,cc,0)  
    uu1[1:-1,jj] =  xx1[1:-1] 

  #bc 
  bc(uu1)


  #####fix first index
  aa=np.empty((my_my))
  bb=np.empty((my_my+1))
  cc=np.empty((my_my+1))
  aa[:]=-beta
  cc[:]=-beta
  if(left==MPI.PROC_NULL):
    cc[0] = 0.0
  dd =np.zeros((my_my+1))
  xx1 =np.zeros((my_my+2))
  for ii in range(1,my_mx+1):
    bb[:] = 1.0 + 2.0*beta
    if(left==MPI.PROC_NULL):
      bb[0] = 1.0
    dd[:] = alpha * uu1[ii+1,:-1] + (1.0 - 2.0*alpha)  * uu1[ii,:-1] + alpha * uu1[ii-1,:-1]
    thomas(my_my,dd,xx1,aa,bb,cc,1)  
    uu[ii,1:-1] =  xx1[1:-1] 
 
  #bc 
  bc(uu)
  ##solve end
 
  #next
  simTime +=dt
  niter+=1 

##save last time snapshot
writeToFile(uu,nsaved)

if(rank==0):
  print("Last iter %d at time %f" % (niter,simTime))
  simTimes.append(simTime)
  #save times file
  np.savetxt(join(outdir, "times.txt"), np.array(simTimes))  
elapsed_time = time.time() - start_time
print("Time %f" % elapsed_time, " rank %d" % rank)
