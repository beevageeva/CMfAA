import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
plt.ion()

from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join

#outdir="test2d"
outdir="test2dMpi"
#times = np.loadtxt(join(outdir, "times.txt"))  
start = 0
#end= len(times)
end= 100000
#import glob
#glob.glob('145592*.jpg')

def plotAll():
  #p2d = True 
  p2d = False 
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
  cb = None
  
  for nn in range(start,end+1):
    uu = np.loadtxt(join(outdir, ("test2d%04d.txt" % nn)))
    #timeSim=times[nn-1]
    timeSim=0
    ax.cla()  
    ax.set_title(timeSim)
    if(p2d):
      img = ax.imshow(uu, norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0), cmap="Blues")
      if(not cb is None):
        cb.remove()
      
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      cb = plt.colorbar(img,cax=cax)
      fn = "test2d%04d.png" % nn
    else:
      ax.plot(range(uu.shape[1]),uu[uu.shape[0]//2,:],label="hor cut")
      ax.plot(range(uu.shape[0]),uu[:,uu.shape[1]//2],label="vert cut")
      ax.legend()
      fn = "test2d%04d-cut.png" % nn
    fig.canvas.draw()
    plt.draw()
    
    plt.show()
  
    #savePng = False
    savePng = True
    #save file
    if savePng:
      newfn=join(outdir, fn)
      fig.savefig(newfn)
      print("%s saved" % newfn)


def plotAllProc(): 
  nproc=4

  nrows = 2
  ncols = int(nproc/nrows)
  fig, ax1 = plt.subplots(nrows=ncols, ncols=nrows, figsize=(8,8))
  if nproc == 1:
  	ax = [ax1]
  else:
  	ax = []
  	if(len(ax1.shape) == 1):
  		for i in range(ax1.shape[0]):
  				ax.append(ax1[i])
  	else:	
  		for i in range(ax1.shape[0]):
  			for j in range(ax1.shape[1]):
  				ax.append(ax1[i,j])
  cb = None
  
  for nn in range(start,end+1):
    #timeSim=times[nn-1]
    for ii in range(nproc):
      ax[ii].cla()  
      uu = np.loadtxt(join(outdir, ("test2d%04d-%04d.txt" % (ii,nn))))
      img = ax[ii].imshow(uu, norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0), cmap="Blues")
      ax[ii].set_title("Proc %d" % ii)
    if(not cb is None):
      cb.remove()
    
    divider = make_axes_locatable(ax[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(img,cax=cax)
    #fig.suptitle(timeSim)
    fig.suptitle(nn)
    fig.canvas.draw()
    plt.draw()
    
    plt.show()
  
    #savePng = False
    savePng = True
    #save file
    if savePng:
      newfn=join(outdir, ("test2d-proc-%04d.png" % nn))
      fig.savefig(newfn)
      print("%s saved" % newfn)



#plotAllProc()
plotAll()
