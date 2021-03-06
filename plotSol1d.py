import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import pylab as plt


import sys
if(len(sys.argv)!=2):
  print("One argument needed: implicit or explicit")
  sys.exit()
else:
  method = sys.argv[1]
  if(not method in ["implicit","explicit"]):
    print("method must be implicit or explicit")
    sys.exit()

outdir="test1dMpi-%s" % method

from os.path import join


LL = 1.0
#nproc=4
#myLL = float(LL/nproc)



plt.ion()
#times = np.loadtxt(join(outdir, "times.txt"))  
start = 0
#end= len(times)
end= 100000
#import glob
#glob.glob('145592*.jpg') 
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
for nn in range(start,end+1):
  yy = np.loadtxt(join(outdir, ("test1d%04d.txt" % nn)))
  mx=len(yy)
  xx = np.linspace(0,LL,mx,endpoint=False)
  #timeSim=times[nn-1]
  #timeSim=0
  ax.cla()  
  #ax.set_title(timeSim)
  ax.set_title(nn)
  ax.set_ylim(0.0,1.0)
  ax.plot(xx,yy)
  fig.canvas.draw()
  plt.draw()
  
  plt.show()

  #savePng = False
  savePng = True
  #save file
  if savePng:
    newfn=join(outdir, ("test1d%04d.png" % nn))
    fig.savefig(newfn)
    print("%s saved" % newfn)
