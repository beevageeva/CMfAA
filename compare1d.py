import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
plt.ion()

from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join


import sys
if(len(sys.argv)!=2):
  print("One argument needed: implicit or explicit")
  sys.exit()
else:
  method = sys.argv[1]
  if(not method in ["implicit","explicit"]):
    print("method must be implicit or explicit")
    sys.exit()

outdirs=["test1d-%s" % method,"test1dMpi-%s" % method]
#times = np.loadtxt(join(outdirs[0], "times.txt"))  
start = 0
#end= len(times)
end= 100000

LL=1.0

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
  
for nn in range(start,end+1):
  ax.cla()  
  ax.set_title(nn)
  for outdir in outdirs:
    yy = np.loadtxt(join(outdir, ("test1d%04d.txt" % nn)))
    mx=len(yy)
    xx = np.linspace(0,LL,mx,endpoint=False)
    #timeSim=times[nn-1]
    ax.plot(xx,yy,label=outdir)
  ax.set_ylim((0.0,1.0))
  ax.legend()
  fn = "comp-test1d%04d.png" % nn
  fig.canvas.draw()
  plt.draw()
  
  plt.show()

  #savePng = False
  savePng = True
  #save file
  if savePng:
    newfn=join(outdirs[0], fn)
    fig.savefig(newfn)
    print("%s saved" % newfn)


