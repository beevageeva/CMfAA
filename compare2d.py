import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
plt.ion()

from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join

outdirs=["test2d","test2dMpi"]
#times = np.loadtxt(join(outdirs[0], "times.txt"))  
start = 0
#end= len(times)
end= 100000

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
  
for nn in range(start,end+1):
  ax.cla()  
  ax.set_title(nn)
  for outdir in outdirs:
    uu = np.loadtxt(join(outdir, ("test2d%04d.txt" % nn)))
    #timeSim=times[nn-1]
    ax.plot(range(uu.shape[1]),uu[uu.shape[0]//2,:],label="%s hor cut" % outdir)
    ax.plot(range(uu.shape[0]),uu[:,uu.shape[1]//2],label="%s vert cut" % outdir)
  ax.set_ylim((0.0,1.0))
  ax.legend()
  fn = "comp-test2d%04d-cut.png" % nn
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


