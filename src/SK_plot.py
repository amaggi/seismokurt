import numpy as np
import matplotlib.pyplot as plt

def plot_GridComparison(grid1, grid2, title1, title2, Level_w, freq_w, filename='', comptype='abs'):

  # decide if plot to file or not
  if filename=='' : tofile = False
  else : tofile = True 

  # calculate difference according to comparison type
  if comptype=='abs':
    grid_diff=np.abs(grid2 - grid1)
    title_diff="Abs difference"
  elif comptype=='rel':
    grid_diff=100*(np.abs(grid2 - grid1)/grid1)
    title_diff="% Rel difference"
  else:
    raise UserWarning('Invalid comptype %s'%comptype)
    
  extent = (freq_w[0],freq_w[-1],(nlevel*2)-0.5,-0.5)
  nticks = nlevel*2
  ticks = ["%.1f"%t for t in Level_w]

  # do plot
  for i, gr in enumerate([('grid1',title1),('grid2',title2),('grid_diff',title_diff)]):
      if i == 0:  ax = plt.subplot(1,3,i+1)
      else: plt.subplot(1,3,i+1,sharex=ax,sharey=ax)
      plt.imshow(eval(gr[0]),aspect='auto',extent=extent,interpolation='none')
      plt.colorbar()
      plt.yticks(np.arange(nticks+1),ticks)
      plt.title(gr[1])
      plt.ylim(extent[-2],extent[-1])
      plt.xlim(extent[0],extent[1])
      plt.grid(True)
      plt.ylabel("Level")
      plt.xlabel("Frequency (Hz)")

  
  # plot to file or screen
  if tofile : plt.savefig(filename)
  else: plt.show()


if __name__=='__main__':

  from scipy.io.matlab import loadmat
  from SK_grid import Fast_Kurtogram

  v1 = loadmat("test_data/VOIE1.mat")
  x = v1['v1']
  Fs = 100
  nlevel= 8
  grid, Level_w, freq_w = Fast_Kurtogram(x, nlevel, Fs)
    

  #loading and comparing with matlab:
  matlab = np.fromfile('test_data/matlab_grid.np').reshape(16,768)
  plot_GridComparison(matlab,grid,'Matlab','Python',Level_w, freq_w,comptype='rel')
