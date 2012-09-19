import numpy as np
import matplotlib.pyplot as plt

def plot_GridComparison(grid1, grid2, title1, title2, filename='', comptype='abs'):

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


  # do plot
  ax = plt.subplot(131)
  plt.imshow(grid1,aspect='auto',interpolation='none')
  plt.colorbar()
  plt.title(title1)

  plt.subplot(132,sharex=ax,sharey=ax)
  plt.imshow(grid2,aspect='auto',interpolation='none')
  plt.colorbar()
  plt.title(title2)

  plt.subplot(133,sharex=ax,sharey=ax)
  plt.imshow(grid_diff,aspect='auto',interpolation='none')
  plt.colorbar()
  plt.title(title_diff)


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
  plot_GridComparison(matlab,grid,'Matlab','Python',comptype='rel')

