import numpy as np
import matplotlib.pyplot as plt
from obspy.signal import bandpass, envelope, highpass, lowpass

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

def plot_FilteredResult(c,Fr,bw,level,spec = 1):
  sig = np.median(np.abs(c))/np.sqrt(np.pi/2.)
  threshold = sig*raylinv(np.array([.999,]),np.array([1,]))
  t = np.arange(len(x))/float(Fs)
  tc = np.linspace(t[0],t[-1],len(c))

  fig = plt.figure()
  ax1 = plt.subplot(2+spec,1,1)

  Fr *= Fs
  filt = highpass(x,Fr-bw/2, Fs, corners=2)
  filt = lowpass(filt,Fr+bw/2, Fs, corners=2)
  filt /= np.max(filt)

  plt.plot(t,x,'k',label='Original Signal')
  plt.plot(tc,c,'r',label='filtered Signal')
  plt.plot(t,filt,label='Obspy Filterded Signal')
  plt.legend()
  plt.grid(True)

  plt.subplot(2+spec,1,2,sharex=ax1)

  plt.plot(tc,np.abs(c),'k')
  plt.axhline(threshold,c='r')
  
  for ti in tc[ np.where(np.abs(c) >= threshold)[0]]:
    plt.axvline(ti,c='g',zorder=-1)
    ax1.axvline(ti,c='g',zorder=-1)
    plt.xlabel('time [s]')
    plt.grid(True)
  if spec == 1:
    print nextpow2(len(c))
    nfft = int(nextpow2(len(c)))
    env = np.abs(c)**2
    S = np.abs(np.fft.fft( (env.ravel()-np.mean(env))*np.hanning(len(env))/len(env),nfft))
    f = np.linspace(0, 0.5*Fs/2**level,nfft/2)
    plt.subplot(313)
    plt.plot(f,S[:nfft/2],'k')

    plt.title('Fourier transform magnitude of the squared envelope')
    plt.xlabel('frequency [Hz]')
    plt.grid(True)
  plt.show()

if __name__=='__main__':

  from scipy.io.matlab import loadmat
  from SK_grid import Fast_Kurtogram, get_h_parameters, nextpow2, get_GridMax


  v1 = loadmat("test_data/VOIE1.mat")
  x = v1['v1']
  Fs = 100
  nlevel= 8
  grid, Level_w, freq_w = Fast_Kurtogram(x, nlevel, Fs)
  

  #loading and comparing with matlab:
  matlab = np.fromfile('test_data/matlab_grid.np').reshape(16,768)
  plot_GridComparison(matlab,grid,'Matlab','Python',Level_w, freq_w,comptype='rel')

  from SK_process import Find_wav_kurt, raylinv
  N=16
  fcut=0.4
  h, g, h1, h2, h3 = get_h_parameters(N, fcut)
  M, index = get_GridMax(grid)
  
  f1 = freq_w[index[1]]
  l1 = Level_w[index[0]]
  fi = (index[1])/3./2**(nlevel+1)
  fi += 2.**(-2-l1)
  bw = Fs * 2 **-(l1) /2
  fc = Fs * fi
  
  c,Bw,fc = Find_wav_kurt(x,h, g, h1, h2, h3,nlevel,l1,fi,Fs)
  plot_FilteredResult(c, 0.019531,Bw*Fs,Level_w[11])