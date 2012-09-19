import numpy as np

def getBandwidthAndFrequency(nlevel, Fs, level_w, freq_w, level_index, freq_index):

  f1 = freq_w[freq_index]
  l1 = level_w[level_index]
  fi = (freq_index)/3./2**(nlevel+1)
  fi += 2.**(-2-l1)
  bw = Fs * 2 **-(l1) /2
  fc = Fs * fi
 
  return bw, fc
