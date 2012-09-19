#filter 

import numpy as np
import scipy.signal as si
import logging
import matplotlib.pyplot as plt
import sys, os
from scipy.io.matlab import loadmat
from obspy.signal import bandpass, envelope, highpass, lowpass

from SK_grid import *

def getFTSquaredEnvelope(c):
#  print nextpow2(len(c))
  nfft = int(nextpow2(len(c)))
  env = np.abs(c)**2
  S = np.abs(np.fft.fft( (env.ravel()-np.mean(env))*np.hanning(len(env))/len(env),nfft))
  return S

def getBandwidthAndFrequency(nlevel, Fs, level_w, freq_w, level_index, freq_index):

  f1 = freq_w[freq_index]
  l1 = level_w[level_index]
  fi = (freq_index)/3./2**(nlevel+1)
  fi += 2.**(-2-l1)
  bw = Fs * 2 **-(l1) /2
  fc = Fs * fi
 
  return bw, fc, fi

def binary(i,k):
    # return the coefficients of the binary expansion of i:
    # i = a(1)*2^(k-1) + a(2)*2^(k-2) + ... + a(k)

    if i>=2**k:
        logging.error('i must be such that i < 2^k !!')
    
    a = np.zeros(k+1)
    #~ print a.shape
    temp = i
    for l in np.arange(k,0,-1):
        a[k-l] = np.fix(temp/2**l)
        temp = temp - a[k-l]*2**l
    return a

def raylinv(p,b):
    #RAYLINV  Inverse of the Rayleigh cumulative distribution function (cdf).
    #   X = RAYLINV(P,B) returns the Rayleigh cumulative distribution 
    #   function with parameter B at the probabilities in P.

    #~ if nargin <  1: 
        #~ logging.error('Requires at least one input argument.') 

    # Initialize x to zero.
    x = np.zeros(len(p))
    # Return NaN if the arguments are outside their respective limits.
    k = np.where(((b <= 0)| (p < 0)| (p > 1)))[0]
    
    if len(k) != 0: 
        tmp  = np.NaN
        x[k1] = tmp(len(k))

    # Put in the correct values when P is 1.
    k = np.where(p == 1)[0]
    #~ print k
    if len(k)!=0:
        tmp  = Inf
        x[k] = tmp(len(k))

    k = np.where(((b > 0) & (p > 0) & (p < 1)))[0]
    #~ print k
    
    if len(k)!=0:
        pk = p[k]
        bk = b[k]
        #~ print pk, bk
        x[k] = np.sqrt((-2*bk ** 2) * np.log(1 - pk))
    return x


def Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,Sc,Fr,Fs=1):
    # [c,Bw,fc,i] = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,Sc,Fr,opt2)
    # Sc = -log2(Bw)-1 with Bw the bandwidth of the filter
    # Fr is in [0 .5]
    #
    # -------------------
    # J. Antoni : 12/2004
    # -------------------
    level = np.fix((Sc))+ ((Sc%1) >= 0.5) * (np.log2(3)-1)
    Bw = 2**(-level-1)
    bw = Fs * Bw
    freq_w = np.arange(0,2**(level-1)) / 2**(level+1) + Bw/2.
    J = np.argmin(np.abs(freq_w-Fr))
    fc = freq_w[J]
    i = int(np.round(fc/Bw-1./2))
    if level % 1 == 0:
        acoeff = binary(i, level)
        bcoeff = np.array([])
        temp_level = level
    else:
        i2 = np.fix((i/3.))
        temp_level = np.fix((level))-1
        acoeff = binary(i2,temp_level)
        bcoeff = np.array([i-i2*3,])
    acoeff = acoeff[::-1]
    c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,temp_level)
    
    sig = np.median(np.abs(c))/np.sqrt(np.pi/2.)
    threshold = sig*raylinv(np.array([.999,]),np.array([1,]))

    return c,Bw,fc

if __name__ == "__main__":
    
  pass
