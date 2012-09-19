#filter 

import numpy as np
import scipy.signal as si
import logging
import matplotlib.pyplot as plt
import sys, os
from scipy.io.matlab import loadmat
from obspy.signal import bandpass, envelope, highpass, lowpass

from SK_grid import *


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


def Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,Sc,Fr,opt,Fs=1):
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
    kx = kurt(c,opt)
    
    print "kx", kx
    
    sig = np.median(np.abs(c))/np.sqrt(np.pi/2.)
    print sig
    threshold = sig*raylinv(np.array([.99999,]),np.array([1,]))
    print "threshold", threshold
    #~ spec = int(raw_input('	Do you want to see the envelope spectrum (yes = 1 ; no = 0): '))
    spec = 1
    
    t = np.arange(len(x))/float(Fs)
    tc = np.linspace(t[0],t[-1],len(c))
    
    fig = plt.figure()
    ax1 = plt.subplot(2+spec,1,1)
    

    Fr *= Fs 
    print Fr, Fs, bw, Bw
    #print Fr, Fs, Bw
    filt = highpass(x,Fr-bw/2, Fs, corners=2)
    filt = lowpass(filt,Fr+bw/2, Fs, corners=2)
    filt /= np.max(filt)
    
    
    plt.plot(t,x,'k',label='Original Signal')
    
    #~ mx = loadmat('test_data/x.mat')['x']
    #~ plt.plot(t,mx,'k',ls='--',label='Original Signal (matlab)')
    
    plt.plot(tc,c,'r',label='filtered Signal')
    
    #~ plt.plot(tc,(c/np.max(c))*np.max(np.abs(x)),c='r',label='Filtered Signal')
    plt.plot(t,filt,label='Obspy Filterded Signal')
    
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2+spec,1,2,sharex=ax1)
    
    plt.plot(tc,np.abs(c),'k')
    
    #~ mc = loadmat('test_data/c.mat')['c']
    #~ plt.plot(tc,np.abs(mc),'r')
    
    #~ plt.plot(tc,envelope(c),'k')
    #~ plt.plot(tc,threshold*np.ones(len(c)),'--r')
    plt.axhline(threshold,c='r')
    
    
    for ti in tc[ np.where(np.abs(c) >= threshold)[0]]:
        plt.axvline(ti,c='g',zorder=-1)
        #~ ax1.axvline(ti,c='g',zorder=-1)
    
    #~ plt.title('Envlp of the filtr sgl, Bw=Fs/2^{'+(level+1)+'}, fc='+(Fs*fc)+'Hz, Kurt='+(np.round(np.abs(10*kx))/10)+', \alpha=.1%']
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
        
        #~ mS = loadmat('test_data/S.mat')['S']
        #~ plt.plot(f,mS[:nfft/2],'r')
        
        plt.title('Fourier transform magnitude of the squared envelope')
        plt.xlabel('frequency [Hz]')
        plt.grid(True)
    plt.show()
    return c,Bw,fc

if __name__ == "__main__":
    

    v1 = loadmat("test_data/VOIE1.mat")
    x = v1['v1']
    x = np.array(x,dtype=float)
    Fs = 100
    nlevel= 8
    opt1 = 1
    opt2 = 1
    grid, Level_w, freq_w = Fast_Kurtogram(x, nlevel, Fs)
    
    index = np.argmax(grid)
    M = np.amax(grid)
    index = np.unravel_index(index,grid.shape)
    f1 = freq_w[index[1]]
    l1 = Level_w[index[0]]
    fi = (index[1])/3./2**(nlevel+1)
    fi += 2.**(-2-l1)
    bw = Fs * 2 **-(l1) /2
    fc = Fs * fi
    
    
    N = 16			
    fc = .4					# a short filter is just good enough!
    h, g, h1, h2, h3 = get_h_parameters(N, fc)
    
    #~ test = int(raw_input('Do you want to filter out transient signals from the kurtogram (yes = 1 ; no = 0): '))
    test = 1
    lev = l1
    while test == 1:
        #~ fi = float(raw_input('	Enter the optimal carrier frequency (between 0 and Nyquist) where to filter the signal: '))
        #~ fi = fi/Fs
        fi = fi
        if opt1 == 1:
            #~ lev = int(raw_input('	Enter the optimal level where to filter the signal: '))
            lev = l1
        if opt2 == 1:
            c,Bw,fc = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,lev,fi,'kurt2',Fs)
        #~ else
            #~ [c,Bw,fc] = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,lev,fi,'kurt1',Fs);
        #~ test = int(raw_input('Do you want to keep on filtering out transients (yes = 1 ; no = 0): '))
        test = 0


    print Bw, fc
