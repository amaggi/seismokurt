import numpy as np
import scipy.signal as si
import logging
import matplotlib.pyplot as plt
import sys, os

eps = np.finfo(float).eps

def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i

def get_h_parameters(NFIR, fcut):
    """NFIR : length of FIR filter
    fcut: fraction of Nyquist for filter"""
    h = si.firwin(NFIR+1,fcut) * np.exp(2*1j*np.pi*np.arange(NFIR+1)*0.125)
    n = np.arange(2,NFIR+2)
    g = h[(1-n)%NFIR]*(-1)**(1-n)
    NFIR = np.fix((3./2.*NFIR))
    h1 = si.firwin(NFIR+1,2./3*fcut)*np.exp(2j*np.pi*np.arange(NFIR+1)*0.25/3.)
    h2 = h1*np.exp(2j*np.pi*np.arange(NFIR+1)/6.)
    h3 = h1*np.exp(2j*np.pi*np.arange(NFIR+1)/3.)  
    return (h, g, h1, h2, h3)

def get_GridMax(grid):
    index = np.argmax(grid)
    M = np.amax(grid)
    index = np.unravel_index(index,grid.shape)
    return M, index

def Fast_Kurtogram(x, nlevel, Fs=1, NFIR=16, fcut=0.4, opt1=None, opt2=None):
    # Fast_Kurtogram(x,nlevel,Fs)
    # Computes the fast kurtogram of signal x up to level 'nlevel'
    # Maximum number of decomposition levels is log2(length(x)), but it is 
    # recommed to stay by a factor 1/8 below this.
    # Fs = sampling frequency of signal x (default is Fs = 1)
    # opt1 = 1: classical kurtosis based on 4th order statistics
    # opt1 = 2: robust kurtosis based on 2nd order statistics of the envelope
    # (if there is any difference in the kurtogram between the two measures, this is
    # due to the presence of impulsive additive noise)
    # opt2 = 1: the kurtogram is computed via a fast decimated filterbank tree
    # opt2 = 2: the kurtogram is computed via the short-time Fourier transform
    # (option 1 is faster and has more flexibility than option 2 in the design of the
    # analysis filter: a short filter in option 1 gives virtually the same results as option 2)
    #
    # -------------------
    # J. Antoni : 02/2005
    # Translation to Python: T. Lecocq 02/2012
    # -------------------
    N = len(x)
    N2 = np.log2(N) - 7
    if nlevel > N2:
       logging.error('Please enter a smaller number of decomposition levels')

    if opt2 is None:
        #~ opt2 = int(raw_input('Choose the kurtosis measure (classic = 1  robust = 2): '))
        opt2 = 1
    if opt1 is None:
        #~ opt1  = int(raw_input('Choose the algorithm (filterbank = 1  stft-based = 2): '))
        opt1  = 1
    # Fast computation of the kurtogram
    ####################################
    
    if opt1 == 1:
        # 1) Filterbank-based kurtogram
        ############################
        # Analytic generating filters
        
        h, g, h1, h2, h3 = get_h_parameters(NFIR, fcut)
        
        if opt2 == 1:
            Kwav = K_wpQ(x,h,g,h1,h2,h3,nlevel,'kurt2')				# kurtosis of the complex envelope
        #~ else:
            #~ Kwav = K_wpQ(x,h,g,h1,h2,h3,nlevel,'kurt1')				# variance of the envelope magnitude

        #print "Kwav.shape", Kwav.shape
        #~ print "Kwav",Kwav
        
        # keep positive values only!
        Kwav[Kwav <= 0] = 0
        
        #~ plt.subplot(ratio='auto')
        Level_w = np.arange(1,nlevel+1)
        Level_w = np.array([Level_w, Level_w + np.log2(3.)-1])
        Level_w = sorted(Level_w.ravel())
        Level_w = np.append(0,Level_w[0:2*nlevel-1])

        freq_w = Fs*(np.arange(0,3*2.0**nlevel-1+1))/(3*2**(nlevel+1)) + 1.0/(3.*2.**(2+nlevel))
        #~ plt.imshow(Kwav,aspect='auto',extent=(freq_w[0],freq_w[-1],Level_w[0],Level_w[-1]),interpolation='none')
        grid = Kwav
        
        
        return grid, Level_w, freq_w
        
        
    else:
        logging.error('stft-based is not implemented')
    

def K_wpQ(x,h,g,h1,h2,h3,nlevel,opt,level=0):
    # K = K_wpQ(x,h,g,h1,h2,h3,nlevel)
    # Calculates the kurtosis K of the complete quinte wavelet packet transform w of signal x, 
    # up to nlevel, using the lowpass and highpass filters h and g, respectively. 
    # The WP coefficients are sorted according to the frequency decomposition.
    # This version handles both real and analytical filters, but does not yiels WP coefficients
    # suitable for signal synthesis.
    #
    # -----------------------
    # J Antoni : 12/2004 
    # Translation: T. Lecocq 02/2012
    # -----------------------   
    L = np.floor(np.log2(len(x)))
    if level == 0:
        if nlevel >= L:
            logging.error('nlevel must be smaller')
        level=nlevel
    x = x.ravel()
    #~ print "THIS"
    #~ print h, g
    KD, KQ = K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level)
    K = np.zeros((2*nlevel,3*2**nlevel))
    #~ print "******************************************************"
    K[0,:] = KD[0,:]
    for i in range(nlevel-1):
        K[2*i+1,:] = KD[i+1,:]
        K[2*i+2,:] = KQ[i,:]
      
    K[2*nlevel-1,:] = KD[nlevel,:]
    
    #~ print "K Final Shape", K.shape
    return K

def K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level):
    #~ print "LEVEL", level
    #~ print "*" * level
    a,d = DBFB(x,h,g)
    
    N = len(a)
    d = d*np.power(-1.,np.arange(1,N+1))
    K1 = kurt(a[len(h)-1:],opt)
    K2 = kurt(d[len(g)-1:],opt)

    if level > 2:
        a1,a2,a3 = TBFB(a,h1,h2,h3)
        d1,d2,d3 = TBFB(d,h1,h2,h3)
        Ka1 = kurt(a1[len(h)-1:],opt)
        Ka2 = kurt(a2[len(h)-1:],opt)
        Ka3 = kurt(a3[len(h)-1:],opt)
        Kd1 = kurt(d1[len(h)-1:],opt)
        Kd2 = kurt(d2[len(h)-1:],opt)
        Kd3 = kurt(d3[len(h)-1:],opt)
    else:
        Ka1 = 0
        Ka2 = 0
        Ka3 = 0
        Kd1 = 0
        Kd2 = 0
        Kd3 = 0
    
    if level == 1:
        K =np.array([K1*np.ones(3),K2*np.ones(3)]).flatten()
        #~ print 'K.shape',K.shape
        KQ = np.array([Ka1,Ka2,Ka3,Kd1,Kd2,Kd3])
        #~ print 'KQ.shape',KQ.shape
    if level > 1:
        #~ print "entering rec with level %i"%(level-1)
        #~ print "doing A"
        Ka,KaQ = K_wpQ_local(a,h,g,h1,h2,h3,nlevel,opt,level-1)
        #~ print "doing D"
        Kd,KdQ = K_wpQ_local(d,h,g,h1,h2,h3,nlevel,opt,level-1)
        #~ print "out of rec level %i" % (level -1)
        #~ print Ka.shape, Kd.shape
        K1 = K1*np.ones(np.max(Ka.shape))
        K2 = K2*np.ones(np.max(Kd.shape))
        K12 = np.append(K1,K2)
        Kad = np.hstack((Ka, Kd)) #good
        K = np.vstack((K12,Kad)) # good

        Long = 2./6*np.max(KaQ.shape)
        Ka1 = Ka1*np.ones(Long)
        Ka2 = Ka2*np.ones(Long)
        Ka3 = Ka3*np.ones(Long)
        Kd1 = Kd1*np.ones(Long)
        Kd2 = Kd2*np.ones(Long)
        Kd3 = Kd3*np.ones(Long)
        tmp = np.hstack((KaQ,KdQ))
        #~ print "HEEEERE"
        #~ print tmp.shape
        KQ = np.concatenate((Ka1,Ka2,Ka3,Kd1,Kd2,Kd3))
        KQ = np.vstack([KQ, tmp])
        
        #~ if tmp.shape[0] != KQ.shape[0]:
            #~ tmp = tmp.T
        #~ for i in range(tmp.shape[0]):
            #~ KQ = np.vstack((KQ,tmp[i]))
        
        #~ print "4", K.shape, KQ.shape
        

    
    if level == nlevel:
        K1 = kurt(x,opt)
        K = np.vstack((K1*np.ones(np.max(K.shape)), K))
        #~ print "K shape", K.shape
        
        a1,a2,a3 = TBFB(x,h1,h2,h3)
        Ka1 = kurt(a1[len(h)-1:],opt)
        Ka2 = kurt(a2[len(h)-1:],opt)
        Ka3 = kurt(a3[len(h)-1:],opt)
        Long = 1./3*np.max(KQ.shape)
        Ka1 = Ka1*np.ones(Long)
        Ka2 = Ka2*np.ones(Long)
        Ka3 = Ka3*np.ones(Long)
        tmp = np.array(KQ[0:-2])
        #~ print "level==nlevel"
        
        KQ = np.concatenate((Ka1,Ka2,Ka3))
        #~ KQ = np.vstack((tmp,KQ))
        KQ = np.vstack((KQ,tmp))
    
    #~ print "i'm leaving level=%i and K.shape="%level,K.shape, "and KQ.shape=",KQ.shape

    return K, KQ

def kurt(x, opt):
    if opt=='kurt2':
        if np.all(x==0):
            K=0
            E=0
            return K
        x -= np.mean(x)
        E = np.mean(np.abs(x)**2)
        if E < eps:
            K=0
            return K
        
        K = np.mean(np.abs(x)**4)/E**2

        if np.all(np.isreal(x)):
            K = K - 3
        else:
            K = K - 2
    return K

def DBFB(x,h,g):
    # Double-band filter-bank.
    #   [a,d] = DBFB(x,h,g) computes the approximation
    #   coefficients vector a and detail coefficients vector d,
    #   obtained by passing signal x though a two-band analysis filter-bank.
    #   h is the decomposition low-pass filter and
    #   g is the decomposition high-pass filter.
    
    N = len(x)
    La = len(h)
    Ld = len(g)

    # lowpass filter
    a = si.lfilter(h,1,x)
    a = a[1:N:2]
    a = a.ravel()

    # highpass filter
    d = si.lfilter(g,1,x)
    d = d[1:N:2]
    d = d.ravel()
    return (a,d)

def TBFB(x,h1,h2,h3):
    # Trible-band filter-bank.
    #   [a1,a2,a3] = TBFB(x,h1,h2,h3) 
    
    N = len(x)
    La1 = len(h1)
    La2 = len(h2)
    La3 = len(h3)

    # lowpass filter
    a1 = si.lfilter(h1,1,x)
    a1 = a1[2:N:3]
    a1 = a1.ravel()

    # passband filter
    a2 = si.lfilter(h2,1,x)
    a2 = a2[2:N:3]
    a2 = a2.ravel()

    # highpass filter
    a3 = si.lfilter(h3,1,x)
    a3 = a3[2:N:3]
    a3 = a3.ravel()
    return (a1,a2,a3)



    
def K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level=0):
    # c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level)
    # Calculates the kurtosis K of the complete quinte wavelet packet transform w of signal x, 
    # up to nlevel, using the lowpass and highpass filters h and g, respectively. 
    # The WP coefficients are sorted according to the frequency decomposition.
    # This version handles both real and analytical filters, but does not yiels WP coefficients
    # suitable for signal synthesis.
    #
    # -----------------------
    # J Antoni : 12/2004 
    # -----------------------   
    nlevel = len(acoeff)
    L = np.floor(np.log2(len(x)))
    if level==0:
        if nlevel >= L:
            logging.error('nlevel must be smaller !!')
        level = nlevel
    x = x.ravel()
    if nlevel == 0:
        if np.empty(bcoeff):
            c = x
        else:
            c1, c2, c3 = TBFB(x,h1,h2,h3)
            if bcoeff == 0:
                c = c1[len(h1)-1:]
            elif bcoeff == 1:
                c = c2[len(h2)-1:]
            elif bcoeff == 2:
                c = c3[len(h3)-1:]
    else:
        c = K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level)
    return c

def  K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level):
    #print level, x[:10]
    a,d = DBFB(x,h,g)         # perform one analysis level into the analysis tree
    N = len(a)                       
    d = d*np.power(-1.,np.arange(1,N+1))
    level = int(level)
    if level == 1:
        #~ print "bcoeff", bcoeff
        if len(bcoeff) ==0:
          if acoeff[level-1] == 0:
             c = a[len(h)-1:]
          else:
             c = d[len(g)-1:]
        else:
            if acoeff[level-1] == 0:
                c1,c2,c3 = TBFB(a,h1,h2,h3)
            else:
                c1,c2,c3 = TBFB(d,h1,h2,h3)
            if bcoeff == 0:
                c = c1[len(h1)-1:]
            elif bcoeff == 1:
                c = c2[len(h2)-1:]
            elif bcoeff == 2:
                c = c3[len(h3)-1:]
    if level > 1:
        #~ print "acoeff", acoeff[level-1]
        if acoeff[level-1] == 0:
            c = K_wpQ_filt_local(a,h,g,h1,h2,h3,acoeff,bcoeff,level-1)
        else:
            c = K_wpQ_filt_local(d,h,g,h1,h2,h3,acoeff,bcoeff,level-1)
    #print 'kurt', kurt(c,'kurt2')
    #~ print 'c.shape', c.shape
    return c



if __name__ == "__main__":
    from scipy.io.matlab import loadmat
    v1 = loadmat("test_data/VOIE1.mat")
    x = v1['v1']
    Fs = 100

    nlevel= 8
    grid, Level_w, freq_w = Fast_Kurtogram(x, nlevel, Fs)

    extent = (freq_w[0],freq_w[-1],(nlevel*2)-0.5,-0.5)
    
    plt.imshow(np.sqrt(grid),aspect='auto',extent=extent,interpolation='none',origin="upper")
    
    M, index = get_GridMax(grid)
    f1 = freq_w[index[1]]
    l1 = Level_w[index[0]]
    fi = (index[1])/3./2**(nlevel+1)
    fi += 2.**(-2-l1)
    bw = Fs * 2 **-(l1) /2
    fc = Fs * fi
    print "The maximum has be reached at:"
    print "level =",l1
    print "bw =",bw
    print "fc =",fc    
    
    plt.colorbar()
    plt.scatter([fc,],np.where(Level_w == l1)[0],marker=(5,1),c="yellow",s=100)
    plt.ylim(extent[-2],extent[-1])
    plt.xlim(extent[0],extent[1])
    plt.grid()
    plt.show()
