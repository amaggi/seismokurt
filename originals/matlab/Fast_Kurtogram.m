function c = Fast_Kurtogram(x,nlevel,Fs,opt1,opt2)
% Fast_Kurtogram(x,nlevel,Fs)
% Computes the fast kurtogram of signal x up to level 'nlevel'
% Maximum number of decomposition levels is log2(length(x)), but it is 
% recommended to stay by a factor 1/8 below this.
% Fs = sampling frequency of signal x (default is Fs = 1)
% opt1 = 1: classical kurtosis based on 4th order statistics
% opt1 = 2: robust kurtosis based on 2nd order statistics of the envelope
% (if there is any difference in the kurtogram between the two measures, this is
% due to the presence of impulsive additive noise)
% opt2 = 1: the kurtogram is computed via a fast decimated filterbank tree
% opt2 = 2: the kurtogram is computed via the short-time Fourier transform
% (option 1 is faster and has more flexibility than option 2 in the design of the
% analysis filter: a short filter in option 1 gives virtually the same results as option 2)
%
% -------------------
% J. Antoni : 02/2005
% -------------------

N = length(x);
N2 = log2(N) - 7;
if nlevel > N2
   error('Please enter a smaller number of decomposition levels');
end

if nargin < 5
   opt2 = input('Choose the kurtosis measure (classic = 1 ; robust = 2): ');
   if nargin < 4
      opt1  = input('Choose the algorithm (filterbank = 1 ; stft-based = 2): ');
      if nargin < 3
         Fs = 1;
      end
   end
 end

% Fast computation of the kurtogram
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opt1 == 1
   % 1) Filterbank-based kurtogram
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Analytic generating filters
   N = 16;			fc = .4;					% a short filter is just good enough!
   h = fir1(N,fc).*exp(2i*pi*(0:N)*.125);
   n = 2:N+1;
   g = h(1+mod(1-n,N)).*(-1).^(1-n);
   % 
   N = fix(3/2*N);
   h1 = fir1(N,2/3*fc).*exp(2i*pi*(0:N)*.25/3);
   h2 = h1.*exp(2i*pi*(0:N)/6);
   h3 = h1.*exp(2i*pi*(0:N)/3);  
   % 
   if opt2 == 1
      Kwav = K_wpQ(x,h,g,h1,h2,h3,nlevel,'kurt2');				% kurtosis of the complex envelope
   else
      Kwav = K_wpQ(x,h,g,h1,h2,h3,nlevel,'kurt1');				% variance of the envelope magnitude
   end
   Kwav = Kwav.*(Kwav>0);												% keep positive values only!
   
else 
   % 2) STFT-based kurtogram
   %%%%%%%%%%%%%%%%%%%%%%%%%
   Nfft = 2.^[3:nlevel+2];				% level 1 of wav_kurt roughly corresponds to a 4-sample hanning window with stft_kurt
   %											  or a 8-sample flattop		
   temp = [3*Nfft(1)/2 3*Nfft(1:end-2);Nfft(2:end)];
   Nfft = [Nfft(1) temp(:)'];
   if opt2 == 1
      Kstft = Kf_fft(x,Nfft,1,'kurt2');							% kurtosis of the complex envelope
      Kx = kurt(x,'kurt2');
   else
      Kstft = Kf_fft(x,Nfft,1,'kurt1');							% variance of the envelope magnitude
      Kx = kurt(x,'kurt1');
   end
   Kstft = [Kx*ones(1,size(Kstft,2));Kstft];
   Kstft = Kstft.*(Kstft>0);											% keep positive values only!
   
end

% Graphical results
%%%%%%%%%%%%%%%%%%%
figure
if opt1 == 1
   Level_w = 1:nlevel;	Level_w = [Level_w;Level_w+log2(3)-1];	Level_w = Level_w(:); Level_w = [0 Level_w(1:2*nlevel-1)'];
   freq_w = Fs*((0:3*2^nlevel-1)/(3*2^(nlevel+1)) + 1/(3*2^(2+nlevel)));
   imagesc(freq_w,1:2*nlevel,Kwav),colorbar,[I,J,M] = max_IJ(Kwav);
   xlabel('frequency [Hz]'),set(gca,'ytick',1:2*nlevel,'yticklabel',round(Level_w*10)/10),ylabel('level k')
   fi = (J-1)/3/2^(nlevel+1);   fi = fi + 2^(-2-Level_w(I));
   if opt2 == 1
      title(['fb-kurt.2 - K_{max}=',num2str(round(10*M)/10),' @ level ',num2str(fix(10*Level_w(I))/10),', Bw= ',num2str(Fs*2^-(Level_w(I)+1)),'Hz, f_c=',num2str(Fs*fi),'Hz'])
   else
      title(['fb-kurt.1 - K_{max}=',num2str(round(10*M)/10),' @ level ',num2str(fix(10*Level_w(I))/10),', Bw= ',num2str(Fs*2^-(Level_w(I)+1)),'Hz, f_c=',num2str(Fs*fi),'Hz'])
   end
else
   LNw_stft = [0 log2(Nfft)];			
   freq_stft = Fs*((0:Nfft(end)/2-1)/Nfft(end) + 1/Nfft(end)/2);
   %freq_stft = Fs*(0:Nfft(end)/2-1)/Nfft(end);
   imagesc(freq_stft,1:2*nlevel,Kstft),colorbar,[I,J,M] = max_IJ(Kstft);	
   fi = (J-1)/Nfft(end);	
   xlabel('frequency [Hz]'),set(gca,'ytick',1:2*nlevel,'yticklabel',round(LNw_stft*10)/10),ylabel('level: log2(Nw)')
   if opt2 == 1
      title(['stft-kurt.2 - K_{max}=',num2str(round(10*M)/10),' @ Nw=2^{',num2str(fix(10*LNw_stft(I))/10),'}, fc=',num2str(Fs*fi),'Hz'])
   else
      title(['stft-kurt.1 - K_{max}=',num2str(round(10*M)/10),' @ Nw=2^{',num2str(fix(10*LNw_stft(I))/10),'}, fc=',num2str(Fs*fi),'Hz'])
   end
end

% Signal filtering
%%%%%%%%%%%%%%%%%%
c = [];
test = input('Do you want to filter out transient signals from the kurtogram (yes = 1 ; no = 0): ');
while test == 1
   fi = input(['	Enter the optimal carrier frequency (btw 0 and ',num2str(Fs/2),') where to filter the signal: ']);
   fi = fi/Fs;
   if opt1 == 1
      lev = input(['	Enter the optimal level (btw 0 and ',num2str(nlevel),') where to filter the signal: ']);
      if opt2 == 1
         [c,Bw,fc] = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,lev,fi,'kurt2',Fs);
      else
         [c,Bw,fc] = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,lev,fi,'kurt1',Fs);
      end
   else
      lev = input(['	Enter the optimal level (btw 0 and ',num2str(nlevel+2),') where to filter the signal: ']);
      if opt2 == 1
         [c,Nw,fc] = Find_stft_kurt(x,nlevel,lev,fi,'kurt2',Fs);
      else
         [c,Nw,fc] = Find_stft_kurt(x,nlevel,lev,fi,'kurt1',Fs);
      end
   end
   test = input('Do you want to keep on filtering out transients (yes = 1 ; no = 0): ');
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I,J,M] = max_IJ(X)
% [I,J] = max_IJ(X) returns the row and column indices of
% the maximum in the matrix X.
%
% -------------------
% J. Antoni : 07/2004
% -------------------

[temp,tempI] = max(X);
[M,J] = max(temp);
I = tempI(J);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function K = K_wpQ(x,h,g,h1,h2,h3,nlevel,opt,level)
% K = K_wpQ(x,h,g,h1,h2,h3,nlevel)
% Calculates the kurtosis K of the complete quinte wavelet packet transform w of signal x, 
% up to nlevel, using the lowpass and highpass filters h and g, respectively. 
% The WP coefficients are sorted according to the frequency decomposition.
% This version handles both real and analytical filters, but does not yiels WP coefficients
% suitable for signal synthesis.
%
% -----------------------
% Jérôme Antoni : 12/2004 
% -----------------------   

L = floor(log2(length(x)));
if nargin == 8
   if nlevel >= L
      error('nlevel must be smaller !!');
   end
   level = nlevel;
end
x = x(:);										 % shapes the signal as column vector if necessary

[KD,KQ] = K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level);

K = zeros(2*nlevel,3*2^nlevel);
K(1,:) = KD(1,:);
for i = 1:nlevel-1
   K(2*i,:) = KD(i+1,:);
   K(2*i+1,:) = KQ(i,:);
end
K(2*nlevel,:) = KD(nlevel+1,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [K,KQ] = K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level)

[a,d] = DBFB(x,h,g);                    % perform one analysis level into the analysis tree

N = length(a);                       
d = d.*(-1).^(1:N)';

K1 = kurt(a(length(h):end),opt);
K2 = kurt(d(length(g):end),opt);

if level > 2
   [a1,a2,a3] = TBFB(a,h1,h2,h3);
   [d1,d2,d3] = TBFB(d,h1,h2,h3);
   
   Ka1 = kurt(a1(length(h):end),opt);
   Ka2 = kurt(a2(length(h):end),opt);
   Ka3 = kurt(a3(length(h):end),opt);
   Kd1 = kurt(d1(length(h):end),opt);
   Kd2 = kurt(d2(length(h):end),opt);
   Kd3 = kurt(d3(length(h):end),opt);
else
   Ka1 = 0;
   Ka2 = 0;
   Ka3 = 0;
   Kd1 = 0;
   Kd2 = 0;
   Kd3 = 0;
end

if level == 1
   K =[K1*ones(1,3),K2*ones(1,3)];
   KQ = [Ka1 Ka2 Ka3 Kd1 Kd2 Kd3];
end

if level > 1
   [Ka,KaQ] = K_wpQ_local(a,h,g,h1,h2,h3,nlevel,opt,level-1);
   [Kd,KdQ] = K_wpQ_local(d,h,g,h1,h2,h3,nlevel,opt,level-1);
   
   K1 = K1*ones(1,length(Ka));
   K2 = K2*ones(1,length(Kd));
   K = [K1 K2; Ka Kd];
   
   Long = 2/6*length(KaQ);
   Ka1 = Ka1*ones(1,Long);
   Ka2 = Ka2*ones(1,Long);
   Ka3 = Ka3*ones(1,Long);
   Kd1 = Kd1*ones(1,Long);
   Kd2 = Kd2*ones(1,Long);
   Kd3 = Kd3*ones(1,Long);
   KQ = [Ka1 Ka2 Ka3 Kd1 Kd2 Kd3; KaQ KdQ];
end

if level == nlevel
   K1 = kurt(x,opt);
   K = [ K1*ones(1,length(K));K];
   
   [a1,a2,a3] = TBFB(x,h1,h2,h3);
   Ka1 = kurt(a1(length(h):end),opt);
   Ka2 = kurt(a2(length(h):end),opt);
   Ka3 = kurt(a3(length(h):end),opt);
   Long = 1/3*length(KQ);
   Ka1 = Ka1*ones(1,Long);
   Ka2 = Ka2*ones(1,Long);
   Ka3 = Ka3*ones(1,Long);   
   KQ = [Ka1 Ka2 Ka3; KQ(1:end-2,:)];
end

% --------------------------------------------------------------------
function K = kurt(x,opt)
if strcmp(opt,'kurt2')
   if all(x == 0), K = 0; E = 0;return;end
   x = x - mean(x);
   E = mean(abs(x).^2);
   if E < eps, K = 0; return;end
   K = mean(abs(x).^4)/E^2;
   if all(isreal(x))
      K = K - 3;								% real signal
   else
      K = K - 2;
   end
elseif strcmp(opt,'kurt1')
   if all(x == 0), K = 0; E = 0;return;end
   x = x - mean(x);
   E = mean(abs(x));
   if E < eps, K = 0; return;end
   K = mean(abs(x).^2)/E^2;
   if all(isreal(x))
      K = K - 1.57;							% real signal
   else
      K = K - 1.27;
   end
end

% ------------------------------------------------------------------------
function [a,d] = DBFB(x,h,g)
% Double-band filter-bank.
%   [a,d] = DBFB(x,h,g) computes the approximation
%   coefficients vector a and detail coefficients vector d,
%   obtained by passing signal x though a two-band analysis filter-bank.
%   h is the decomposition low-pass filter and
%   g is the decomposition high-pass filter.

N = length(x);
La = length(h);
Ld = length(g);

% lowpass filter
a = filter(h,1,x);
a = a(2:2:N);
a = a(:);

% highpass filter
d = filter(g,1,x);
d = d(2:2:N);
d = d(:);

% ------------------------------------------------------------------------
function [a1,a2,a3] = TBFB(x,h1,h2,h3)
% Trible-band filter-bank.
%   [a1,a2,a3] = TBFB(x,h1,h2,h3) 

N = length(x);
La1 = length(h1);
La2 = length(h2);
La3 = length(h3);

% lowpass filter
a1 = filter(h1,1,x);
a1 = a1(3:3:N);
a1 = a1(:);

% passband filter
a2 = filter(h2,1,x);
a2 = a2(3:3:N);
a2 = a2(:);

% highpass filter
a3 = filter(h3,1,x);
a3 = a3(3:3:N);
a3 = a3(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function K = Kf_fft(x,Nfft,foverlap,opt)

if rem(log2(Nfft),1)~=0
   error('Nfft should contain powers of two only !!')
end
if rem(foverlap,1)~=0 | foverlap ==0
   error('foverlap must be a non-zero integer !!')
end

N = length(Nfft);
L = Nfft(N)*foverlap;
K = zeros(N,L/2);

for i = 1:N
   Window = hanning(Nfft(i));		% bandwidth(3dB) ~ .6/N (small N) --> .7/N (large N)
   Nw = Nfft(i);
   Noverlap = fix(3*Nw/4); 
   NFFT = 2^nextpow2(Nfft(i)*foverlap);
   temp = Kf_W2(x,NFFT,Noverlap,Window,opt);
   temp = temp(1:NFFT/2);
   temp = repmat(temp',L/2/length(temp),1);
   K(i,:) = reshape(temp,L/2,1)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Kf,M4,M2,k] = Kf_W(x,Nfft,Noverlap,Window,opt)
% [Kf,M4,M2] = Kf_W(x,Nfft,Noverlap,Window) 
% Welch's estimate of :       
%       1) the freq.-conditionned Kurtosis :  Kf(f) = M4(f)/M2(f)^2 - 2 
%       2) the 4th-order moment spectrum :   M4(f) = E{|X(f)|^4}
%       3) the 2nd-order moment spectrum :   M2(f) = E{|X(f)|^2}
%
% Caution : this version applies to stationary signals only !!
%
% x and y are divided into overlapping sections (Noverlap taps), each of which is
% detrended, windowed and zero-padded to length Nfft. 
% Note : use analytic signal to avoid correlation between + and - frequencies
% -----
%
% --------------------------
% Author: J. Antoni, 11-2003
% --------------------------

Window = Window(:)/norm(Window);		% Window Normalization
n = length(x);								% Number of data points
nwind = length(Window); 				% length of window
if nwind<=Noverlap,
   error('nwind must be > Noverlap');
end
x = x(:);		
k = fix((n-Noverlap)/(nwind-Noverlap));	% Number of windows

% 1) Moment-based spectrum
% -------------------------
index = 1:nwind;
f = (0:Nfft-1)/Nfft;
t = (0:n-1)';
M4 = 0;
M2 = 0;

for i=1:k
   xw = Window.*x(index);
   Xw = fft(xw,Nfft);	
   if strcmp(opt,'kurt2')
      M4 = abs(Xw).^4 + M4;   
      M2 = abs(Xw).^2 + M2;  
   else
      M4 = abs(Xw).^2 + M4;   
      M2 = abs(Xw) + M2;
   end
   index = index + (nwind - Noverlap);
end

% normalize
M4 = M4/k;   
M2 = M2/k; 
Kf = M4./M2.^2;

if strcmp(opt,'kurt2')
   Kf = Kf - 2;
   b = 1;
else
   Kf = Kf - 1.27;
   b = .3;
end

% reduce biais near f = 0 mod(1/2)
W = abs(fft(Window.^2,Nfft)).^2;
Wb = zeros(Nfft,1);
for i = 0:Nfft-1,
   Wb(1+i) = W(1+mod(2*i,Nfft))/W(1);
end;
Kf = Kf - b*Wb;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xf,Nw,fc] = Find_stft_kurt(x,nlevel,LNw,Fr,opt,Fs)
% [xf,Nw,fc] = Find_stft_kurt(x,nlevel,LNw,Fr,opt2)
% LNw = log2(Nw) with Nw the analysis window of the stft
% Fr is in [0 .5]
%
% -------------------
% J. Antoni : 12/2004
% -------------------

if nargin < 6
   Fs = 1;
end

Nfft = 2.^[3:nlevel+2];					
temp = [3*Nfft(1)/2 3*Nfft(1:end-2);Nfft(2:end)];
Nfft = [Nfft(1) temp(:)'];
LNw_stft = [0 log2(Nfft)];			
[temp,I] = min(abs(LNw_stft-LNw));
Nw = 2^LNw_stft(I);

NFFT = 2^nextpow2(Nw);
freq_stft = (0:NFFT/2-1)/NFFT;
[temp,J] = min(abs(freq_stft-Fr+1/NFFT/4));
fc = freq_stft(J);

if LNw > 0
   b = hanning(Nw)';
   b = b/sum(b);
   b = b.*exp(2i*pi*(0:Nw-1)*fc);
   xf = fftfilt(b,x);
   xf = xf(fix(Nw/2)+1:end);
   dt = fix(Nw/4);					% downsample by at least 4 samples per window (this corresponds to 75% overlap)
else
   xf = x;
   Nw = 0;
   dt = 1;
end
env = abs(xf(dt:dt:end)).^2;

%temp = xf.*exp(-2i*pi*(0:length(xf)-1)'*fc);
%figure,subplot(211),plot(real(temp))
%subplot(212),plot(real(temp(dt:dt:end)))
kx = kurt(xf(dt:dt:end),opt);
sig = median(abs(xf(dt:dt:end)))/sqrt(pi/2);
threshold = sig*raylinv(.999,1);

spec = input('	Do you want to see the envelope spectrum (yes = 1 ; no = 0): ');
figure
t = (0:length(x)-1)/Fs;
tf = t(fix(Nw/2)+1:end);
subplot(2+spec,1,1),plot(t,x,'k'),title('Original signal')
subplot(2+spec,1,2),plot(tf,real(xf),'k'),hold on,plot(tf,threshold*ones(size(xf)),':r'),plot(tf,-threshold*ones(size(xf)),':r'),%plot(abs(xf),'r'),
title(['Filtered signal, Nw=2^{',num2str(LNw_stft(I)),'}, fc=',num2str(Fs*fc),'Hz, Kurt=',num2str(fix(10*kx)/10),', \alpha=.1%'])
xlabel('time [s]')
if spec == 1
   nfft = 2^nextpow2(length(env));
   S = abs(fft((env(:)-mean(env)).*hanning(length(env)),nfft)/length(env));
   f = linspace(0,.5*Fs/dt,nfft/2);
   subplot(3,1,3),plot(f,S(1:nfft/2),'k'),title('Fourier transform magnitude of the squared filtered signal'),xlabel('frequency [Hz]')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [c,Bw,fc,i] = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,Sc,Fr,opt,Fs)
% [c,Bw,fc,i] = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,Sc,Fr,opt2)
% Sc = -log2(Bw)-1 with Bw the bandwidth of the filter
% Fr is in [0 .5]
%
% -------------------
% J. Antoni : 12/2004
% -------------------

if nargin < 11
   Fs = 1;
end

level = fix(Sc) + (rem(Sc,1)>=0.5)*(log2(3)-1);
Bw = 2^(-level-1);
freq_w = (0:2^level-1)/(2^(level+1))+Bw/2;
[temp,J] = min(abs(freq_w-Fr));
fc = freq_w(J);
i = round((fc/Bw-1/2));

if rem(level,1) == 0
   acoeff = binary(i,level);
   bcoeff = [];
   temp_level = level;
else
   i2 = fix(i/3);
   temp_level = fix(level)-1;
   acoeff = binary(i2,temp_level);
   bcoeff = i-i2*3;
end
acoeff = acoeff(end:-1:1);

c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,temp_level);

kx = kurt(c,opt);
sig = median(abs(c))/sqrt(pi/2);
threshold = sig*raylinv(.999,1);

spec = input('	Do you want to see the envelope spectrum (yes = 1 ; no = 0): ');
figure
t = (0:length(x)-1)/Fs;
tc = linspace(t(1),t(end),length(c));
subplot(2+spec,1,1),plot(t,x,'k'),title('Original signal')
%subplot(2+spec,1,2),plot(tc,real(c),'k'),hold on,plot(tc,threshold*ones(size(c)),':r'),plot(tc,-threshold*ones(size(c)),':r')
%title(['Complx envlp of the filtr sgl (real part), Bw=Fs/2^{',num2str(level+1),'}, fc=',num2str(Fs*fc),'Hz, Kurt=',num2str(fix(10*kx)/10),', \alpha=.1%'])
subplot(2+spec,1,2),plot(tc,abs(c),'k'),hold on,plot(tc,threshold*ones(size(c)),':r')
title(['Envlp of the filtr sgl, Bw=Fs/2^{',num2str(level+1),'}, fc=',num2str(Fs*fc),'Hz, Kurt=',num2str(fix(10*kx)/10),', \alpha=.1%'])
xlabel('time [s]')
if spec == 1
   nfft = 2^nextpow2(length(c));
   env = abs(c).^2;
   S = abs(fft((env(:)-mean(env)).*hanning(length(env))/length(env),nfft));
   f = linspace(0,.5*Fs/2^level,nfft/2);
   subplot(313),plot(f,S(1:nfft/2),'k'),title('Fourier transform magnitude of the squared envelope'),xlabel('frequency [Hz]')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function a = binary(i,k)
% return the coefficients of the binary expansion of i:
% i = a(1)*2^(k-1) + a(2)*2^(k-2) + ... + a(k)

if i>=2^k
   error('i must be such that i < 2^k !!')
end
a = zeros(1,k);
temp = i;
for l = k-1:-1:0
   a(k-l) = fix(temp/2^l);
   temp = temp - a(k-l)*2^l;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level)
% c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level)
% Calculates the kurtosis K of the complete quinte wavelet packet transform w of signal x, 
% up to nlevel, using the lowpass and highpass filters h and g, respectively. 
% The WP coefficients are sorted according to the frequency decomposition.
% This version handles both real and analytical filters, but does not yiels WP coefficients
% suitable for signal synthesis.
%
% -----------------------
% Jérôme Antoni : 12/2004 
% -----------------------   

nlevel = length(acoeff);
L = floor(log2(length(x)));
if nargin == 8
   if nlevel >= L
      error('nlevel must be smaller !!');
   end
   level = nlevel;
end
x = x(:);										 % shapes the signal as column vector if necessary

if nlevel == 0
   if isempty(bcoeff)
      c = x;
   else
      [c1,c2,c3] = TBFB(x,h1,h2,h3);
      if bcoeff == 0;
         c = c1(length(h1):end);
      elseif bcoeff == 1;
         c = c2(length(h2):end);
      elseif bcoeff == 2;
         c = c3(length(h3):end);
      end
   end
else
   c = K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function c = K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level)

[a,d] = DBFB(x,h,g);                % perform one analysis level into the analysis tree

N = length(a);                       
d = d.*(-1).^(1:N)';

if level == 1
   if isempty(bcoeff)
      if acoeff(level) == 0
         c = a(length(h):end);
      else
         c = d(length(g):end);
      end
   else
      if acoeff(level) == 0
         [c1,c2,c3] = TBFB(a,h1,h2,h3);
      else
         [c1,c2,c3] = TBFB(d,h1,h2,h3);
      end
      if bcoeff == 0;
         c = c1(length(h1):end);
      elseif bcoeff == 1;
         c = c2(length(h2):end);
      elseif bcoeff == 2;
         c = c3(length(h3):end);
      end     
   end
end

if level > 1
   if acoeff(level) == 0
      c = K_wpQ_filt_local(a,h,g,h1,h2,h3,acoeff,bcoeff,level-1);
   else
      c = K_wpQ_filt_local(d,h,g,h1,h2,h3,acoeff,bcoeff,level-1);
   end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = raylinv(p,b)
%RAYLINV  Inverse of the Rayleigh cumulative distribution function (cdf).
%   X = RAYLINV(P,B) returns the Rayleigh cumulative distribution 
%   function with parameter B at the probabilities in P.

if nargin <  1, 
    error('Requires at least one input argument.'); 
end

% Initialize x to zero.
x = zeros(size(p));

% Return NaN if the arguments are outside their respective limits.
k1 = find(b <= 0| p < 0 | p > 1);
if any(k1) 
    tmp   = NaN;
    x(k1) = tmp(ones(size(k1)));
end

% Put in the correct values when P is 1.
k = find(p == 1);
if any(k)
    tmp  = Inf;
    x(k) = tmp(ones(size(k))); 
end

k=find(b > 0 & p > 0  &  p < 1);
if any(k),
    pk = p(k);
    bk = b(k);
    x(k) = sqrt((-2*bk .^ 2) .* log(1 - pk));
end


