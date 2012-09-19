#compare grids

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat

from SK_grid import Fast_Kurtogram

v1 = loadmat("test_data/VOIE1.mat")
x = v1['v1']
Fs = 100
nlevel= 8
grid, Level_w, freq_w = Fast_Kurtogram(x, nlevel, Fs)
    

#loading and comparing with matlab:
matlab = np.fromfile('test_data/matlab_grid.np').reshape(16,768)

ax = plt.subplot(131)
plt.imshow(matlab,aspect='auto',interpolation='none')
plt.colorbar()
plt.title('Matlab')

plt.subplot(132,sharex=ax,sharey=ax)
plt.imshow(grid,aspect='auto',interpolation='none')
plt.colorbar()
plt.title('Python')

plt.subplot(133,sharex=ax,sharey=ax)
diff = np.abs(100*(grid - matlab) / matlab)
plt.imshow(diff,aspect='auto',interpolation='none')
plt.colorbar()
plt.title('|100*(Python - Matlab) / Matlab|')

plt.show()
