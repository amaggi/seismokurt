import os, glob, unittest
import numpy as np
from SK_grid import Fast_Kurtogram

def suite():
  suite = unittest.TestSuite()
  suite.addTest(KurtogramTests('test_MatlabGrid'))
  return suite

#@unittest.skip('Not bothering with this test')
class KurtogramTests(unittest.TestCase):

  def setUp(self):
    # read the input data for the test - this is the original data from Antoni
    # input data come in the form of a Matlab file
    from scipy.io.matlab import loadmat

    v1 = loadmat("test_data/VOIE1.mat")
    self.x = v1['v1']
    self.Fs = 100
    self.nlevel = 8

  def test_MatlabGrid(self):

    # Test you get the same output for the kurtogram grid as the original Matlab code does
    matlab_grid = np.fromfile('test_data/matlab_grid.np').reshape(16,768)

    # calculate the kurtogram
    grid, Level_w, freq_w = Fast_Kurtogram(self.x, self.nlevel, self.Fs)

    # do test
    np.testing.assert_allclose(grid,matlab_grid,atol=0.003)
    

if __name__ == '__main__':

  import logging
  logging.basicConfig(level=logging.INFO, format='%(levelname)s : %(asctime)s : %(message)s')
 
  unittest.TextTestRunner(verbosity=2).run(suite())
 
