import os, glob, unittest
import numpy as np
from SK_grid import Fast_Kurtogram
from SK_process import getBandwidthAndFrequency

def suite():
  suite = unittest.TestSuite()
  suite.addTest(KurtogramTests('test_MatlabGrid'))
  suite.addTest(KurtogramTests('test_BandwidthFrequency'))
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
    # calculate the kurtogram
    self.grid, self.level_w, self.freq_w = Fast_Kurtogram(self.x, self.nlevel, self.Fs)

  def test_MatlabGrid(self):

    # Test you get the same output for the kurtogram grid as the original Matlab code does
    matlab_grid = np.fromfile('test_data/matlab_grid.np').reshape(16,768)


    # do test
    np.testing.assert_allclose(self.grid,matlab_grid,atol=0.003)
    
  def test_BandwidthFrequency(self):

    level_index=11
    freq_index=24
    level_exp=6
    bw_exp=0.78125
    fc_exp=1.953125

    # get the bandwith and central frequency
    bw, fc = getBandwidthAndFrequency(self.nlevel,self.Fs, self.level_w, self.freq_w, level_index, freq_index)

    # do test
    self.assertAlmostEqual(bw,bw_exp)
    self.assertAlmostEqual(fc,fc_exp)
    

if __name__ == '__main__':

  import logging
  logging.basicConfig(level=logging.INFO, format='%(levelname)s : %(asctime)s : %(message)s')
 
  unittest.TextTestRunner(verbosity=2).run(suite())
 
