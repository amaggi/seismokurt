import unittest
import os, glob
import logging

def suite():
  suite = unittest.TestSuite()
  suite.addTest(SetupTests('test_setup'))
  return suite

def setUpModule():

      pass

   
class SetupTests(unittest.TestCase):

  def test_setup(self):
    self.assertTrue(True)

if __name__ == '__main__':

  import logging
  import test_example
  logging.basicConfig(level=logging.INFO, format='%(levelname)s : %(asctime)s : %(message)s')
 

  suites={}
  suites['ThisSuite']=suite()
  suites['ExampleSuite']=test_example.suite()

  alltests=unittest.TestSuite(suites.values())

  unittest.TextTestRunner(verbosity=2).run(alltests)
 
