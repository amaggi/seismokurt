import os, glob, unittest

def suite():
  suite = unittest.TestSuite()
  suite.addTest(ExampleTests('test_dummy'))
  return suite

#@unittest.skip('Not bothering with this test')
class ExampleTests(unittest.TestCase):

  def setUp(self):
    pass

  def test_dummy(self):
    self.assertTrue(True)

if __name__ == '__main__':

  import logging
  logging.basicConfig(level=logging.INFO, format='%(levelname)s : %(asctime)s : %(message)s')
 
  unittest.TextTestRunner(verbosity=2).run(suite())
 
