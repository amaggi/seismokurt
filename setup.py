#!/usr/bin/env python

from distutils.core import setup

setup(name='seismokurt',
	version='0.1.0',
	description='Kurtosis for seismology',
	author='Thomas Lecocq and Alessia Maggi',
	author_email='alessia.maggi@unistra.fr',
	url='http://github.com/amaggi/waveloc',
	packages=['seismokurt'],
	package_dir = {'seismokurt' : 'src'},
)
