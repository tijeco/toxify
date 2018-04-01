#!/usr/bin/env python3

from setuptools import setup

setup(name='toxify',
      version='0.1',
      description='The toxify joke in the world',
      url='http://github.com/tijeco/toxify',
      author='Flying Circus',
      scripts = ['bin/toxify'],
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['toxify'],
      zip_safe=False)
