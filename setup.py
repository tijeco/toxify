

from setuptools import setup

setup(name='toxify',
      version='0.1',
      description='The toxify joke in the world',
      include_package_data=True,
      url='http://github.com/tijeco/toxify',
      author='Jeff Cole',
      scripts = ['bin/toxify'],
      author_email='coleti16@students.ecu.edu',
      license='GNU',
      packages=['toxify'],
      zip_safe=False)
