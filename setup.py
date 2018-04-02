
# from distutils.core import setup
from setuptools import setup, find_packages
# replace LIB_PATH definition
setup(name='toxify',
      version='0.1',
      description='The toxify joke in the world',
      url='http://github.com/tijeco/toxify',
      author='Jeff Cole',
      scripts = ['bin/toxify'],
      author_email='coleti16@students.ecu.edu',
      license='GNU',
      packages=['toxify'],
      package_data={'toxify':['data/RaptorX_Property_Fast/']},
      include_package_data = True,
      zip_safe=False)
