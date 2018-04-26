
# from distutils.core import setup
try:
    from setuptools import *
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
finally:
    from setuptools import *
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
      include_package_data = True,
      zip_safe=False)
