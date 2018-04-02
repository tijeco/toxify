

from setuptools import setup
CURRENT_ABSOLUTE = os.path.dirname(os.path.realpath(__file__))
# replace LIB_PATH definition
LIB_PATH = [libpath['find_lib_path']()[0].replace(CURRENT_ABSOLUTE + '/', '')]
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
