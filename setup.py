from setuptools import setup

setup(name='toxify',
      version='0.1.78',
      description='The toxify joke in the world',
      url='http://github.com/tijeco/toxify',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      entry_points = {'console_scripts': ['toxify=toxify.cli:main'],},
      license='MIT',
      packages=['toxify'],
      include_package_data=True,
      zip_safe=False)
