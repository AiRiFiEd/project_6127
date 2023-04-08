from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='project_6127',
      version='0.0.04',
      description='Package for AI6127 Project',
      long_description=long_description,
      url='https://github.com/AiRiFiEd/project_6127',
      author='yqlim',
      author_email='yuanqing87@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)