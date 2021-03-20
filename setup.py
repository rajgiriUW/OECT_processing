from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='oect',
    version='0.0.1',
    description='Organic Electrochemical Transistor Processing',
    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Rajiv Giridharagopal',
    author_email='rgiri@uw.edu',
    license='MIT',
	  url='https://github.com/rajgiriUW/OECT_processing',

    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    test_suite='pytest',
    install_requires=['numpy >=1.18',
                      'pandas >=1.1.3',
                      'scipy >=1.5.2',
                      'matplotlib',
                      'seaborn',
                      'h5py',
                      'configparser'
                      ],


)
