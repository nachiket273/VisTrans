from setuptools import setup, find_packages
from codecs import open
from os import path

curr = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(curr, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('VisTrans/version.py').read())
setup(
    name='VisTrans',
    version=__version__,
    description='(Unofficial) PyTorch Image Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='git@github.com:nachiket273/VisTrans-sdk-package.git',
    author='Nachiket Tanksale',
    author_email='nachiket.tanksale@gmail.com',
    classifiers=[
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='pytorch transformer vision',
    packages=find_packages(exclude=[]),
    include_package_data=True,
    install_requires=['torch >= 1.4', 'torchvision'],
    python_requires='>=3.6',
)
