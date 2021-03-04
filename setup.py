from codecs import open
import os
from setuptools import setup, find_packages


path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(path, 'README.md'), encoding='utf-8') as f:
    desc = f.read()

exec(open(os.path.join(path, 'vistrans', 'version.py')).read())

setup(
    name='vistrans',
    version=__version__,
    description='Unofficial implementations of transfomers models for vision.',
    long_description=desc,
    long_description_content_type='text/markdown',
    url='https://github.com/nachiket273/VisTrans',
    author='Nachiket Tanksale',
    author_email='nachiket.tanksale@gmail.com',
    classifiers=[
        'Intended Audience :: Education',
        'License :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Engineering',
        'Topic :: Engineering :: Artificial Intelligence'
    ],
    keywords='pytorch transformer vision python deep-learning',
    packages=find_packages(exclude=[]),
    include_package_data=True,
    install_requires=['torch >= 1.4', 'torchvision'],
    python_requires='>=3.6',
    zip_safe=False
)
