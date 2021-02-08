from setuptools import setup, find_packages


setup(
    name='VisTrans',
    version='0.0.1',
    description='Unofficial implementations of transfomers models for vision.',
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
    keywords='pytorch transformer vision',
    packages=find_packages(exclude=[]),
    include_package_data=True,
    install_requires=['torch >= 1.4', 'torchvision'],
    python_requires='>=3.6',
    zip_safe=False
)
