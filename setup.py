#!/usr/bin/env python
import setuptools

VERSION = '0.0.1'


# Read in requirements
with open('docker/requirements.txt', 'r') as rf:
    # Setuptools can't deal with git+https references
    required_packages = [p for p in rf.readlines() if not p.startswith('git+https')]


# Fill in all this for the specific project
setuptools.setup(
    name='gray_zone_assessment',
    version=VERSION,
    description='Project description',
    author='Andreanne Lemay',
    platforms=['Linux'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    # Requirements match exactly the dockerfile requirements
    install_requires=required_packages,
    # Everything in the packages resources directory is added as a resource
    # for inclusion wherever the package is installed
    # This is intended for config files
    package_data={
        '': ['resources/*']
    },
    # Set up the main entrypoint script for the entire project
    # This defines the ml_proj command-line command
    entry_points={
        'console_scripts': [
            'gray_zone=gray_zone.main:main'
        ]
    }
)
