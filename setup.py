#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Karin van Garderen",
    author_email='k.vangarderen@erasmusmc.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Automatic delineation of relevant structures for the analysis of GLASS imaging, using deep neural networks.",
    entry_points={
        'console_scripts': [
            'glassimaging=glassimaging.cli:cli',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    include_package_data=True,
    keywords='glassimaging',
    name='glassimaging',
    packages=find_packages(include=['glassimaging', 'glassimaging.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/karinvangarderen/glassimaging',
    version='0.1.0',
    zip_safe=False,
)
