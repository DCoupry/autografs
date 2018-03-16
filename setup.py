# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='AuToGraFS',
    version='2.0-alpha',
    description='Generator for topological frameworks and chemical structures.',
    long_description=readme,
    author='Damien Coupry',
    author_email='damien.coupry@uni-leipzig.de',
    url='https://github.com/DCoupry/autografs',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
