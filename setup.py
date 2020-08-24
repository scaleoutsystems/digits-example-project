from setuptools import setup, find_packages

setup(
    name='src',
    version='0.10',
    packages=find_packages(exclude=["tests", "tests.*"]),

   )