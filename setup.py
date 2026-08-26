#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# catkin-aware setup.py: makes `sc_ros_empathic` (the control-law library
# under src/) importable as a regular Python package by the scripts in
# scripts/, once CMakeLists.txt calls catkin_python_setup().
d = generate_distutils_setup(
    packages=['sc_ros_empathic'],
    package_dir={'': 'src'},
)

setup(**d)
