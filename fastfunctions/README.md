fastfunctions module
====================

This module contains a series of functions that are designed to be fast.
The file converters.pyx contains a function toarr that uses cython to
speed up the conversion of lists to numpy arrays.

boost::python is used to wrap C/C++ code to load text files into python
lists. Currently this affords no speed up over an equivalent python
implementation, so it is not used.

To build the fastread module you'll need cmake 2.
