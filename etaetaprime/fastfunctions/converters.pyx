# Taken from code by IanH on stack overflow
# http://stackoverflow.com/questions/17973507/ ...
# ... why-is-converting-a-long-2d-list-to-numpy-array-so-slow
from numpy cimport ndarray as ar
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def toarr(xy):
    cdef int i, j, h=len(xy), w=len(xy[0])
    cdef ar[double,ndim=2] new = np.empty((h,w))
    for i in xrange(h):
        for j in xrange(w):
            new[i,j] = xy[i][j]
    return new
