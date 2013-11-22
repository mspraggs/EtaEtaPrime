#from numpy cimport ndarray as ar
from numpy import unique
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def diff(a, b):
    cdef int i, j, h=a.size, w=b.size
    
    cdef np.ndarray[np.int32_t, ndim=2] new = np.empty((h, w), dtype=np.int32)
    
    for i in xrange(h):
        for j in xrange(w):
            new[i, j] = a[i] - b[j]
            if new[i, j] < 0:
                new[i, j] = -new[i, j]
            
    return new
