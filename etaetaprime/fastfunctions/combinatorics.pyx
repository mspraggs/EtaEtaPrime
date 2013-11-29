import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def diff(a, b, T):
    cdef int i, j, h=a.size, w=b.size
    
    cdef np.ndarray[np.int_t, ndim=2] new = np.empty((h, w), dtype=np.int)
    
    for i in xrange(h):
        for j in xrange(w):
            new[i, j] = a[i] - b[j]
            if new[i, j] < 0:
                new[i, j] = new[i, j] + T
            
    return new

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def av_prods(ts, diffs, prods):
    cdef int i, j, k, num_t = ts.size, h = diffs.shape[0], w = diffs.shape[1]
    
    cdef np.ndarray[np.complex128_t, ndim=1] new \
      = np.zeros(num_t, dtype=np.complex128)
    
    cdef np.ndarray[np.int_t, ndim=1] frequency \
      = np.zeros(num_t, dtype=np.int)
    
    cdef np.ndarray[np.int_t, ndim=1] ts_c = ts
    cdef np.ndarray[np.int_t, ndim=2] diffs_c = diffs
    
    for i in xrange(num_t):
        for j in xrange(h):
            for k in xrange(w):
                if diffs_c[j, k] == ts_c[i]:
                    new[i] = new[i] + prods[j, k]
                    frequency[i] += 1
                    
    return new / frequency
