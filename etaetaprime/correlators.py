import ama
import numpy as np
import fileio
import itertools
import pyximport
pyximport.install()
from fastfunctions import combinatorics

def combine_traces(first_trace, second_trace, first_timeslices=None,
                   second_timeslices=None):
    """Computes a correlator from a pair of traces and
    their corresponding timeslices
    
    :param first_trace: The first set of traces
    :type first_trace: :class:`numpy.ndarray`
    :param second_trace: The second set of traces
    :type second_trace: :class:`numpy.ndarray`
    :param first_timeslices: The set of timeslices over which the
    first traces are taken
    :type first_timeslices: :class:`numpy.ndarray` or :class:`list`
    :param second_timeslices: The set of timeslices over which the
    second traces are taken
    :type first_timeslices: :class:`numpy.ndarray` or :class:`list`
    """
    
    # If no timeslices are given, assume the form they take
    if first_timeslices == None:
        first_timeslices = np.arange(first_trace.size)
    if second_timeslices == None:
        second_timeslices = np.arange(second_trace.size)
        
    diffs = combinatorics.diff(first_timeslices, second_timeslices)
    prods = np.outer(first_trace, second_trace)
    timeslices = np.unique(diffs)
    correlator = np.zeros(timeslices.size, dtype=np.complex)
    
    for i, t in enumerate(timeslices):
        correlator[i] = np.sum(prods[diffs == t])
        
    return np.array([timeslices, correlator])
