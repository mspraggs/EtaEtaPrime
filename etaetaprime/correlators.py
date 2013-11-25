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
        
    # Make sure the supplied timeslices are in int64 format
    first_timeslices = np.int64(first_timeslices)
    second_timeslices = np.int64(second_timeslices)
    
    # Get all possible differences between the various timeslice pairings
    # (This works like an outer product, returning a 2d array)
    diffs = combinatorics.diff(first_timeslices, second_timeslices)
    # Get the products of all the correlators
    prods = np.outer(first_trace, second_trace)
    # Get the output timeslices by determining the uniqe list of diffs
    timeslices = np.unique(diffs)
    # The output array
    correlator = np.zeros(timeslices.size, dtype=np.complex)
    
    # Now loop over the unique timeslices and sum the corresponding products
    for i, t in enumerate(timeslices):
        correlator[i] = np.sum(prods[diffs == t])
    # Turn the result into a numpy array
    return np.array([timeslices, correlator])
