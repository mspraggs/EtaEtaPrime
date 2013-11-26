import ama
import numpy as np
import fileio
import itertools
import pyximport
pyximport.install(reload_support=True)
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
    first_timeslices = np.int64(first_timeslices.real)
    second_timeslices = np.int64(second_timeslices.real)
    
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
    correlator = combinatorics.av_prods(timeslices, diffs, prods)
    # Turn the result into a numpy array
    return np.array([timeslices, correlator])

def parse_connected(exact_data, sloppy_data):
    """Takes the exact and sloppy connected correlators and generates the
    correlators required for the AMA process
    
    :param exact_data: The array containing the exact results
    :type exact_data: :class:`numpy.ndarray`
    :param sloppy_data: The file containing the sloppy results
    :type sloppy_data: :class:`numpy.ndarray`
    
    :returns: :class:`tuple` of :class:`numpy.ndarray`
    """
    
    # We can grab the time_extent and the number of sources from the input
    # data by looking at the number of unique entries in each case
    time_extent = np.unique(np.int64(exact_data[:, 1])).size
    num_src = np.unique(np.int64(exact_data[:, 0])).size

    # Convert the last two columns into a complex value
    # We reshape so the first index in the array corresponds to the source
    exact_correlators \
      = np.reshape(exact_data[:, 2], (num_src, time_extent))
    
    # Move every num_srcth datum to a list of sources file
    exact_t_src = np.int64(exact_data[::time_extent, 0])
    
    # Check that there are time_extent * num_src data
    if exact_correlators.size != time_extent * num_src:
        raise ValueError("Expected {} rows in exact data file, found {}"
                         .format(time_extent * num_src, exact_correlators.size))

    # Average over the sources
    exact_source_average = np.mean(exact_correlators, axis=0)

    # Now look at the sloppies
    sloppy_correlators \
      = np.reshape(sloppy_data[:, 2], (time_extent, time_extent))
    
    sloppy_source_average = np.mean(sloppy_correlators, axis=0)

    sloppy_restricted_correlators = sloppy_correlators[exact_t_src]
    sloppy_restr_corr_src_av = np.mean(sloppy_restricted_correlators, axis=0)
    
    return exact_source_average, sloppy_restr_corr_src_av, sloppy_source_average
