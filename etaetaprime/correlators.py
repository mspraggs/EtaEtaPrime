import numpy as np
import fileio
import itertools
import pyximport
pyximport.install(reload_support=True)
from fastfunctions import combinatorics

def combine_traces(first_trace, second_trace, first_timeslices=None,
                   second_timeslices=None, num_timeslices=None):
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
    :param num_timeslices: The lattice temporal extent
    :type num_timeslices: :class:`int`
    
    :returns: :class:`numpy.ndarray` containing timeslices and correlator values
    """
    
    # If no timeslices are given, assume the form they take
    if first_timeslices == None:
        first_timeslices = np.arange(first_trace.size)
    if second_timeslices == None:
        second_timeslices = np.arange(second_trace.size)
        
    # Make sure the supplied timeslices are in int64 format
    first_timeslices = np.int64(first_timeslices.real)
    second_timeslices = np.int64(second_timeslices.real)
    
    if num_timeslices == None:
        num_timeslices = max(np.max(first_timeslices),
                             np.max(second_timeslices)) + 1
    
    # Get all possible differences between the various timeslice pairings
    # (This works like an outer product, returning a 2d array)
    diffs = combinatorics.diff(first_timeslices, second_timeslices,
                               num_timeslices)

    # Get the products of all the correlators
    prods = np.outer(first_trace, second_trace)
    # Get the output timeslices by determining the uniqe list of diffs
    timeslices = np.unique(diffs)
    
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

def parse_disconnected(exact_data, sloppy_data, num_timeslices):
    """Takes the exact and sloppy traces supplied by load_traces and creates
    the required correlators for the AMA
    
    :param exact_data: An array of exact traces loaded by the load_traces function in fileio
    :type exact_data: :class:`numpy.ndarray`
    :param sloppy_data: An array of sloppy traces loaded by the load_traces function in fileio
    :type sloppy_data: :class:`numpy.ndarray`
    :param num_timeslices: The temporal extent of the lattice
    :type num_timeslices: :class:`int`
    
    :returns: :class:`tuple` of :class:`numpy.ndarray`
    """

    # We *should* have traces for all sloppy timeslices, so don't need
    # to bother with the split here
    sloppy_timeslices = np.int64(sloppy_data[:, 0].real)
    sloppy_trace_1 = sloppy_data[:, 1]
    sloppy_trace_2 = sloppy_data[:, 2]
    
    exact_timeslices_1, exact_timeslices_2, exact_trace_1, exact_trace_2 \
      = fileio.split_traces(exact_data)
      
    # Now do the combinatorics for the traces, starting with exact
    exact_correlator = combine_traces(exact_trace_1,
                                      exact_trace_2,
                                      exact_timeslices_1,
                                      exact_timeslices_2,
                                      num_timeslices)
    
    sloppy_trace_1_r = sloppy_trace_1[exact_timeslices_1]
    
    sloppy_restricted_correlator = combine_traces(sloppy_trace_1_r,
                                                  sloppy_trace_2,
                                                  exact_timeslices_1,
                                                  exact_timeslices_2,
                                                  num_timeslices)
    
    sloppy_correlator = combine_traces(sloppy_trace_1,
                                       sloppy_trace_2,
                                       sloppy_timeslices,
                                       sloppy_timeslices,
                                       num_timeslices)

    return exact_correlator[1], sloppy_restricted_correlator[1], sloppy_correlator[1]

def ama(exact_correlator, sloppy_restricted_correlator, sloppy_correlator):
    """Takes the three correlators required for the ama and does the
    subtraction
    
    :param exact_correlator: The source-averaged exact correlator
    :type exact_correlator: :class:`numpy.ndarray`
    :param sloppy_restricted_correlator: The source-averaged restricted sloppy correlator
    :type sloppy_restricted_correlator: :class:`numpy.ndarray`
    :param sloppy_correlator: The source-averaged sloppy correlator
    :type sloppy_correlator: :class:`numpy.ndarray`
    
    :returns: :class:`numpy.ndarray`
    """

    return exact_correlator - sloppy_restricted_correlator + sloppy_correlator

def run_one(exact_file, sloppy_file, num_timeslices=96, connected=False):
    """Applied the AMA procedure to the correlators or traces in the specified files
    
    :param exact_file: The file containing the exact correlators or traces
    :type exact_file: :class:`str`
    :param sloppy_file: The file containing the sloppy correlators or traces
    :type sloppy_file: :class:`str`
    :param connected: Determines whether diagram is connected or not
    :type connected: :class:`bool`
    
    :returns: :class:`numpy.ndarray` containing the correlator
    """
    
    if connected:
        exact_data = fileio.load_correlators(exact_file)
        sloppy_data = fileio.load_correlators(sloppy_file)
        
        correlators = parse_connected(exact_data, sloppy_data)
        # Pass the tuple result from the previous function as an *args expression
        # so each item in the tuple becomes a separate argument
        correlator_ama = ama(*correlators)
        
        return [correlator_ama] + list(correlators)
    
    else:
        exact_data = fileio.load_traces(exact_file)
        sloppy_data = fileio.load_traces(sloppy_file)
        
        correlators = parse_disconnected(exact_data, sloppy_data, num_timeslices)
        correlator_ama = ama(*correlators)
        
        return [correlator_ama] + list(correlators)
        
def run_all(exact_folder, sloppy_folder, input_prefix, output_prefix, start, stop, step,
            num_timeslices=96, connected=False):
    """Applies the all-mode average to all files in the specified directories
    and saves the results in a set of numpy binaries
    
    :param exact_folder: The folder containing the exact results
    :type exact_folder: :class:`str`
    :param sloppy_folder: The folder containing the sloppy results
    :type sloppy_folder: :class:`str`
    :param input_prefix: The common prefix used for the input data files in each folder
    :type input_prefix: :class:`str`
    :param output_prefix: The common prefix used for the output data files in each folder
    :type output_prefix: :class:`str`
    :param start: The first configuration number used at the end of the input filename
    :type start: :class:`int`
    :param stop: The last configuration number used at the end of the input filename
    :type stop: :class:`int`
    :param step: The difference between consecutive configuration numbers
    :type step: :class:`int`
    :param num_timeslices: The temporal extent of the lattice
    :type num_timeslices: :class:`int`
    :param connected: Determines whether the associated diagram is connected
    :type connected: :class:`bool`
    """
    
    for i in xrange(start, stop + step, step):
        try:
            correlator \
              = run_one("{}/{}.{}".format(exact_folder, input_prefix, i),
                        "{}/{}.{}".format(sloppy_folder, input_prefix, i),
                        num_timeslices, connected)
        
            np.save("{}{}".format(output_prefix, i), correlator)
            
        except IOError:
            print("Results for configuration {} missing; skipping.".format(i))
