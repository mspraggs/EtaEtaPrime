import numpy as np
import fileio

def read2pt(exact_filename, sloppy_filename, time_extent, num_src):
    """Reads in the exact and sloppy results from the supplied files
    and does the AMA
    
    :param exact_filename: The file containing the exact results
    :type exact_filename: :class:`str`
    :param sloppy_filename: The file containing the sloppy results
    :type sloppy_filename: :class:`str`
    :param time_extent: The correlator time extent
    :type time_extent: :class:`int`
    :param num_src: The number of sources
    :type num_src: :class:`int`
    """
    
    exact_raw_data = fileio._load_data(exact_filename)
    # Convert the last two columns into a complex value
    # We reshape so the first index in the array corresponds to the source
    exact_correlators \
      = np.reshape(exact_raw_data[:, 2] + 1j * exact_raw_data[:, 3],
                   (num_src, time_extent))
    
    # Move every num_srcth datum to a list of sources file
    exact_t_src = np.int64(exact_raw_data[::time_extent, 0])
    
    # Check that there are time_extent * num_src data
    if exact_correlators.size != time_extent * num_src:
        raise IOError("Expected {} rows in exact data file, found {}"
                      .format(time_extent * num_src, exact_correlators.size))
    
    # Here we reshape the correlator into a 2d array of shape
    # num_src x time_extent and then average over the sources
    # (axis zero of the new array)
    exact_source_average = np.mean(exact_correlators, axis=0)
    
    # Now read in the sloppies
    sloppy_raw_data = fileio._load_data(sloppy_filename)
    sloppy_correlators \
      = np.reshape(sloppy_raw_data[:, 2] + 1j * sloppy_raw_data[:, 3],
                   (time_extent, time_extent))
    
    sloppy_source_average = np.mean(sloppy_correlators, axis=0)
    
    # Subtract relevant sloppy correlators from corresponding exact correlators
    for i, src in enumerate(exact_t_src):
        exact_correlators[i] -= sloppy_correlators[src]
        
    residual_source_average = np.mean(exact_correlators, axis=0)
    
    ama_source_average = residual_source_average + sloppy_source_average
    
    return exact_source_average, sloppy_source_average, \
      residual_source_average, ama_source_average
