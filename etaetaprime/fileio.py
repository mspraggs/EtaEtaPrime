# Use cython to speed up the conversion to a numpy array from a list
import pyximport
pyximport.install()
from fastfunctions import converters

def file_to_list(filename):
    """Reads the supplied file into a compound list
    
    :param filename: The file to import as a list
    :type filename: :class:`str`
    :returns: Compound :class:`list`
    """
    
    with open(filename) as f:
        out = [[float(x) for x in line.split()] for line in f.readlines()]
        
    return out

def load_data(filename):
    """Reads the supplied file into a numpy array, using cython to
    speed up the numpy array conversion
    
    :param filename: The file to import
    :type filename: :class:`str`
    :returns: :class:`numpy.ndarray`
    """

    return converters.toarr(file_to_list(filename))

def load_correlators(filename):
    """Loads the connected two point function in the specified file
    
    :param filename: The file from which to load the correlators
    :type filename: :class:`str`
    :returns: :class:`numpy.ndarray`
    """
    
    raw_data = load_data(filename)
    out = np.zeros((raw_data.shape[0], 3), dtype=np.complex)
    
    out[:, 0:2] = raw_data[:, 0:2]
    out[:, 2] = raw_data[:, 2] + 1j * raw_data[:, 3]
    
    return out

def load_traces(filename):
    """Loads the two traces in the specified file
    
    :param filename: The file from which to load the traces
    :type filename: :class:`str`
    :returns: :class:`numpy.ndarray`
    """
    
    raw_data = load_data(filename)
    out = np.zeros((raw_data.shape[0], 4), dtype=np.complex)
    
    out[:, 0] = raw_data[:, 0]
    out[:, 1] = raw_data[:, 1] + 1j * raw_data[:, 2]
    out[:, 2] = raw_data[:, 3] + 1j * raw_data[:, 4]

    return out
