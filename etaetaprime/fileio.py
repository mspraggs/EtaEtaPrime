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
