# Use cython to speed up the conversion to a numpy array from a list
import pyximport
pyximport.install()
# Comment out the following line if not using the boost compiled file reader
#from fastfunctions import fastread
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
    
    # Could use fastread.read_csv here instead of _file_to_list, but at the
    # moment this affords no performance increase
    list_read_function = _file_to_list
    return converters.toarr(list_read_function(filename))
