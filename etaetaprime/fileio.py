# Use cython to speed up the conversion to a numpy array from a list
import pyximport
pyximport.install()
# Comment out the following line if not using the boost compiled file reader
#from fastfunctions import fastread
from fastfunctions import converters

def _file_to_list(filename):
    """Reads the supplied file into a compound list"""
    
    with open(filename) as f:
        # This is messy, but it shaves a few milliseconds of the function runtime
        out = [[float(line[:4]), float(line[4:8]),
                float(line[8:26]), float(line[26:])]
                for line in f.readlines()]
        
    return out

def _load_data(filename):
    """Reads the supplied file into a numpy array"""
    
    # Could use fastread.read_csv here instead of _file_to_list, but at the
    # moment this affords no performance increase
    list_read_function = _file_to_list
    return converters.toarr(list_read_function(filename))
