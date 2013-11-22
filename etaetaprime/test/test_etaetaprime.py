import sys, os
import itertools
import numpy as np
import numpy.random as npr
sys.path.insert(0, os.path.abspath('../..'))
import etaetaprime
from etaetaprime import fileio

data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

class TestFileIO:
    
    def test_file_to_list(self):
        
        output = fileio.file_to_list("{}/connected_test_data".format(data_dir))
        
        assert len(output) == 96**2
        assert len(output[0]) == 4
        
        for i, x in enumerate(itertools.product(range(96), range(96))):
            assert output[i][0] == float(x[0])
            assert output[i][1] == float(x[1])
            
    def test_cython_converter(self):
        
        nrows = npr.randint(100)
        ncols = npr.randint(100)
        input_list = [[npr.rand() for i in xrange(ncols)]
                      for j in xrange(nrows)]
            
        numpy_array = fileio.converters.toarr(input_list)
        
        assert numpy_array.shape == (nrows, ncols)
        
        for i, j in itertools.product(range(nrows), range(ncols)):
            assert numpy_array[i, j] == input_list[i][j]
                
    def test_load_data(self):
        
        data = fileio.load_data("{}/connected_test_data".format(data_dir))
