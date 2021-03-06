import sys, os
import itertools
import numpy as np
import numpy.random as npr
sys.path.insert(0, os.path.abspath('../..'))

import etaetaprime
from etaetaprime import fileio
from etaetaprime import correlators

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
        assert data.shape == (9216, 4)
        
        data = fileio.load_data("{}/disconnected_test_data".format(data_dir))
        assert data.shape == (96, 5)
        
    def test_load_correlators(self):
        
        data = fileio.load_correlators("{}/connected_test_data".format(data_dir))
        
        assert data.shape == (9216, 3)
        
    def test_load_traces(self):
        
        data = fileio.load_traces("{}/disconnected_test_data".format(data_dir))

        assert data.shape == (96, 3)
        
    def test_split_traces(self):
        
        data = np.zeros((10, 3), dtype=np.complex)
        
        data[:, 0] = np.arange(10)
        data[3, 1] = 2 + 3j
        data[4, 1] = 2 - 1j
        random = npr.rand(10) + 1j * npr.rand(10)
        data[:, 2] = random
        
        split = fileio.split_traces(data)
        
        assert (split[0] == np.array([3, 4])).all()
        assert (split[1] == np.arange(10)).all()
        assert (split[2] == np.array([2 + 3j, 2 - 1j])).all()
        assert (split[3] == random).all()
        
class TestCorrelators:

    def test_cython_diff(self):
        
        t1 = np.arange(4)
        t2 = np.array([0, 2])
        
        expected_output = np.zeros((4, 2), dtype=int)
        expected_output[:, 0] = (t1 - t2[0]) % 4
        expected_output[:, 1] = (t1 - t2[1]) % 4
        
        diffs = correlators.combinatorics.diff(t1, t2, 4)
        
        assert (diffs == expected_output).all()

    def test_cython_av_prods(self):
        
        tolerance = 1e-8
        
        t1 = np.arange(4)
        t2 = np.array([0, 2])
        
        diffs = np.zeros((4, 2), dtype=int)
        diffs[:, 0] = (t1 - t2[0]) % 4
        diffs[:, 1] = (t1 - t2[1]) % 4

        prods = npr.rand(4, 2)
        
        av_prods = correlators.combinatorics.av_prods(t1, diffs, prods)
        
        for i in xrange(4):
            assert np.abs(av_prods[i].real
                          - (prods[i, 0] + prods[(i + 2) % 4, 1]) / 2) \
                          < tolerance
        
    def test_combine_traces(self):

        T = 96
        
        tolerance = 1e-8 * np.ones(T)
        
        trace1 = npr.rand(T)
        trace2 = npr.rand(6)
        
        combined_trace \
          = correlators.combine_traces(trace1, trace2,
                                       np.arange(T),
                                       np.arange(20, 44, 4))
        expected_trace = np.zeros(T)
        
        outer_prod = np.outer(trace1, trace2)
        
        for i in xrange(T):
            for k, j in enumerate(xrange(20, 44, 4)):
                expected_trace[i] \
                  += outer_prod[(i + j) % T, k]
            expected_trace[i] /= 6
                              
        assert (combined_trace[0].real == np.arange(T)).all()
        assert (np.abs(combined_trace[1].real - expected_trace)
                < tolerance).all()

    def test_ama(self):
        
        a = npr.rand(10)
        b = npr.rand(10)
        c = npr.rand(10)
        
        result = correlators.ama(a, b, c)
        expected = a - b + c
        
        assert (result - expected < 1e-10 * np.ones(10)).all()
        
        b = np.zeros((2, 10))
        b[0] = np.arange(10)
        b[1] = npr.rand(10)
        
        result = correlators.ama(a, b, c)
        expected = a - b[1] + c
        
        assert (result - expected < 1e-10 * np.ones(10)).all()
        
    def test_parse_connected(self):
        
        exact_data = np.zeros((30, 3))
        exact_data[:, 0] = np.arange(30) / 10 * 3
        exact_data[:, 1] = np.arange(30) % 10
        exact_data[:, 2] = npr.rand(30)
        
        sloppy_data = np.zeros((100, 3))
        sloppy_data[:, 0] = np.arange(100) / 10
        sloppy_data[:, 1] = np.arange(100) % 10
        sloppy_data[:, 2] = npr.rand(100)
        
        connected_correlators \
          = correlators.parse_connected(exact_data, sloppy_data)
          
        assert len(connected_correlators) == 3

        for connected_correlator in connected_correlators:
            assert connected_correlator.size == 10
        
    def test_parse_disconnected(self):
        
        exact_sources = -1 * np.ones(3, dtype=int)
        for i in range(3):
            x = 0
            while x in exact_sources:
                x = npr.randint(10)
                
            exact_sources[i] = x
        
        exact_data = np.zeros((10, 3))
        exact_data[:, 0] = np.arange(10)
        exact_data[exact_sources, 1] = npr.rand(3)
        exact_data[:, 2] = npr.rand(10)
        
        sloppy_data = np.zeros((10, 3))
        sloppy_data[:, 0] = np.arange(10)
        sloppy_data[:, 1:3] = npr.rand(10, 2)
        
        disconnected_correlators \
          = correlators.parse_disconnected(exact_data,
                                           sloppy_data,
                                           10)
          
        assert len(disconnected_correlators) == 3

        assert disconnected_correlators[0].size == 10
        assert disconnected_correlators[1].size == 20
        assert disconnected_correlators[2].size == 10
