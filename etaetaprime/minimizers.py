import itertools
import numpy as np
import IPython

def grid_search(function, start_range, args=[], grid_points=10, tolerance=1e-6,
                max_iterations=1000):
    """Finds the minimimum value of the supplied function in the specified
    range
    
    :param function: The function to minimize
    :type function: :class:`function`
    :param start_range: Specification of the start and end range for each parameter
    :type start_range: :class:`list` or :class:`tuple`
    :param grid_points: Number of points to use in the grid
    :type grid_point: :class:`int`
    :param tolerance: The required tolerance in the result
    :type tolerance: :class:`float`
    :param max_iterations: The maximum number of iterations
    :type max_iterations: :class:`int`
    :returns: :class:`list`
    """
    
    if type(start_range[0]) != list and type(start_range[0]) != tuple:
        start_range = [start_range]

    iteration = 0
    old_parameters = 100 * np.ones(len(start_range))
    new_parameters = np.zeros(len(start_range))
    
    if type(grid_points) != list:
        grid_points = [grid_points for i in (start_range)]
    
    while iteration < max_iterations \
      and (np.abs(new_parameters - old_parameters)
           > tolerance * np.abs(old_parameters)).any():
        linspaces = [np.linspace(r[0], r[1], g)
                     for r, g in zip(start_range, grid_points)]
        grid_step = [(x[1] - x[0]) / (y - 1)
                     for x, y in zip(start_range, grid_points)]
        grid_values = list(itertools.product(*linspaces))
        old_parameters = new_parameters.copy()
    
        function_values = np.array(map(lambda x: function(x, *args),
                                       grid_values))
        
        new_minimum = function_values.min()
        min_position = np.where(function_values == new_minimum)[0][0]
        grid_value = grid_values[min_position]
        new_parameters = np.array(grid_value)
        start_range = [[point - diff, point + diff]
                       for point, diff in zip(grid_value, grid_step)]
        
        iteration += 1
        
    if iteration == max_iterations:
        print("Warning: max iterations reached.")
        
    return grid_value
