import sys
sys.path.insert(0, '/home/matthew/Programming/Python/pyQCD')
import numpy as np
import pyQCD

def constrained_two_state_fit(twopoint, correlator, fit_range, b_init,
                              b_est=None, b_err_est=None, stddev=None):
    """Performs a constrained fit on the supplied two-point function
    
    :param twopoint: TwoPoint object containing the correlators
    :type twopoint: :class:`TwoPoint` of :class:`BareTwoPoint`
    :param correlator: Specification of the correlator to fit
    :type correlator: :class:`str`
    :param fit_range: The ranger of times over which to perform the fit
    :type fit_range: :class:`list` with two elements
    :param b_init: Initial estimate for the fitted parameters
    :type b_init: :class:`list`
    :param b_est: Estimated central value for use when performing a constrained fit
    :type b_est: :class:`list`
    :param b_err_est: Estimated standard deviation for use when performing a constrained fit
    :type b_err_est: :class:`list`
    :param stddev: The standard deviation in the specified correlator
    :type stddev: :class:`numpy.ndarray`
    :returns: :class:`list` containing the fitted masses and square amplitudes
    """

    t = np.arange(twopoint.T)
    correlator = getattr(twopoint, correlator)
    
    if stddev == None:
        stddev = np.ones(twopoint.T)
    
    x = t[range(*fit_range)]
    y = t[range(*fit_range)]
    err = stddev[range(*fit_range)]
    
    fit_function \
      = lambda b, t: b[0] * np.exp(-b[1] * t) + b[2] * np.exp(-b[3] * t)

    result = spop.minimize(pyQCD.TwoPoint._chi_squared, b_init,
                           args=(x, y, err, fit_function, b_est, b_err_est),
                           method="Powell")
    
    result_values = result['x']
    result_values[0] /= 2 * result_values[1]
    result_values[2] /= 2 * result_values[3]
    
    return result_values

def separate_fits(twopoint, correlator, first_fit_range, second_fit_range,
                  stddev=None):
    """Fits the ground state and first excited state by performing a one-state
    fit on the correlator before substracting the result and performing a second
    fit.
    
    :param twopoint: The two-point object containing the correlator
    :type twopoint: :class:`TwoPoint` or :class:`BareTwoPoint`
    :param correlator: The correlator to perform the fit over
    :type correlator: :class:`str`
    :param first_fit_range: The range of timeslices over which to perform the first fit
    :type first_fit_range: :class:`list`
    :param second_fit_range: The range of timeslices over which to perform the second fit
    :type first_fit_range: :class:`list`
    :param stddev: The standard deviation in the specified correlator
    :type stddev: :class:`numpy.ndarray`
    :returns: :class:`list` containing the fitted masses and square amplitudes
    """

    fit_function = lambda b, t: b[0] * np.exp(-b[1] * t)
    
    t = np.arange(twopoint.T)
    correlator = getattr(twopoint, correlator)
    
    fit_results = [0, 0, 0, 0]
    
    timeslices = range(*first_fit_range)
    
    for i in xrange(2):
        x = t[timeslices]
        y = correlator[timeslices]
        yerr = correlator[timeslices]

        result = spop.minimize(pyQCD.TwoPoint._chi_squared, [correlator[0], 1.0],
                               args=(x, y, yerr, fit_function), method="Powell")
        
        fit_results[2 * i] = result['x'][0] / (2 * result['x'][1])
        fit_results[2 * i + 1] = result['x'][1]
        
        correlator = correlator - fit_function(result['x'], t)
        timeslices = range(*second_fit_range)
        
    return fit_results

def excited_effmass(twopoint, fit_function, args):
    """Computes the effective mass for the excited state based on the fit
    results from fit_function
    
    :param twopoint: The two-point object containing the correlator to calculate the effective mass from
    :type twopoint: :class:`TwoPoint` or :class:`BareTwoPoint`
    :param fit_function: The function used to extract the ground and excited states
    :type fit_function: :class:`function`
    :param args: The arguments for the supplied fitting function, the first of which should specify the correlator to use
    :type args: :class:`list`
    :returns: :class:`numpy.ndarray`
    """

    fitting_results = fit_function(twopoint, *args)
    
    excited_correlator = getattr(twopoint, args[0]) \
      - fitting_results[0] * np.exp(-fitting_results[1] * np.arange(twopoint.T))

    return np.log(np.abs(excited_correlator / np.roll(excited_correlator, -1)))
