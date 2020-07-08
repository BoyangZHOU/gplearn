import numpy as np
from joblib import wrap_non_picklable_objects

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


    def make_function(function, name, arity, wrap=True):
        """Make a function node, a representation of a mathematical relationship.

        This factory function creates a function node, one of the core nodes in any
        program. The resulting object is able to be called with NumPy vectorized
        arguments and return a resulting vector based on a mathematical
        relationship.

        Parameters
        ----------
        function : callable
            A function with signature `function(x1, *args)` that returns a Numpy
            array of the same shape as its arguments.

        name : str
            The name for the function as it should be represented in the program
            and its visualizations.

        arity : int
            The number of arguments that the `function` takes.

        wrap : bool, optional (default=True)
            When running in parallel, pickling of custom functions is not supported
            by Python's default pickler. This option will wrap the function using
            cloudpickle allowing you to pickle your solution, but the evolution may
            run slightly more slowly. If you are running single-threaded in an
            interactive Python session or have no need to save the model, set to
            `False` for faster runs.

        """
        if not isinstance(arity, int):
            raise ValueError('arity must be an int, got %s' % type(arity))
        if not isinstance(function, np.ufunc):
            if function.__code__.co_argcount != arity:
                raise ValueError('arity %d does not match required number of '
                                 'function arguments of %d.'
                                 % (arity, function.__code__.co_argcount))
        if not isinstance(name, str):
            raise ValueError('name must be a string, got %s' % type(name))
        if not isinstance(wrap, bool):
            raise ValueError('wrap must be an bool, got %s' % type(wrap))

        # Check output shape
        args = [np.ones(10) for _ in range(arity)]
        try:
            function(*args)
        except ValueError:
            raise ValueError('supplied function %s does not support arity of %d.'
                             % (name, arity))
        if not hasattr(function(*args), 'shape'):
            raise ValueError('supplied function %s does not return a numpy array.'
                             % name)
        if function(*args).shape != (10,):
            raise ValueError('supplied function %s does not return same shape as '
                             'input vectors.' % name)

        # Check closure for zero & negative input arguments
        args = [np.zeros(10) for _ in range(arity)]
        if not np.all(np.isfinite(function(*args))):
            raise ValueError('supplied function %s does not have closure against '
                             'zeros in argument vectors.' % name)
        args = [-1 * np.ones(10) for _ in range(arity)]
        if not np.all(np.isfinite(function(*args))):
            raise ValueError('supplied function %s does not have closure against '
                             'negatives in argument vectors.' % name)

        if wrap:
            return _Function(function=wrap_non_picklable_objects(function),
                             name=name,
                             arity=arity)
        return _Function(function=function,
                         name=name,
                         arity=arity)


    def _protected_division(x1, x2):
        """Closure of division (x1/x2) for zero denominator."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


    def _protected_sqrt(x1):
        """Closure of square root for negative arguments."""
        return np.sqrt(np.abs(x1))


    def _protected_log(x1):
        """Closure of log for zero arguments."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


    def _protected_inverse(x1):
        """Closure of log for zero arguments."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


    def _sigmoid(x1):
        """Special case of logistic function to transform to probabilities."""
        with np.errstate(over='ignore', under='ignore'):
            return 1 / (1 + np.exp(-x1))


    add2 = _Function(function=np.add, name='add', arity=2)
    sub2 = _Function(function=np.subtract, name='sub', arity=2)
    mul2 = _Function(function=np.multiply, name='mul', arity=2)
    div2 = _Function(function=_protected_division, name='div', arity=2)
    sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
    log1 = _Function(function=_protected_log, name='log', arity=1)
    neg1 = _Function(function=np.negative, name='neg', arity=1)
    inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
    abs1 = _Function(function=np.abs, name='abs', arity=1)
    max2 = _Function(function=np.maximum, name='max', arity=2)
    min2 = _Function(function=np.minimum, name='min', arity=2)
    sin1 = _Function(function=np.sin, name='sin', arity=1)
    cos1 = _Function(function=np.cos, name='cos', arity=1)
    tan1 = _Function(function=np.tan, name='tan', arity=1)
    sig1 = _Function(function=_sigmoid, name='sig', arity=1)

####
    def decay_linear(x, delay):
        x =np.array(x)
        weights = np.arange(1, delay+1)
        weights_sum = np.sum(weights)
        result = [np.dot(x[i-delay: i], weights)/weights_sum for i in range(delay, len(x)+1)]
        result = [np.nan] * (delay - 1) + result
        return result

    def ts_min(x, delay):
        x = np.array(x)
        result =[np.min(x[i-delay: i]) for i in range(delay, len(x)+1)]
        result = [np.nan] * (delay - 1) + result
        return result

    def ts_max(x, delay):
        x = np.array(x)
        result =[np.max(x[i-delay: i]) for i in range(delay, len(x)+1)]
        result = [np.nan] * (delay - 1) + result
        return result

    def ts_argmax(x, delay):
        x = np.array(x)
        result = [np.argmax(x[i-delay: i]) for i in range(delay, len(x)+1)]
        result = [np.nan] * (delay - 1) + result
        return result

    def ts_argmin(x, delay):
        x = np.array(x)
        result = [np.argmin(x[i-delay: i]) for i in range(delay, len(x)+1)]
        result = [np.nan] * (delay - 1) + result
        return result

    def sum_delay(x, delay):
        x = np.array(x)
        result = [np.sum(x[i-delay: i])[-1] for i in range(delay, len(x)+1)]
        result = [np.nan] * (delay - 1) + result
        return result

    def product_delay(x, delay):
        x = np.array(x)
        result = [np.cumprod(x[i-delay: i])[-1] for i in range(delay, len(x)+1)]
        result = [np.nan] * (delay - 1) + result
        return result

    def stddev_delay(x, delay):
        x = np.array(x)
        result = [np.std(x[i-delay: i])[-1] for i in range(delay, len(x)+1)]
        result = [np.nan] * (delay - 1) + result
        return result

    greater2 = _Function(function=_greater, name='greater', arity=2)
    smaller2 = _Function(function=_smaller, name='smaller', arity=2)
    rank1 = _Function(function=_rank, name='rank', arity=1)
    delay2 = _Function(function=_delay, name='delay', arity=2)
    correlation3 = _Function(function=_correlation, name='correlation', arity=3)
    covariance3 = _Function(function=_covariance, name='covariance', arity=3)
    scale2 = _Function(function=_scale, name='scale', arity=2)
    delta2 = _Function(function=_delta, name='delta', arity=2)
    decay_linear2 = _Function(function=_decay_linear, name='decay_linear', arity=2)
    ts_min2 = _Function(function=_ts_min, name='ts_min', arity=2)
    ts_max2 = _Function(function=_ts_max, name='ts_max', arity=2)
    ts_argmax2 = _Function(function=_ts_argmax, name='ts_argmax', arity=2)
    ts_argmin2 = _Function(function=_ts_argmin, name='ts_argmin', arity=2)
    sum_delay2 = _Function(function=_sum_delay, name='sum_delay', arity=2)
    product_delay2 = _Function(function=_product_delay, name='product_delay', arity=2)
    stddev_delay2 = _Function(function=_stddev_delay, name='stddev_delay', arity=2)

    ####
    _function_map = {'add': add2,
                     'sub': sub2,
                     'mul': mul2,
                     'div': div2,
                     'sqrt': sqrt1,
                     'log': log1,
                     'abs': abs1,
                     'neg': neg1,
                     'inv': inv1,
                     'max': max2,
                     'min': min2,
                     'sin': sin1,
                     
                     'greater': greater2,
                     'smaller': smaller2,
                     'rank': rank1,
                     'delay': delay2,
                     'correlation': correlation3,
                     'covariance': covariance3,
                     'scale': scale2,
                     'delta': delta2,
                     'decay_linear': decay_linear2,
                     'ts_min': ts_min2,
                     'ts_max': ts_max2,
                     'ts_argmax': ts_argmax2,
                     'ts_argmin': ts_argmin2,
                     'sum_delay': sum_delay2,
                     'product_delay': product_delay2,
                     'stddev_delay': stddev_delay2}


