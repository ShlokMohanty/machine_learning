import warnings 
warnings.warn("Importing from numpy.matlib is deprecated since 1.19.0"
             "The matrix subclass is not the recommended way to represent "
             "matrices or deal with the linear algebra(see"
             "https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-user.html)."
             "please adjust your code to use regular ndarray", 
             PendingDeprecationWarning, stacklevel=2) #gives a warning for adjusting to ndarrays 
import numpy as np
from numpy.matrixlib.defmatrix import matrix, asmatrix
from numpy import *
__version__ = np.__version__
__all__ = np.__all__[:] #all is assigned to all the columns and all the rows 
__all__ += ['rand', 'randn', 'repmat'] #append more items like rand randn and repmat to the __all__
def empty(shape, dtype=None, order='C'):
