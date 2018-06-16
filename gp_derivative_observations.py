

import numpy as np
from GPy.core import GP
from GPy import util
from derivative_observations import DerivativeObservations
from stationary_extended import ExpQuadExtended

class GPDerivativeObservations(GP):
    """
    Gaussian Process model for Derivative Observations

    This is a thin wrapper around the models.GP class, with a set of sensible defaults
    The format is very similar to that used for the coregionalisation class.

    :param X_list: list of input observations corresponding to each output
    :type X_list: list of numpy arrays
    :param Y_list: list of observed values related to the different noise models
    :type Y_list: list of numpy arrays
    :param index: list of output indices
    :type index: list of integers of the same length as X_list and Y_list
    :param kernel: a GPy Derivative Observations kernel (if previously defined)
    :param base_kernel: a GPy kernel from which to build a Derivative 
                        Observations kernel ** defaults to RBF **
    :type kernel: None | GPy.kernel defaults
    :param base_kernel: a GPy kernel ** defaults to RBF **
    :type base_kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
    :type likelihoods_list: None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """
    def __init__(self, X_list, Y_list, index=None, kernel=None, likelihoods_list=None, base_kernel=None, name='GPDO', kernel_name='deriv_obs'):

        #Input and Output
        assert kernel is None or base_kernel is None
        Ny = len(Y_list)
        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list,index)
        
        # WHY DO WE NEED THE FOLLOWING?
        # Make output index unique ordered values
        # _,id_u = np.unique(self.output_index,return_inverse=True)
        # is id_u.min() always 0?
        # self.output_index = (id_u.min() + id_u).reshape(self.output_index.shape)

        #Kernel
        if kernel is None:
            if base_kernel is None:
                base_kernel = ExpQuadExtended(X.shape[1]-1)
            kernel = DerivativeObservations(X.shape[1], output_dim=Ny, active_dims=None, base_kernel=base_kernel)

        #Likelihood
        likelihood = util.multioutput.build_likelihood(Y_list,self.output_index,likelihoods_list)

        super().__init__(X,Y,kernel,likelihood, Y_metadata={'output_index':self.output_index})

    def set_XY(self, X_list=None, Y_list=None, index=None):
        """
        Set the input / output data of the model.
        This is useful if we wish to change our existing data but maintain the same model.
        Note that the length of X_list, Y_list, and index (i.e., the number of outputs) cannot change.

        :param X_list: list of input observations corresponding to each output
        :type X_list: list of numpy arrays
        :param Y_list: list of observed values related to the different noise models
        :type Y_list: list of numpy arrays
        :param index: list of output indices
        :type index: list of integers of the same length as X_list and Y_list
        """
        assert X_list is not None or Y_list is not None
        output_dim = len(Y_list) if Y_list is not None else len(X_list)
        assert self.kern.output_dim == output_dim   # Output dim cannot change

        if X_list is None:
            Y, X, self.output_index = util.multioutput.build_XY(Y_list, index=index)
            # As Y contains the indices in the last column, we must move Y's last column to X
            X = self.X
            X[:,-1] = Y[:,-1]
            Y = np.delete(Y, -1, 1)
        else:
            X, Y, self.output_index = util.multioutput.build_XY(X_list, Y_list, index)

        self.update_model(False)
        self.Y_metadata['output_index'] = self.output_index      # Must update this before updating X, Y
        super().set_XY(X, Y)
        
