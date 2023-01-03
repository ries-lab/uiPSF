"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     

@author: Jonas Hellgoth, Sheng Liu
"""

from abc import ABCMeta, abstractmethod
import time

import numpy as np
import scipy as sp
import scipy.optimize as optimize
import tensorflow as tf
import sys
import tkinter as tk
from tkinter import messagebox as mbox

class OptimizerABC:
    """
    Abstract base class for optimizers. It ensures consistency and compatability between Fitters and Optimizers.
    Core function is 'minimize' which is called by the fitter. The rest is handled by the optimizer.
    Allows to use different TensorFLow optimizers and also the L-BFGS-B optimizer from scipy.
    Is basically a wraper around those optimizers to call them similarly in the Fitter.
    Defines an interface for other optimizers (basically the minimize function) that self-made optimizers must fulfill.
    """

    __metaclass__ = ABCMeta

    def __init__(self, maxiter, options, kwargs) -> None:
        self.maxiter = maxiter

        self.print_step_size = np.max((np.round(self.maxiter / 10).astype(int),20))
        self.print_width = len(str(self.print_step_size))

        self.history = [['step', 'time', 'loss']]

        self.objective = None
        self.variables = None
        self.weight = None
        self.opt = self.create_actual_optimizer(options, kwargs)

    @abstractmethod
    def create_actual_optimizer(self, options):
        """
        Here the actual underlying optimizer should be created.
        """
        raise NotImplementedError("You need to implement a 'create_actual_optimizer' method in your optimizer class.")

    def minimize(self, objective, variables, pbar):
        """
        Adapts the given variables in a way that minimizes the given objective.
        Returns the new state of the variables after the optimization.
        """
        variables = [tf.Variable(variable) for variable in variables]

        for step in range(self.maxiter):
            start = time.time()
            with tf.GradientTape() as tape:
                tape.watch(variables)
                loss = objective(variables)
            pbar.update(1)
            #self.write_output(step, loss)
            pbar.set_description("current loss %f" % loss)
            gradients = tape.gradient(loss, variables)
            self.opt.apply_gradients(zip(gradients, variables))     

            self.update_history(step+1, time.time()-start, loss.numpy())

        #self.write_output(self.maxiter, objective(variables), True)

        result_variables = [variable.numpy() for variable in variables]

        return result_variables

    def objective_wrapper_for_optimizer(self):
        """
        Wrapper around the actual objective. Needed since TensorFLow optimizer
        can only optimize a function that takes no arguments and returns a loss.
        """
        return self.objective(self.variables)

    def write_output(self, step, loss, do_anyway=False):
        """
        Writes output to the console in a nicely formatted way.
        Used to show user the progress of the optimization.
        """
        # TODO: one could caluclate an estimate how long the optimization still takes
        self.print_step_size =  np.max((np.round(self.maxiter / 10).astype(int),20))
        if (step % self.print_step_size == 0) or do_anyway:
            #tf.print(f"[{step:5}/{self.maxiter}]  loss={loss:>8.2f} ")
            tf.print("step:",step,"loss:",loss)
        return

    def update_history(self, step, time, loss):
        """
        Save information of each iteration to the history.
        This history can be later used to analyze the efficiency of the optimization.
        """
        self.history.append([step, time, loss])
        return

class Adadelta(OptimizerABC):
    """
    Wrapper around TensorFlows Adadelta optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, rho=0.95, epsilon=1e-07,
                name='Adadelta', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Adadelta(**options, **kwargs)


class Adagrad(OptimizerABC):
    """
    Wrapper around TensorFlows Adagrad optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,
                name='Adagrad', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Adagrad(**options, **kwargs)


class Adam(OptimizerABC):
    """
    Wrapper around TensorFlows Adam optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Adam(**options, **kwargs)


class Adamax(OptimizerABC):
    """
    Wrapper around TensorFlows Adamax optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, 
                name='Adamax', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Adamax(**options, **kwargs)


class Ftrl(OptimizerABC):
    """
    Wrapper around TensorFlows Ftrl optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
                l1_regularization_strength=0.0, l2_regularization_strength=0.0,
                name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0, **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Ftrl(**options, **kwargs)


class Nadam(OptimizerABC):
    """
    Wrapper around TensorFlows Nadam optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                name='Nadam', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Nadam(**options, **kwargs)


class RMSprop(OptimizerABC):
    """
    Wrapper around TensorFlows RMSprop optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                name='RMSprop', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.RMSprop(**options, **kwargs)


class SGD(OptimizerABC):
    """
    Wrapper around TensorFlows SGD optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.01, momentum=0.0, nesterov=False,
                name='SGD', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.SGD(**options, **kwargs)


class L_BFGS_B(OptimizerABC):
    """
    Wrapper around scipys L-BFGS-B optimizer.
    There is not L-BFGS-B optimizer available in TensorFlow.
    """
    # this is the type of optimizer Rainer used
    # alternatve would be optimizer.lbfgs_minimize from tensorflow-probability
    # this works similar but seemed to be slower

    # the problem with the scipy (and also the tensorflow-probability) implementation is that
    # they can only handle 1D tensors/arrays, therefore we must flatten our variables,
    # save shapes and lengths and reshape them again in objective
    # this is similar to https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
    # right now this works fine
    def __init__(self, maxiter, gtol=1e-10, **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

        self.step = 0
        self.status = None # to access final output from outside

        self.shapes = []
        self.lengths = []
        self.dtypes = []

    def create_actual_optimizer(self, options, kwargs):
        """
        Implemented to allow to inherit from ABC. Is just a placeholder in this case since
        the scipy API does only provide a function not a real optimizer object.
        Used to adapt the options to fit the scipy API.
        """
        self.options = {**options, **kwargs}
        return None

    def minimize(self, objective, variables,pbar):
        """
        Adapts the given variables in a way that minimizes the given objective.
        Returns the new state of the variables after the optimization.
        ABC overwritten since optimization works a bit different for the scipy optimizer.
        """
        self.objective = objective
        self.step = 0
        init_var = self.flatten_variables(variables)
        self.options['maxiter'] = self.maxiter
        start_time = pbar.postfix[-1]['time']
        result = optimize.minimize(fun=self.objective_wrapper_for_optimizer, x0=init_var, args=(pbar,start_time), jac=True, method='L-BFGS-B', options=self.options)
        
        #self.write_output(self.maxiter, result.fun, True)
        
        self.status = result

        result_var = result.x
        variables = self.reshape_variables_np(result_var)
        return variables

    def flatten_variables(self, variables):
        """
        Flattens and concatenates a list of variables to one large vector. Needed since
        L-BFGS-B optimizer implementation can only handle vectors. Shapes and lengths
        of varibales are saved as attributes for later reconstruction.
        """
        flat_variables = []
        self.shapes = []
        self.lengths = []
        self.dtypes = []
        # at one point we were experimenting with complex tensors
        # but there was no optimizer available that can handle complex tensors/arrays
        # therefore we switched back to float32
        # so dtypes and casting are actually not needed anymore
        # but left active since they do no harm and may be useful in the future

        for variable in variables:
            shape = variable.shape
            self.shapes.append(shape)
            self.lengths.append(np.product(shape))
            self.dtypes.append(variable.dtype)
            flat_variables.append(variable.flatten())

        return np.concatenate(flat_variables)

    def reshape_variables_np(self, var):
        """
        Reconstructs/reshapes the current state of the variables from the current guess
        of the optimizer (var vector). Called for final reconstruction and therefore
        implemented with numpy.
        """
        variables = []
        idx_count = 0

        for i, (shape, length, dtype) in enumerate(zip(self.shapes, self.lengths, self.dtypes)):
            variable = var[idx_count : idx_count+length]
            variables.append(np.reshape(variable, shape).astype(dtype)) # dtypes probably not needed anymore (see above)
            idx_count += length

        return variables

    def reshape_variables_tf(self, var):
        """
        Reconstructs/reshapes the current state of the variables from the current guess
        of the optimizer (var vector). Called in each iteration of the optimization
        and therefore implemented with tensorflow.
        """
        variables = [None] * len(self.shapes)
        idx_count = 0

        for i, (shape, length, dtype) in enumerate(zip(self.shapes, self.lengths, self.dtypes)):
            variable = var[idx_count : idx_count+length]
            variables[i] = tf.cast(tf.reshape(variable, shape), dtype) # dtypes probably not needed anymore (see above)
            idx_count += length

        return variables

    def objective_wrapper_for_optimizer(self, var, pbar,start_time):
        """
        Wrapper around the actual objective. This is needed since the L-BFGS-B implementation is based
        on Fortran code that needs double precision (float64) whereas tensorflow
        works with single precision (float32).
        """
        return [np.real(tensor.numpy()).astype(np.float64) for tensor in self.objective_wrapper_for_gradient(tf.cast(var, tf.float32),pbar,start_time)]

    #@tf.function
    def objective_wrapper_for_gradient(self, var, pbar,start_time=None):
        """
        Another wrapper around the actual objective. Needed to allow graient calculation
        via tf.function decorator. When working with tf.GradientTape() this can probabaly
        be done in first wrapper.
        """
        
        loss = 0.0
        Np = len(self.shapes)
        Nfit = self.shapes[0][0]
        start = time.time()
        grad = [None]*Np
        batchsize = self.batch_size
        variables = self.reshape_variables_tf(var)
        ind = list(np.int32(np.linspace(0,Nfit,Nfit//batchsize+2)))
        var1 = [None]*Np
        for i in range(len(ind)-1):
            for k in range(Np):
                if (Nfit) in self.shapes[k]:
                    if self.shapes[k].index(Nfit)==0:
                        var1[k] = variables[k][ind[i]:ind[i+1]]
                    elif self.shapes[k].index(Nfit)==1:
                        var1[k] = variables[k][:,ind[i]:ind[i+1]]
                    else:
                        var1[k] = variables[k]
                        if i == 0:
                            grad[k] = 0.0
                else:
                    var1[k] = variables[k]
                    if i == 0:
                        grad[k] = 0.0

                    

            with tf.GradientTape() as tape:
                tape.watch(var1)
                loss1 = self.objective(var1,ind[i:i+2])
            w1 = var1[0].shape[0]/Nfit
            loss = loss + loss1*w1   
            grad1 = tape.gradient(loss1, var1)

            for k in range(Np):
                if grad1[k] is None:
                    grad1[k] = var1[k]*0
                    
            for k in range(Np):
                if (Nfit) in self.shapes[k]:
                    if grad[k] is None:
                        grad[k] = grad1[k]
                    else:
                        grad[k] = tf.concat((grad[k],grad1[k]),axis=self.shapes[k].index(Nfit))
                else:
                    grad[k] = grad[k]+grad1[k]*w1


        pbar.postfix[-1]['loss'] = loss
        pbar.postfix[-1]['time'] = start_time +pbar._time()-pbar.start_t
        pbar.update(1)
        #pbar.set_description("current loss %f" % loss)
        self.update_history(self.step+1, time.time()-start, loss.numpy())
        self.step += 1
        # while using tf.function one maybe needs to do something like:
        # tf.py_function(self.history.append, inp=[loss], Tout=[])
        gradvec = tf.reshape(grad[0],[-1])
        for g in grad[1:]:
            gradvec = tf.concat((gradvec,tf.reshape(g,[-1])),axis = 0)
        

        return loss, gradvec
        
    def objective_wrapper_for_gradient_copy(self, var, pbar,start_time=None):
        """
        Another wrapper around the actual objective. Needed to allow graient calculation
        via tf.function decorator. When working with tf.GradientTape() this can probabaly
        be done in first wrapper.
        """

        start = time.time()
        with tf.GradientTape() as tape:
            tape.watch(var)
            variables = self.reshape_variables_tf(var)
            loss = self.objective(variables)
        #self.write_output(self.step, loss)
        
        pbar.postfix[-1]['loss'] = loss
        pbar.postfix[-1]['time'] = start_time +pbar._time()-pbar.start_t
        pbar.update(1)
        #pbar.set_description("current loss %f" % loss)
        self.update_history(self.step+1, time.time()-start, loss.numpy())
        self.step += 1
        # while using tf.function one maybe needs to do something like:
        # tf.py_function(self.history.append, inp=[loss], Tout=[])
        grad = tape.gradient(loss, var)  

        #gpu = tf.config.list_physical_devices('GPU')
        #memlimit = int(6.7e9)
        
        #if gpu:
        #    meminfo = tf.config.experimental.get_memory_info('GPU:0')
        #    if meminfo['peak']>memlimit:
        #        raise MemoryError('GPU is out of memory, reduce number of beads')
        return loss, grad
        
