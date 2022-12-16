from regpy.solvers import Solver

import logging
import numpy as np

import copy

'''
Gradient Descent Algorithm for minimizing 1/2*||Ff-g||^2+alpha*penalty_term(f)

In:
->setting: Contains forward operator F and domain and codomain of the forward operator
->rhs: Data g
->init: Initial guess for gradient decent algorithm
->stepsize: Stepsize of gradient step
->alpha: Offset between data fidelity term and penalty term
->penalty: Functional derivative of penalty term. Default penalty term is 1/2*||f||^2 with functional derivative: Gram_matrix @ f

Out:
Solution f to minimization problem
'''

class Gradient_Descent(Solver):

    def __init__(self, setting, rhs, init, stepsize=None, alpha=1, penalty=None):
        super().__init__()
        self.setting = setting
        self.rhs = rhs
        self.x = init
        self.y, self.deriv = self.setting.op.linearize(self.x)
        self.stepsize = stepsize or 1 / self.deriv.norm()**2
        self.alpha=alpha
        self.penalty=penalty or setting.Hdomain.gram

    def _next(self):
        residual = self.y - self.rhs
        gy_residual = self.setting.Hcodomain.gram(residual)
        residual_step = self.setting.Hdomain.gram_inv(self.deriv.adjoint(gy_residual))
        penalty_step = self.alpha * self.penalty(self.x)
        update = residual_step+penalty_step
	#Non-normalized gradient descent step
        self.x -=self.stepsize*update#/self.setting.Hdomain.norm(update)
        self.y, self.deriv = self.setting.op.linearize(self.x)

        if self.log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(np.vdot(residual, gy_residual)))
            self.log.info('|residual| = {}'.format(norm_residual))
            self.log.info('|norm| = {}'.format(np.vdot(self.x, self.penalty(self.x))))

class Landweber(Solver):
    
    def __init__(self, setting, rhs, init, stepsize=None, nr_freeze=10):
        super().__init__()
        self.setting = setting
        self.op = copy.deepcopy(self.setting.op)
        """The problem setting."""
        self.rhs = rhs
        """The right hand side."""
        self.x = init
        self.y, deriv = self.setting.op.linearize(self.x)
        self.deriv = deriv
        """The derivative at the current iterate."""
        self.stepsize = stepsize or 1 / self.deriv.norm()**2
        """The stepsize."""
        self.nr_freeze = nr_freeze
        """Frozen derivative for which number of iterations"""
        
        self.counter = 0

    def _next(self):
        self.counter += 1
        
        residual = self.y - self.rhs
        gy_residual = self.setting.Hcodomain.gram(residual)
        self.x -= self.stepsize * self.setting.Hdomain.gram_inv(self.deriv.adjoint(gy_residual))
        
        if self.counter >= self.nr_freeze:
            self.y, self.deriv = self.setting.op.linearize(self.x)
            self.counter = 0
        else:
            self.y = self.op._eval(self.x, differentiate=False, linearized=True)

        if self.log.isEnabledFor(logging.INFO):
            norm_residual = np.sqrt(np.real(np.vdot(residual, gy_residual)))
            self.log.info('|residual| = {}'.format(norm_residual))