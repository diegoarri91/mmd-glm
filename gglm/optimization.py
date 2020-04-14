from functools import partial
import time

import numpy as np
from scipy.linalg import solveh_banded


class NewtonMethod:

    def __init__(self, model=None, gh_objective=None, learning_rate=1e-1, initial_learning_rate=None,
                 max_iterations=200, stop_cond=5e-4, warm_up_iterations=5, verbose=False, use_hessian=True,
                 clip_theta=None):

        self.model = model
        self.learning_rate = learning_rate
        self.initial_learning_rate = initial_learning_rate if initial_learning_rate is not None else learning_rate
        self.max_iterations = max_iterations
        self.stop_cond = stop_cond
        self.warm_up_iterations = warm_up_iterations
        self.use_hessian = use_hessian
        self.clip_theta = clip_theta
        self.verbose = verbose

        self.gh_objective = gh_objective

        self.theta_iterations = []
        self.obj_iterations = []
        self.metrics_iterations = None
        self.g_obj = None
        self.h_obj = None
        self.fit_status = None

    def optimize(self):

        theta = self.model.get_params()
        
        status = ''
        converged = nan_parameters = False

        if self.verbose:
            print('Starting gradient ascent... \n')

        t0 = time.time()

        for ii in range(self.max_iterations):

            obj, g_obj, h_obj, metrics = self.gh_objective()
            if ii == 0 and metrics is not None:
                self.metrics_iterations = {key: [] for key in metrics.keys()}

            if self.verbose:
                print('\r', 'Iteration {} of {}'.format(ii, self.max_iterations), '|',
                      'Elapsed time: {} seconds'.format(np.round(time.time() - t0, 2)), '|',
                      # 'log_prior={}'.format(np.round(log_prior, 2)), '|',
                      'objective={}'.format(np.round(obj, 2)), end='')

            # self.log_prior_iterations += [log_prior]
            self.obj_iterations.append(obj)
            self.theta_iterations.append(theta)
            if self.metrics_iterations is not None:
                for key in metrics.keys():
                    self.metrics_iterations[key].append(metrics[key])

            old_obj = np.nan if len(self.obj_iterations) < 2 else self.obj_iterations[-2]
            diff_log_posterior = obj - old_obj
            if ii > self.warm_up_iterations and np.abs(diff_log_posterior / old_obj) < self.stop_cond:
                status += '\n Iteration {} of {} | Converged | '.format(ii, self.max_iterations)
                converged = True
                nan_parameters = False
                n_iterations = ii + 1
                break
            elif np.any(np.isnan(theta)):
                status += "\n There are nan parameters. "
                nan_parameters = True
                converged = False
                n_iterations = ii + 1
                break
            elif ii == self.max_iterations - 1:
                status += '\n Not converged after {} iterations. '.format(self.max_iterations)
                converged = False
                nan_parameters = False
                n_iterations = ii + 1
            
            if ii > self.warm_up_iterations:
                learning_rate = self.learning_rate
            else:
                learning_rate = self.initial_learning_rate
#             print(learning_rate)
            if self.use_hessian:
                theta = theta - learning_rate * np.linalg.solve(h_obj, g_obj)
            else:
                theta = theta + learning_rate * g_obj
            if self.clip_theta is not None:
                theta[theta > self.clip_theta] = self.clip_theta
                theta[theta < -self.clip_theta] = -self.clip_theta
            self.model.set_params(theta)

        fitting_time = (time.time() - t0) / 60.

        status += 'Elapsed time: {} minutes | '.format(np.round(fitting_time, 4))

        if nan_parameters:
            log_posterior_monotonic = None
        elif np.any(np.diff(self.obj_iterations) < 0.):
            status += 'Log posterior is not monotonic \n'
            log_posterior_monotonic = False
        else:
            status += 'Log posterior is monotonic \n'
            log_posterior_monotonic = True

        if self.verbose:
            print('\n', status)

        self.fit_status = {'n_iterations': n_iterations, 'converged': converged, 'nan_parameters': nan_parameters,
                           'fitting_time': fitting_time, 'log_posterior_monotonic': log_posterior_monotonic,
                           'status': status}

        # self.theta_iterations = np.stack(self.theta_iterations, 1)
        self.obj_iterations = np.array(self.obj_iterations)
        if self.metrics_iterations is not None:
            for key in self.metrics_iterations.keys():
                self.metrics_iterations[key] = np.array(self.metrics_iterations[key])
        # self.log_prior_iterations = np.array(self.log_prior_iterations)
        # self.g_log_posterior = g_log_posterior
        # self.h_log_posterior = h_log_posterior

        return self

