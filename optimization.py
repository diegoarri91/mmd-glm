import time

import numpy as np
from scipy.linalg import solveh_banded


class NewtonMethod:

    def __init__(self, theta0=None, g_log_prior=None, h_log_prior=None, gh_log_prior=None, gh_log_likelihood=None,
                 theta_independent_h=False, learning_rate=1e-1, initial_learning_rate=1e-2,
                 max_iterations=200, stop_cond=5e-4, warm_up_iterations=5, verbose=False, use_hessian=True):

        self.theta0 = theta0

        self.g_log_prior = g_log_prior
        self.h_log_prior = h_log_prior
        self.gh_log_prior = gh_log_prior
        self.gh_log_likelihood = gh_log_likelihood

        self.use_prior = True if self.gh_log_prior is not None or self.g_log_prior is not None else False

        self.learning_rate = learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.max_iterations = max_iterations
        self.stop_cond = stop_cond
        self.warm_up_iterations = warm_up_iterations
        self.use_hessian = use_hessian
        self.verbose = verbose

        self.theta_iterations = []
        self.log_prior_iterations = []
        self.log_posterior_iterations = []
        self.g_log_posterior = None
        self.h_log_posterior = None
        self.fit_status = None

    def optimize(self):

        log_prior = np.nan
        theta = self.theta0
        
        status = ''
        converged = nan_parameters = False

        if self.verbose:
            print('Starting gradient ascent... \n')

        t0 = time.time()

        for ii in range(self.max_iterations):

            log_likelihood, g_log_likelihood, h_log_likelihood = self.gh_log_likelihood(theta)
            max_likelihood_band = h_log_likelihood.shape[0]

            if self.use_prior:
                log_prior, g_log_prior, h_log_prior = self.gh_log_prior(theta)
                log_posterior = log_likelihood + log_prior
                g_log_posterior = g_log_likelihood + g_log_prior
                if self.banded_h:
                    if max_likelihood_band >= max_prior_band:
                        h_log_posterior = h_log_likelihood
                        h_log_posterior[:max_prior_band, :] += h_log_prior
                    else:
                        h_log_posterior = h_log_prior
                        h_log_posterior[:max_likelihood_band, :] += h_log_likelihood
                else:
                    h_log_posterior = h_log_likelihood + h_log_prior
            else:
                log_posterior = log_likelihood
                g_log_posterior = g_log_likelihood
                h_log_posterior = h_log_likelihood

            if self.verbose:
                print('\r', 'Iteration {} of {}'.format(ii, self.max_iterations), '|',
                      'Elapsed time: {} seconds'.format(np.round(time.time() - t0, 2)), '|',
                      'log_prior={}'.format(np.round(log_prior, 2)), '|',
                      'log_posterior={}'.format(np.round(log_posterior, 2)), end='')

            self.log_prior_iterations += [log_prior]
            self.log_posterior_iterations += [log_posterior]
            self.theta_iterations += [theta]

            old_log_posterior = np.nan if len(self.log_posterior_iterations) < 2 else self.log_posterior_iterations[-2]
            diff_log_posterior = log_posterior - old_log_posterior
            if ii > self.warm_up_iterations and np.abs(diff_log_posterior / old_log_posterior) < self.stop_cond:
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
            if self.use_hessian:
                theta = theta - learning_rate * np.linalg.solve(h_log_posterior, g_log_posterior)
            else:
                theta = theta + learning_rate * g_log_posterior

        fitting_time = (time.time() - t0) / 60.

        status += 'Elapsed time: {} minutes | '.format(np.round(fitting_time, 4))

        if nan_parameters:
            log_posterior_monotonic = None
        elif np.any(np.diff(self.log_posterior_iterations) < 0.):
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

        self.theta_iterations = np.stack(self.theta_iterations, 1)
        self.log_posterior_iterations = np.array(self.log_posterior_iterations)
        self.log_prior_iterations = np.array(self.log_prior_iterations)
        self.g_log_posterior = g_log_posterior
        self.h_log_posterior = h_log_posterior

        return self

