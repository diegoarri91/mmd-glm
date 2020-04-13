import numpy as np

from .base import GLM
from ..utils import get_dt, shift_array

# should leave if I implement things properly
from functools import partial
from ..optimization import NewtonMethod

class WGLM(GLM):

    def __init__(self, u0, eta, critic):
        super().__init__(u0=u0, eta=eta)
        self.critic = critic

    def gh_objective(self, dt, X_te, mask_spikes_te, n_samples_fr, newton_kwargs_critic):

        theta = self.get_params()
        n_samples_te = mask_spikes_te.shape[1]
        t = np.arange(X_te.shape[0]) * dt

        u_te = np.einsum('tka,a->tk', X_te, theta)
        r_te = np.exp(u_te)

        u_fr, r_fr, mask_spikes_fr = self.sample(t, shape=(n_samples_fr,))
        X_fr = self.objective_kwargs(t, mask_spikes_fr)['X_te']

        mask_spikes = np.concatenate((mask_spikes_te, mask_spikes_fr), axis=1)
        u = np.concatenate((u_te, u_fr), axis=1)
        r = np.concatenate((r_te, r_fr), axis=1)
        y = np.concatenate((np.ones(n_samples_te), np.zeros(n_samples_fr)))

        # print(mask_spikes.shape, u.shape)
        optimizer = self.critic.fit(t, mask_spikes, u, r, y, u0=self.u0, newton_kwargs=newton_kwargs_critic)
        theta_critic = self.critic.get_params()

        log_likelihood = np.sum(u_te[mask_spikes_te]) - dt * np.sum(r_te)
        g_log_likelihood = np.sum(X_te[mask_spikes_te, :], axis=0) - dt * np.einsum('tka,tk->a', X_te, r_te)
        h_log_likelihood = - dt * np.einsum('tka,tk,tkb->ab', X_te, r_te, X_te)

        w_distance = optimizer.obj_iterations[-1]
        
#         c_w_distance = np.mean(self.critic.transform(t, mask_spikes_fr, u_fr, r_fr))
        c_w_distance = -w_distance
        g_c_w_distance = np.zeros(len(theta))
        h_c_w_distance = np.zeros((len(theta), len(theta)))
        
        ii = 0
        if self.critic.u_kernel is not None:
            g_c_w_distance = g_c_w_distance + (np.sum(self.critic.u_kernel.convolve_continuous(t, X_fr), axis=(0, 1)) / n_samples_fr - np.sum(self.critic.u_kernel.convolve_continuous(t, X_te), axis=(0, 1)) / n_samples_te)
#             g_c_w_distance = g_c_w_distance + np.sum(self.critic.u_kernel.convolve_continuous(t, X_fr), axis=(0, 1)) / n_samples_fr
            ii += len(self.critic.u_kernel.coefs)
    
        if self.critic.u_spk_kernel is not None:
#             args_fr = np.where(shift_array(mask_spikes_fr, 1, fill_value=False))
#             t_spk_fr = (t[args_fr[0]],) + args_fr[1:]
#             conv_u_spk_fr = self.critic.u_spk_kernel.convolve_discrete(t, t_spk_fr) # TODO CORRECT SO I PASS SHAPE
            conv_u_spk_fr = self.critic.u_spk_kernel.convolve_continuous(t, shift_array(mask_spikes_fr, 1, fill_value=False))
            g_fr = np.sum(conv_u_spk_fr[..., None] * X_fr[..., 1:], axis=(0, 1)) / n_samples_fr
            
#             args_te = np.where(shift_array(mask_spikes_te, 1, fill_value=False))
#             t_spk_te = (t[args_te[0]],) + args_te[1:]
#             conv_u_spk_te = self.critic.u_spk_kernel.convolve_discrete(t, t_spk_te)
            conv_u_spk_te = self.critic.u_spk_kernel.convolve_continuous(t, shift_array(mask_spikes_te, 1, fill_value=False))
            g_te = np.sum(conv_u_spk_te[..., None] * X_te[..., 1:], axis=(0, 1)) / n_samples_te
            
            g_c_w_distance[1:] = g_c_w_distance[1:] + (g_fr - g_te)
            ii += len(self.critic.u_spk_kernel.coefs)
    
        if self.critic.r_kernel is not None:
            g_fr = np.sum(self.critic.r_kernel.convolve_continuous(t, X_fr * r_fr[..., None]), axis=(0, 1)) / n_samples_fr
            h_fr = np.sum(self.critic.r_kernel.convolve_continuous(t, np.einsum('tka,tkb->tkab', X_fr, X_fr) * r_fr[..., None, None]), axis=(0, 1)) / n_samples_fr
            g_te = np.sum(self.critic.r_kernel.convolve_continuous(t, X_te * r_te[..., None]), axis=(0, 1)) / n_samples_te
            h_te = np.sum(self.critic.r_kernel.convolve_continuous(t, np.einsum('tka,tkb->tkab', X_te, X_te) * r_te[..., None, None]), axis=(0, 1)) / n_samples_te
            g_c_w_distance = g_c_w_distance + (g_fr - g_te)
            h_c_w_distance = h_c_w_distance + (h_fr - h_te)
            
        if self.critic.r_spk_kernel is not None:
            conv_u_spk_fr = self.critic.r_spk_kernel.convolve_continuous(t, shift_array(mask_spikes_fr, 1, fill_value=False))
            g_fr = np.sum(conv_u_spk_fr[..., None] * X_fr * r_fr[..., None], axis=(0, 1)) / n_samples_fr
            h_fr = np.sum(np.einsum('tka,tkb->tkab', X_fr, X_fr) * (r_fr * conv_u_spk_fr)[..., None, None], axis=(0, 1)) / n_samples_fr
            
            conv_u_spk_te = self.critic.r_spk_kernel.convolve_continuous(t, shift_array(mask_spikes_te, 1, fill_value=False))
            g_te = np.sum(conv_u_spk_te[..., None] * X_te * r_te[..., None], axis=(0, 1)) / n_samples_te
            h_te = np.sum(np.einsum('tka,tkb->tkab', X_te, X_te) * (r_te * conv_u_spk_te)[..., None, None], axis=(0, 1)) / n_samples_te
            
            g_c_w_distance = g_c_w_distance + (g_fr - g_te)
            h_c_w_distance = h_c_w_distance + (h_fr - h_te)
            ii += len(self.critic.r_spk_kernel.coefs)
            
        if 'r_spk' in self.critic.features:
            X_r_fr_spk = np.array([np.einsum('ta,t->a', X_fr[mask_spikes_fr[:, sw], sw, :], r_fr[mask_spikes_fr[:, sw], sw]) for sw in range(n_samples_fr)])
            X_r_X_fr_spk = np.array([np.einsum('ta,t,tb->ab', X_fr[mask_spikes_fr[:, sw], sw, :], r_fr[mask_spikes_fr[:, sw], sw], X_fr[mask_spikes_fr[:, sw], sw, :]) for sw in range(n_samples_fr)])
            X_r_te_spk = np.array([np.einsum('ta,t->a', X_te[mask_spikes_te[:, sw], sw, :], r_te[mask_spikes_te[:, sw], sw]) for sw in range(n_samples_te)])
            X_r_X_te_spk = np.array([np.einsum('ta,t,tb->ab', X_te[mask_spikes_te[:, sw], sw, :], r_te[mask_spikes_te[:, sw], sw], X_te[mask_spikes_te[:, sw], sw, :]) for sw in range(n_samples_te)])
            
#             g_c_w_distance = g_c_w_distance + theta_critic[ii] * np.sum(X_r_fr_spk, 0) / n_samples_fr
#             h_c_w_distance = h_c_w_distance + theta_critic[ii] * np.sum(X_r_X_fr_spk, 0) / n_samples_fr
            g_c_w_distance = g_c_w_distance + theta_critic[ii] * (np.sum(X_r_fr_spk, 0) / n_samples_fr - np.sum(X_r_te_spk, 0) / n_samples_te)
            h_c_w_distance = h_c_w_distance + theta_critic[ii] * (np.sum(X_r_X_fr_spk, 0) / n_samples_fr - np.sum(X_r_X_te_spk, 0) / n_samples_te)
            ii += 1
        # print(c_w_distance, g_c_w_distance)
        

        obj = log_likelihood + c_w_distance
        g_obj = g_log_likelihood + g_c_w_distance
        h_obj = h_log_likelihood + h_c_w_distance

        metrics = dict(w_distance=w_distance, c_w_distance=c_w_distance, w_distance2=optimizer.obj_iterations)

        return obj, g_obj, h_obj, metrics

    def objective_kwargs(self, t, mask_spikes, stim=None, n_samples_fr=100, newton_kwargs_critic=None):

        n_kappa = 0
        n_eta = 0 if self.eta is None else self.eta.nbasis

        X_te = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
        X_te[:, :, 0] = -1.

        if self.eta is not None:
            args = np.where(shift_array(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.eta.nbasis
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X_te[:, :, n_kappa + 1:] = -X_eta

        obj_kwargs = dict(dt=get_dt(t), X_te=X_te, mask_spikes_te=mask_spikes, n_samples_fr=n_samples_fr,
                          newton_kwargs_critic=newton_kwargs_critic)

        return obj_kwargs

    def fit(self, t, mask_spikes, stim=None, newton_kwargs=None, verbose=False, n_samples_fr=100,
            newton_kwargs_critic=None):

        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        objective_kwargs = self.objective_kwargs(t, mask_spikes, n_samples_fr=n_samples_fr,
                                                 newton_kwargs_critic=newton_kwargs_critic)
        gh_objective = partial(self.gh_objective, **objective_kwargs)

        optimizer = NewtonMethod(model=self, gh_objective=gh_objective, verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        return optimizer
