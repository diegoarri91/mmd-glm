import sys
sys.path.append("/home/diego/Dropbox/Python/imports/")

import numpy as np

from fun_misc import are_equal
from fun_signals import get_arg, get_dt

class LIF:
    
    def __init__(self, tau = None, R = None, vr = None, vrst = None, tref = 0., \
                 vt = np.inf, vpeak = None):
        self.tau, self.R, self.vr = tau, R, vr
        self.vrst = vrst
        self.tref = tref
        self.vt = vt
        
        if vpeak is None:
            self.vpeak = vt
        else:
            self.vpeak = vpeak

    def sample(self, v0, t, stim, sigma=0):
        tau, R, vr, vt, vrst, tref, vpeak = self.tau, self.R, self.vr, self.vt, self.vrst, self.tref, self.vpeak
        
        dt = get_dt(t)
        
        if stim.ndim==1:
            v = np.zeros((len(t),1)) * np.nan
            mask_spikes = np.zeros((len(t),1), dtype=bool)
        else:
            v = np.zeros(stim.shape ) * np.nan
            mask_spikes = np.zeros(stim.shape, dtype=bool)
            
        v[0,...] = v0
        mask_ref = np.zeros(stim.shape[1:], dtype=bool)
        t_refs = -dt * np.ones(stim.shape[1:]) # So integration stops for exactly tref
        
        j = 0
        while j < len(t)-1: 
            
            dvdt = (-( v[j, ...] - vr ) + stim[j, ...] * R)/tau + sigma * np.sqrt(dt/tau) * np.random.randn(*v.shape[1:]) / dt
            v[j+1, ...] = v[j, ...] + dvdt * dt
            v[j+1, mask_ref] = vrst
            t_refs[mask_ref] += dt
            
            mask_spk = v[j + 1, ...] > vt
            mask_spikes[j + 1, :] = mask_spk
            v[j+1, mask_spk] = vpeak
            
            mask_ref[mask_spk] = True
            mask_release_ref = (t_refs > tref) | are_equal(t_refs, tref)
            mask_ref[mask_release_ref] = False
            t_refs[mask_release_ref] = -dt
            
            j += 1
            
        return v, mask_spikes
