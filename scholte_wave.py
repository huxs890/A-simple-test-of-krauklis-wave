
import numpy as np
from scipy.optimize import brentq

def scholte_wave(f, 
                rho_f,
                rho_s,
                cf,
                cp, 
                cs,
    ):
    """Scholte wave phase velocity"""
    def F(c):
        ap = np.sqrt(1 - (c / cp)**2)
        ass = np.sqrt(1 - (c / cs)**2)
        af = np.sqrt(1 - (c / cf)**2)
        return (2 - (c / cs)**2)**2 - 4*ap*ass + \
               (rho_f / rho_s)*(c / cs)**4*ap/af

    c1 = brentq(F, 1e-4, min(cf, cs)*(1 - 1e-9))
    c  = np.linspace(c1,c1,len(f))
    k  = 2*np.pi*np.asarray(f) / c

    return c, k
