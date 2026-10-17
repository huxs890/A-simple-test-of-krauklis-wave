import numpy as np

def wavenumber_to_group(f, k):
    vg = 2 * np.pi / np.gradient(k.real, f)
    return vg
    
