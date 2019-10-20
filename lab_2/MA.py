import numpy as np


def moving_average(a, n=3, weights=None):
    
    if weights == 'exp':
        weights = np.exp(-np.arange(start=0,stop=n))
    elif weights == 'exp_smoothed':
        weights = np.arange(start=0,stop=n)**(1 - 2/(n+1))
    else:
        weights = np.ones(n)
    
    mov_avarage = []
    
    for i in range(len(a)-n+1):
        mov_avarage.append(a[i:i+n]@weights / sum(weights))
        
    return np.array(mov_avarage)