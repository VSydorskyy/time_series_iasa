import numpy as np
import math

def rmse(y_true, y_pred, model_params):
    resid = (y_true - y_pred) 
    return  (resid.T @ resid / (y_true.shape[0] - model_params))**0.5

def rsse(y_true, y_pred):
    resid = (y_true - y_pred)
    return resid.T @ resid

def dispersion(distribution):
    return ((distribution - distribution.mean())**2).mean()

def determination_coef(y_true, y_pred):
    return dispersion(y_pred)/dispersion(y_true)

def adjusted_det_coef(y_true, y_pred, model_params):
    return 1 - (1 - determination_coef(y_true, y_pred)**2) * ((y_true.shape[0] - 1) / (y_true.shape[0] - model_params - 1))
                                                       
def log_likelihood(y_true, y_pred):
    resid = (y_true - y_pred)
    return  (-y_true.shape[0]/2) * (1 + np.log(2*math.pi) + np.log(resid.T @ resid / y_true.shape[0]))

def mean_dependent_var(y_true, y_pred):
    return (y_true - y_pred).mean()
                                                       
def std_dependent_var(y_true, y_pred):
    return (y_true - y_pred).std()
                                                       
def akkake_criteria(y_true, y_pred, model_params):
    return (- 2*log_likelihood(y_true, y_pred) / y_true.shape[0]) + 2*model_params/y_true.shape[0]
                                                       
def shwarz_criteria(y_true, y_pred, model_params):
    return (- 2*log_likelihood(y_true, y_pred) / y_true.shape[0]) + np.log(y_true.shape[0])*model_params/y_true.shape[0]
