import numpy as np

def rmse(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred)

def rsse(y_true, y_pred):
    return ((y_true - y_pred)**2).sum()

def dispersion(distribution):
    return ((distribution - distribution.mean())**2).mean()

def determination_coef(y_true, y_pred):
    return dispersion(y_pred)/dispersion(y_true)

def akkake_criteria(y_true, y_pred, model_params):
    return y_true.shape[0]*np.log(rsse(y_true,y_pred)) + 2*model_params

def compose_all_metrics(y_true, y_pred, num_model_coef):
    return {
    'rmse': rsse(np.array(y_true), y_pred)/y_pred.shape[0],
    'determination_coef': determination_coef(np.array(y_true), y_pred),
    'akke_coef':akkake_criteria(np.array(y_true), y_pred, num_model_coef)
}