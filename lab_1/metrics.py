import numpy as np

def get_coef_matrix_from_parsing_results(parsing_results):
    signal_time_shift = max(max(parsing_results['out_weights'].keys()), max(parsing_results['manage_weights'].keys()))
    return np.array([parsing_results['out_weights'].get(i, 0.) for i in range(signal_time_shift+1)] + \
                    [parsing_results['manage_weights'].get(i, 0.) for i in range(signal_time_shift+1)])

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

def count_model_coef_rmse(model, parsing_results):
    return rmse(get_coef_matrix_from_parsing_results(parsing_results), model.coef_matrix.squeeze())