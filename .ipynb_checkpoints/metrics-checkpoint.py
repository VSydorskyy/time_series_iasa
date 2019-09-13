import numpy as np

def get_coef_matrix_from_parsing_results(parsing_results):
    signal_time_shift = max(max(parsing_results['out_weights'].keys()), max(parsing_results['manage_weights'].keys()))
    return np.array([parsing_results['out_weights'].get(i, 0.) for i in range(signal_time_shift+1)] + \
                    [parsing_results['manage_weights'].get(i, 0.) for i in range(signal_time_shift+1)])

def mse(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred)

def count_model_coef_rmse(model, parsing_results):
    return mse(get_coef_matrix_from_parsing_results(parsing_results), model.coef_matrix.squeeze())