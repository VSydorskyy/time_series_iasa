import pandas as pd
import numpy as np

from parse_input_file import read_file, parse_signals

def create_signal_df(signal_time_shift, signals, variable_name):
    signal_df = pd.DataFrame([signals[signal_time_shift:]] + \
                             [signals[i:-signal_time_shift+i] for i in reversed(range(signal_time_shift))]).transpose()
    signal_df.columns = [variable_name+'''(k)''']+[variable_name+'''(k-{})'''.format(i) for i in range(1, signal_time_shift+1)]
    return signal_df
    
def compose_signals_df(parsing_results):
    signal_time_shift = max(max(parsing_results['out_weights'].keys()), max(parsing_results['manage_weights'].keys()))
    out_signal_df = create_signal_df(signal_time_shift, parsing_results['out_signals'], 'y')
    manage_signal_df = create_signal_df(signal_time_shift, parsing_results['manage_signals'], 'v')
    
    return pd.concat([out_signal_df, manage_signal_df.iloc[:out_signal_df.shape[0],:]],axis=1)


def create_initial_input(a_coefs, b_coefs, time_series_shape=100, uniform_low=0, uniform_high=10, noise_std=1., manage=None):
    result = dict()
    
    result['manage_weights'] = {i:a_coefs[i] for i in range(len(a_coefs))}
    result['out_weights'] = {i:b_coefs[i] for i in range(len(b_coefs))}
    
    if manage is None:
        manage = np.random.uniform(low=uniform_low, high=uniform_high, size=(time_series_shape))
    else:
        manage = np.array(manage)
        time_series_shape = manage.shape[0]
        
    outs = list(manage[:len(a_coefs)-1] + np.random.normal(size=(len(a_coefs)-1), scale=noise_std))
    for i in range(len(a_coefs)-1,time_series_shape):
        outs.append(a_coefs[0] + sum(a_coefs[j]*outs[i-j] for j in range(1,len(a_coefs))) +\
                    sum(b_coefs[j]*manage[i-j] for j in range(1,len(b_coefs))) + np.random.normal(size=None, scale=noise_std))
        
    result['out_signals'] = outs
    result['manage_signals'] = list(manage)
    
    return result

def create_df_only_outs(path):
    signals = read_file(path)
    signals = parse_signals(signals)
    
    return pd.DataFrame({
        '''y(k)''':signals,
        '''v(k)''':signals
    })