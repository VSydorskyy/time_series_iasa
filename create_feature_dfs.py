import pandas as pd

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