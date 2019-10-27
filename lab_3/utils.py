import numpy as np
import pandas as pd

from MA import moving_average

def build_lagged_features(s,lag=2,dropna=True):
    s = pd.Series(s)
    
    the_range=range(lag+1)
    res=pd.concat([s.shift(i) for i in the_range],axis=1)
    res.columns=['lag_%d' %i for i in the_range]
    
    if dropna:
        return res.dropna()
    else:
        return res
    
def build_ma_on_lags(time_s, lag, window_size=5, ma_type=None):
    lag_df = build_lagged_features(time_s, lag)
    result = {'ma_'+col:moving_average(lag_df[col], n=window_size, weights=ma_type) for col in lag_df.columns}
    return pd.DataFrame(result)