import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from MA import moving_average
from PACF import PACF
from utils import build_lagged_features, build_ma_on_lags

class ARMA(object):
    def __init__(self, ma_window_size=5, ma_type=None, ar_thresh_coef=0.3, ma_thresh_coef=0.2, max_coef_to_inspect=12, on_residuals=True):
        self.ma_window_size = ma_window_size
        self.ma_type = ma_type
        self.ar_thresh_coef = ar_thresh_coef
        self.ma_thresh_coef = ma_thresh_coef
        self.max_coef_to_inspect = max_coef_to_inspect
        self.on_residuals = on_residuals
        
        self.pacf_ar = None
        self.initial_lin_reg_coefs = None
        self.ar_results = None
        
        self.pacf_ma = None
        
        self.arma_coefs = None
        
    def get_important_coef(self, time_s, algorithm):
        pacf_object = PACF(time_s)
        coef_buffer = pd.Series([pacf_object.simmetric_F(i) for i in range(self.max_coef_to_inspect)])
        if algorithm == 'ar':
            self.pacf_ar = coef_buffer
            return coef_buffer[coef_buffer.abs() > self.ar_thresh_coef].index.max()
        elif algorithm == 'ma':
            self.pacf_ma = coef_buffer
            return coef_buffer[coef_buffer.abs() > self.ma_thresh_coef].index.max()
        else:
            raise ValueError('Not Implemented')
    
    def fit_predict(self, X, endogen=None):
        X = np.array(X)
        
        amount_of_ar_coefs = self.get_important_coef(X, algorithm='ar')
        lagged_ar_df = build_lagged_features(X, amount_of_ar_coefs)
        if endogen is not None:
            lagged_ar_df = [lagged_ar_df]
            for col in endogen.columns:
                temp_df = build_lagged_features(endogen[col], amount_of_ar_coefs)
                temp_df.columns = [col+'_'+n for n in temp_df.columns]
                lagged_ar_df.append(temp_df)
            
            lagged_ar_df = pd.concat(lagged_ar_df, axis=1)
        
        lin_reg = LinearRegression()
        
        lin_reg.fit(lagged_ar_df.drop(columns='lag_0'), lagged_ar_df['lag_0'])
        
        self.initial_lin_reg_coefs = [lin_reg.intercept_] + list(lin_reg.coef_)
        lin_reg_pred = lin_reg.predict(lagged_ar_df.drop(columns='lag_0'))
        self.ar_results = lin_reg_pred, lagged_ar_df['lag_0']
        
        if self.on_residuals:
            residuals = lagged_ar_df['lag_0'] - lin_reg_pred
            amount_of_ma_coefs = self.get_important_coef(residuals, algorithm='ma')
            lagged_ma_df = build_ma_on_lags(residuals, lag=amount_of_ma_coefs, window_size=self.ma_window_size, ma_type=self.ma_type)
        else:
            ma_0 = moving_average(X, n=self.ma_window_size, weights=self.ma_type)
            amount_of_ma_coefs = self.get_important_coef(ma_0, algorithm='ma')
            lagged_ma_df = build_ma_on_lags(X, lag=amount_of_ma_coefs, window_size=self.ma_window_size, ma_type=self.ma_type)
            
        
        time_s_m_ma_0 = X[X.shape[0] - lagged_ma_df.shape[0]:] - lagged_ma_df['ma_lag_0']
                
        ar_and_ma_df = pd.concat([lagged_ar_df.iloc[lagged_ar_df.shape[0] - lagged_ma_df.shape[0]:].reset_index(drop=True),
                                  lagged_ma_df.reset_index(drop=True)], axis=1)
        
        lin_reg.fit(ar_and_ma_df.drop(columns=['lag_0','ma_lag_0']), time_s_m_ma_0)
        
        if endogen is not None:
            self.arma_coefs = [lin_reg.intercept_] + list(lin_reg.coef_[:amount_of_ar_coefs*endogen.shape[1]]) + [1] + list(lin_reg.coef_[amount_of_ar_coefs*endogen.shape[1]:])
        else:
            self.arma_coefs = [lin_reg.intercept_] + list(lin_reg.coef_[:amount_of_ar_coefs]) + [1] + list(lin_reg.coef_[amount_of_ar_coefs:])
        
        return lin_reg.predict(ar_and_ma_df.drop(columns=['lag_0','ma_lag_0'])) + lagged_ma_df['ma_lag_0'], ar_and_ma_df['lag_0']