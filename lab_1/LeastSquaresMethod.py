import numpy as np

class LeastSquaresMethod(object):
    def __init__(self, biased):
        self.coef_matrix = None
        self.rmse = None
        self.biased = biased
        
    def fit(self, X, y):
        if self.biased:
            X = np.concatenate([
                np.ones((X.shape[0],1)),
                X
            ], axis=1)
        
        y = np.expand_dims(y, axis=-1)
                
        self.coef_matrix = np.linalg.inv(X.T @ X)@X.T@y
        
        return self
    
    def predict(self, X):
        if self.biased:
            X = np.concatenate([
                np.ones((X.shape[0],1)),
                X
            ], axis=1)
            
        return X@self.coef_matrix