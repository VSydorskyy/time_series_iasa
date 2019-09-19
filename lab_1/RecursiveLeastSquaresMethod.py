import numpy as np


class RecursiveLeastSquaresMethod(object):
    def __init__(self, coef_matrix_shape, biased, coef_matrix_initilizer=0., beta=10):
        self.biased = biased
        if self.biased:
            coef_matrix_shape += 1
            
        self.coef_matrix = np.ones((coef_matrix_shape,1)) * coef_matrix_initilizer    
        self.p_matrix = np.identity(coef_matrix_shape)*beta
        
    def fit_one(self, x, y):
        if self.biased:
            x = np.concatenate([np.zeros(1), x], axis=0)
        
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=-1)
        self.p_matrix = self.p_matrix - ((self.p_matrix@x.T@x@self.p_matrix) / (1+x@self.p_matrix@x.T))
        self.coef_matrix = self.coef_matrix + self.p_matrix@x.T @ (y.T - x@self.coef_matrix)
        
        return self
    
    def predict(self, x):
        if self.biased:
            x = np.concatenate([np.zeros(1), x], axis=0)
            
        x = np.expand_dims(x, axis=0)
        return np.squeeze(x@self.coef_matrix)
    
    def predict_all(self, X, y):
        coefs_mass = []
        y_pred = []
        
        for i in range(X.shape[0]):
            self.fit_one(X[i,:], y[i])
            coefs_mass.append(self.coef_matrix.squeeze())
            y_pred.append(self.predict(X[i,:]).squeeze())
            
        return np.stack(y_pred), np.stack(coefs_mass)