import numpy as np

class PACF(object):
    def __init__(self, time_s):
        self.time_s = np.array(time_s)
        self.pacf_results = {}
        self.time_s_mean = self.time_s.mean()
        self.time_s_var = self.count_var()
        
    def count_var(self):
        return  ((self.time_s - self.time_s_mean)**2).sum() / (self.time_s.shape[0] - 1)
        
    def count_r(self, s):
        to_div = sum( (self.time_s[i] - self.time_s_mean) * (self.time_s[i-s] - self.time_s_mean) for i in range(s, len(self.time_s)))
        divider = (self.time_s_var)*(len(self.time_s) - 1)
        return to_div / divider
        
    def F(self, k,j):
        
        if (k,j) in self.pacf_results.keys():
            return self.pacf_results[(k,j)]
        
            
        if k == j == 1 :
            f = self.count_r(1)
        else:
            to_div = self.count_r(k) - sum(self.F(k - 1, i) * self.count_r(k - i) for i in range(1, k))
            divider = 1 - sum(self.F(k - 1, i) * self.count_r(i) for i in range(1, k))

            f = to_div/divider

        if k == j :
            self.pacf_results[(k, j)] = f
            return self.pacf_results[(k, j)]
        else:
            self.pacf_results[(k, j)] = self.F(k - 1, j) - f * self.F(k - 1, k - j)
            return self.pacf_results[(k, j)]
        
    def simmetric_F(self, k):
        return self.F(k,k)