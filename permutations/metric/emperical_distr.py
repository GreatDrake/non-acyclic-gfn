import torch
import numpy as np
import math

class EmpiricalDistributionMetric:
    def __init__(self, p, tau=2):
        self.p = p
        self.tau = tau

        self.buffer = []
        self.empirical_distr = torch.zeros(self.p + 1)
        self.true_distr = self.build_true_distr()
    
    def build_true_distr(self):
        distr = torch.tensor([math.exp(k/self.tau) * self.number_k_fixed(k) for k in range(self.p + 1)])
        distr = distr / distr.sum()
        return distr

    def number_k_fixed(self, k):
        if self.p == k:
            return 1.0
        res = 0
        k_fact = math.factorial(k)
        n_fact = math.factorial(self.p)
        for j in range(k, self.p+1):
            jk_fact = math.factorial(j - k)
            res += ((-1)**(j - k)) / (k_fact * jk_fact)
        return res * n_fact
    
    def add_to_buffer(self, states):
        self.buffer += [states]
        if len(self.buffer) > 100_000 // states.shape[0]:
            self.buffer = self.buffer[-100_000 // states.shape[0]:]

    def compute_metric(self):
        for batch in self.buffer:
            for state in batch:
                kfix_state = torch.sum(state == torch.arange(self.p)).to(int)
                self.empirical_distr[kfix_state] += 1
        
        self.empirical_distr = self.empirical_distr / self.empirical_distr.sum()
        res =  (torch.abs((self.empirical_distr - self.true_distr) / (self.true_distr + 1e-20))).mean()
        absolute = torch.abs((self.empirical_distr - self.true_distr)).mean()
        return res, absolute
        


    
    