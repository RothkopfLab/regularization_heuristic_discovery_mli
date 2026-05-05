# The code in this file is adapted from https://github.com/marcelbinz/HeuristicsFromBMLI/blob/master/environments.py
import math

import torch
from pyro.distributions.lkj import LKJCorrCholesky
from torch.distributions import MultivariateNormal


class PairedComparison():
    def __init__(self, num_inputs=4, num_targets=1, direction=False, ranking=False, dichotomized=False, var=0.01, feature_means=0, theta=1.0, cov_prior_eta=2.0, weight_std=1.0):
        self.num_inputs = num_inputs
        self.num_targets = num_targets

        self.direction = direction
        self.ranking = ranking
        self.dichotomized = dichotomized

        self.sigma = math.sqrt(var)
        if (not torch.is_tensor(feature_means)):
            feature_means = torch.ones(self.num_inputs)*feature_means
        self.feature_means = feature_means
        self.theta = theta * torch.ones(num_inputs)
        self.cov_prior = LKJCorrCholesky(num_inputs, eta=cov_prior_eta * torch.ones(1))
        self.weight_std = weight_std
    
    def get_batch(self, batch_size, time_steps, device=None):
        n_weights = batch_size*self.num_inputs
        self.weights = torch.normal(torch.zeros(n_weights), torch.ones(n_weights) * self.weight_std).reshape(batch_size, self.num_inputs)
        if self.direction:
            self.weights = self.weights.abs()
        if self.ranking:
            self.weights = self.weights[torch.arange(self.weights.shape[0])[:,None], torch.argsort(torch.abs(self.weights))]
        not_found = True
        while not_found:
            L = self.cov_prior.sample([batch_size])
            if not torch.isnan(L).any():
                not_found = False
        L = L.squeeze()
        feature_dist = MultivariateNormal(self.feature_means.repeat((batch_size,1)), scale_tril=torch.matmul(torch.diag(torch.sqrt(self.theta)).repeat((batch_size, 1, 1)), L))
        inputs_a = feature_dist.sample([time_steps])
        inputs_b = feature_dist.sample([time_steps])
        inputs = inputs_a - inputs_b

        targets = torch.bernoulli(0.5 * torch.erfc(-(self.weights * inputs).sum(-1, keepdim=True) / (2 * self.sigma)))
        return inputs.detach().to(device), targets.detach().to(device), inputs_a.detach().to(device), inputs_b.detach().to(device)

if __name__ == '__main__':
    dl = PairedComparison(4, dichotomized=False, direction=False, ranking=False)
    for i in range(1):
        inputs, targets, _, _ = dl.get_batch(1000, 10)
        print(targets.mean())
        print(inputs.std() ** 2)
