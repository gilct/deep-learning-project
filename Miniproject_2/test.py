from model import *
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_grad_enabled(True)

SEED = 2022
torch.manual_seed(SEED)

class LinearTorchTest(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

class Testing(unittest.TestCase):
    def test_linear(self):
        """Compare our implementation with baseline torch model."""

        # Define dimensions of data and linear layers (1 input, 1 hidden, 1 output)
        in_dim, hidden_dim, out_dim = 50, 20, 15
        n_samples = 200

        # Parameters of distribution of inputs and targets
        mean, std = 0, 20
        unif_lower, unif_upper = 10, 15
        train_input = torch.empty(n_samples, in_dim).normal_(mean, std)
        train_targets = torch.empty(n_samples, out_dim).uniform_(unif_lower,unif_upper)

        # Define our model and torch model with same initial value of all weights (for reproducibility)
        init_val = 0.05
        model_no_torch = Sequential(Linear(in_dim, hidden_dim, init_val=init_val),
                                    Sigmoid(),
                                    Linear(hidden_dim, out_dim, init_val=init_val))

        model_torch = LinearTorchTest(in_dim, hidden_dim, out_dim)
        with torch.no_grad():
            model_torch.fc1.weight = nn.Parameter(torch.full_like(model_torch.fc1.weight, init_val))
            model_torch.fc1.bias = nn.Parameter(torch.full_like(model_torch.fc1.bias, init_val))
            model_torch.fc2.weight = nn.Parameter(torch.full_like(model_torch.fc2.weight, init_val))
            model_torch.fc2.bias = nn.Parameter(torch.full_like(model_torch.fc2.bias, init_val))

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 1, 20

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        # Train
        for e in range(nb_epochs):
            for input, targets in zip(train_input.split(batch_size),
                                      train_targets.split(batch_size)):

                output_no_torch = model_no_torch.forward(input)
                output_torch = model_torch(input)

                # Retain grad for comparing MSE gradients, see below
                # output_torch.retain_grad()

                # Store statistics of the output for comparison
                stats_no_torch = (output_no_torch.mean().item(), output_no_torch.std().item())
                stats_torch = (output_torch.mean().item(), output_torch.std().item())

                loss_no_torch = criterion_no_torch.forward(output_no_torch, targets)
                loss_torch = criterion_torch(output_torch, targets)   

                optimizer_no_torch.zero_grad()
                optimizer_torch.zero_grad() 

                model_no_torch.backward(criterion_no_torch.backward())
                loss_torch.backward()

                # Compare gradients of MSE (make sure .retain_grad() is called on output_torch)
                # print(criterion_no_torch.backward())
                # print(output_torch.grad)

                optimizer_no_torch.step()
                optimizer_torch.step()

                # Assertions
                self.assertAlmostEqual(loss_no_torch.item(), loss_torch.item(), places=5, msg="Equal losses")
                self.assertAlmostEqual(stats_no_torch[0], stats_torch[0], places=5, msg="Equal mean of preds")
                self.assertAlmostEqual(stats_no_torch[1], stats_torch[1], places=6, msg="Equal std of preds")

if __name__ == '__main__':
    unittest.main()