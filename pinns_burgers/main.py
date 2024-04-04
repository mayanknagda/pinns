"""
In this script we try to implement a physics-informed neural net to find the solution for burgers equation.

This is a 
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

e = 1e-6


### Get data ###
def gen_testdata():
    "Taken from https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.html"
    data = np.load("Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    # X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return np.ravel(xx).reshape(-1, 1), np.ravel(tt).reshape(-1, 1), y


### plot data ###
def plot_data(x, t, y):
    fig = plt.figure(figsize=(10, 3))
    plt.scatter(t, x, c=y)
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()


class Net(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )

    def forward(self, *v):
        input = torch.concat(v, dim=1)
        return self.nn(input)


def f(x, t, net):
    x.requires_grad = True
    t.requires_grad = True
    u = net(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]
    return u_t + (u * u_x) - ((0.01 / np.pi) * u_xx)


def phy_data(batch_size=288):
    x = torch.from_numpy(
        np.random.uniform(low=-1, high=1 + e, size=(batch_size, 1))
    ).float()
    t = torch.from_numpy(
        np.random.uniform(low=0, high=1 + e, size=(batch_size, 1))
    ).float()
    return x, t


def bc_data(batch_size=288):
    batch_size //= 3
    # 1
    x1 = np.random.uniform(low=-1, high=1 + e, size=(batch_size, 1))
    t1 = np.zeros_like(x1)
    y1 = -1 * np.sin(np.pi * x1)
    # 2
    t2 = np.random.uniform(low=0, high=1 + e, size=(batch_size, 1))
    x2 = np.ones_like(t2)
    y2 = np.zeros_like(x2)
    # 3
    x3 = x2 * -1
    t3 = t2
    y3 = y2
    # overall
    x = torch.from_numpy(np.concatenate((x1, x2, x3))).float()
    t = torch.from_numpy(np.concatenate((t1, t2, t3))).float()
    y = torch.from_numpy(np.concatenate((y1, y2, y3))).float()
    return x, t, y


## hparams
optm = torch.optim.AdamW
loss = nn.MSELoss()
lr = 1e-3
batch_size = 288
device = "cpu"
iterations = 20000

net = Net()
net.to(device)
opt = optm(net.parameters(), lr)

for i in range(iterations):
    x, t = phy_data(batch_size=batch_size)
    x, t = x.to(device), t.to(device)
    f_pred = f(x, t, net)
    x, t, u = bc_data(batch_size=batch_size)
    x, t, u = x.to(device), t.to(device), u.to(device)
    b_pred = net(x, t)
    opt.zero_grad()
    loss_i = loss(f_pred, torch.zeros_like(f_pred).float())
    loss_i += loss(b_pred, u)
    loss_i /= 2
    loss_i.backward()
    opt.step()
    if i % 1000 == 0:
        print(f"for i: {i+1}, l: {loss_i.item()}")

# plot_data(x, t, u)
x, t, y = gen_testdata()
u = (
    net(torch.from_numpy(x).float().to(device), torch.from_numpy(t).float().to(device))
    .detach()
    .cpu()
    .float()
    .numpy()
)
plot_data(x, t, u)
plot_data(x, t, np.abs(y - u))
print(f"The mae is: {mean_absolute_error(y, u)}")
