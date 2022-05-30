from config import cfg
from data.fashion_mnist_data import Fashion_mnist_data
from vis.fashion_mnist_visualization import Fashion_mnist_visualization

from mlp.model import Model
from mlp.trainer import Trainer
from mlp.tester import Tester

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

LOAD_MODEL = True

data = Fashion_mnist_data()
# data.save_random_image()

trainer = Trainer()
tester = Tester()
model = Model()


if LOAD_MODEL:
    model.load_state_dict(torch.load(cfg["paths"]["model"]))

loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=cfg["paras"]["lr"])

trainer.train(data.train_dataloader, model, loss_fn, optimizer)
info = tester.test(data.test_dataloader, model, loss_fn)

vis = Fashion_mnist_visualization()
vis.plot(10, 400, data.test_data, model, info)
