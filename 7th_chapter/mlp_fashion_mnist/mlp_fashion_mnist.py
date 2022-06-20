#!/usr/bin/python3
import sys

import torch
import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import torch.distributed

import eml.mlp.model
import eml.mlp.trainer
import eml.mlp.tester
# import eml.vis.fashion_mnist

print("################################")
print("# Welcome to EML's MLP example #")
print("################################")

# ------------------- 7.2 Addon ------------------- #
# Init MPI
torch.distributed.init_process_group('mpi')
l_rank = torch.distributed.get_rank()
l_size = torch.distributed.get_world_size()

# cheesy overload


def print(msg):
    if l_rank == 0:
        print(msg, file=sys.stdout)
# ------------------------------------------------- #


# set up datasets
print('setting up datasets')
l_data_train = torchvision.datasets.FashionMNIST(root="data/fashion_mnist",
                                                 train=True,
                                                 download=True,
                                                 transform=torchvision.transforms.ToTensor())

l_data_test = torchvision.datasets.FashionMNIST(root="data/fashion_mnist",
                                                train=False,
                                                download=True,
                                                transform=torchvision.transforms.ToTensor())

# ------------------- 7.2 Addon ------------------- #
# 2 distributed data sampler for train and test data

BATCH_SIZE = 64  # same as mini batch size
MICRO_BATCH_SIZE = BATCH_SIZE // l_size

# @num_replicas, number of running nodes
# @rank, index of current node
# @shuffle, shuffle data, same for all nodes
# @drop_last, drops last batch if smaller than batch_size

l_dist_sampler_train = torch.utils.data.DistributedSampler(
    l_data_train,
    num_replicas=l_size,
    rank=l_rank,
    shuffle=False,
    drop_last=False
)

l_dist_sampler_test = torch.utils.data.DistributedSampler(
    l_data_test,
    num_replicas=l_size,
    rank=l_rank,
    shuffle=False,
    drop_last=False
)

# 2 batch sampler for train and test data with micro batch size

# @batch_size, we use our micro batch size for the nodes
# @drop_last, drops last batch if smaller than micro_batch_size

l_batch_sampler_train = torch.utils.data.BatchSampler(
    sampler=l_dist_sampler_train,
    batch_size=MICRO_BATCH_SIZE,
    drop_last=False
)

l_batch_sampler_test = torch.utils.data.BatchSampler(
    sampler=l_dist_sampler_test,
    batch_size=MICRO_BATCH_SIZE,
    drop_last=False
)

# init data loaders
print('initializing data loaders, ')
l_data_loader_train = torch.utils.data.DataLoader(l_data_train,
                                                  batch_sampler=l_batch_sampler_train)
l_data_loader_test = torch.utils.data.DataLoader(l_data_test,
                                                 batch_sampler=l_batch_sampler_test)
# ------------------------------------------------- #

# set up model, loss function and optimizer
print('setting up model, loss function and optimizer')
l_model = eml.mlp.model.Model()
l_loss_func = torch.nn.CrossEntropyLoss()
l_optimizer = torch.optim.SGD(l_model.parameters(),
                              lr=1E-3)
print(l_model)

# train for the given number of epochs
# For training loss and test with all reduce
l_n_epochs = 25
for l_epoch in range(l_n_epochs):
    print('training epoch #' + str(l_epoch+1))
    l_loss_train = eml.mlp.trainer.train(l_loss_func,
                                         l_data_loader_train,
                                         l_model,
                                         l_optimizer)
    print('  training loss:', l_loss_train)

    l_loss_test, l_n_correct_test = eml.mlp.tester.test(l_loss_func,
                                                        l_data_loader_test,
                                                        l_model)
    l_accuracy_test = l_n_correct_test / len(l_data_loader_test.dataset)
    print('  test loss:', l_loss_test)
    print('  test accuracy:', l_accuracy_test)

# ------------------- 7.2 Addon ------------------- #
# Could be called only for rank 0, but lets ignore it :)

    # visualize results of intermediate model every 100 epochs
    # if((l_epoch+1) % 100 == 0):
    #     l_file_name = 'test_dataset_epoch_' + str(l_epoch+1) + '.pdf'
    #     print('  visualizing intermediate model w.r.t. test dataset: ' + l_file_name)
    #     eml.vis.fashion_mnist.plot(0,
    #                                250,
    #                                l_data_loader_test,
    #                                l_model,
    #                                l_file_name)


# visualize results of final model
# l_file_name = 'test_dataset_final.pdf'
# print('visualizing final model w.r.t. test dataset:', l_file_name)
# eml.vis.fashion_mnist.plot(0,
#                            250,
#                            l_data_loader_test,
#                            l_model,
#                            l_file_name)
# ------------------------------------------------- #

print("#############")
print("# Finished! #")
print("#############")
