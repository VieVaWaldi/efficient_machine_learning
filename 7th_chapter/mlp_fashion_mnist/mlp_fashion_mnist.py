#!/usr/bin/python3
import torch
import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import torch.distributed

import time

import eml.mlp.model
import eml.mlp.trainer
import eml.mlp.tester
# import eml.vis.fashion_mnist

# ------------------- 7.2 Addon ------------------- #
# Init MPI
torch.distributed.init_process_group('mpi')
l_rank = torch.distributed.get_rank()
l_size = torch.distributed.get_world_size()

def print_ol(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)
# ------------------------------------------------- #

print_ol("################################")
print_ol("# Welcome to EML's MLP example #")
print_ol("################################")

start_time = time.time()
print_ol("... Starting time measurement")

# set up datasets
print_ol('setting up datasets')
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

NUM_EPOCHS = 25
BATCH_SIZE = 64  # is mini batch size
MICRO_BATCH_SIZE = BATCH_SIZE // l_size # batch size per node

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
print_ol('initializing data loaders, ')
l_data_loader_train = torch.utils.data.DataLoader(l_data_train,
                                                 batch_sampler=l_batch_sampler_train)
l_data_loader_test = torch.utils.data.DataLoader(l_data_test,
                                                 batch_sampler=l_batch_sampler_test)
# ------------------------------------------------- #

# set up model, loss function and optimizer
print_ol('setting up model, loss function and optimizer')
l_model = eml.mlp.model.Model()
l_loss_func = torch.nn.CrossEntropyLoss()
l_optimizer = torch.optim.SGD(l_model.parameters(),
                              lr=1E-3)
print_ol(l_model)

# train for the given number of epochs
# For training loss and test with all reduce
for l_epoch in range(NUM_EPOCHS):
    print_ol('training epoch #' + str(l_epoch+1))
    l_loss_train = eml.mlp.trainer.train(l_loss_func,
                                         l_size,
                                         l_data_loader_train,
                                         l_model,
                                         l_optimizer)
    # ------------------- 7.3 Addon ------------------- #
    # AllReduce für loss train
    
    print_ol(f'  training loss: {l_loss_train:.3f}')
    l_loss_train_tensor = torch.tensor(l_loss_train)
    torch.distributed.all_reduce(l_loss_train_tensor,
                                 op=torch.distributed.ReduceOp.SUM)
    print_ol(f'  training loss reduce: {l_loss_train_tensor / l_size:.3f}')
    # ------------------------------------------------- #

    l_loss_test, l_n_correct_test = eml.mlp.tester.test(l_loss_func,
                                                        l_data_loader_test,
                                                        l_model)
    l_accuracy_test = l_n_correct_test / (len(l_data_loader_test.dataset) / l_size)

    # ------------------- 7.3 Addon ------------------- #
    # AllReduce und average für loss und accuracy test

    print_ol(f'  test loss: {l_loss_test:.3f}')
    print_ol(f'  test accuracy: {l_accuracy_test:.3f}')

    l_loss_test_tensor = torch.tensor(l_loss_test)
    torch.distributed.all_reduce(l_loss_test_tensor,
                                 op=torch.distributed.ReduceOp.SUM)
    l_accuracy_test_tensor = torch.tensor(l_accuracy_test)
    torch.distributed.all_reduce(l_accuracy_test_tensor,
                                 op=torch.distributed.ReduceOp.SUM)

    print_ol(f'  test loss reduce: {l_loss_test_tensor / l_size:.3f}')
    print_ol(f'  test accuracy reduce: {l_accuracy_test_tensor / l_size:.3f}')
    # ------------------------------------------------- #

# ------------------- 7.2 Addon ------------------- #
# Could be called for rank 0 only, but lets ignore it :)

    # visualize results of intermediate model every 100 epochs
    # if((l_epoch+1) % 100 == 0):
    #     l_file_name = 'test_dataset_epoch_' + str(l_epoch+1) + '.pdf'
    #     print_ol('  visualizing intermediate model w.r.t. test dataset: ' + l_file_name)
    #     eml.vis.fashion_mnist.plot(0,
    #                                250,
    #                                l_data_loader_test,
    #                                l_model,
    #                                l_file_name)


# visualize results of final model
# l_file_name = 'test_dataset_final.pdf'
# print_ol('visualizing final model w.r.t. test dataset:' + l_file_name)
# eml.vis.fashion_mnist.plot(0,
#                            250,
#                            l_data_loader_test,
#                            l_model,
#                            l_file_name)
# ------------------------------------------------- #

print_ol(f"\n... Run took {time.time() - start_time:.0f} seconds.\n")

print_ol("#############")
print_ol("# Finished! #")
print_ol("#############")
