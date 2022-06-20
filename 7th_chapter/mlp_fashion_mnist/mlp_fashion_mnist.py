#!/usr/bin/python3
import torch
import torchvision.datasets
import torchvision.transforms
import torch.utils.data

import eml.mlp.model
import eml.mlp.trainer
import eml.mlp.tester
import eml.vis.fashion_mnist

print( "################################" )
print( "# Welcome to EML's MLP example #" )
print( "################################" )


# set up datasets
print( 'setting up datasets')
l_data_train = torchvision.datasets.FashionMNIST( root      = "data/fashion_mnist",
                                                  train     = True,
                                                  download  = True,
                                                  transform = torchvision.transforms.ToTensor() )

l_data_test = torchvision.datasets.FashionMNIST( root      = "data/fashion_mnist",
                                                 train     = False,
                                                 download  = True,
                                                 transform = torchvision.transforms.ToTensor() )

# init data loaders
print( 'initializing data loaders' )
l_data_loader_train = torch.utils.data.DataLoader( l_data_train,
                                                   batch_size = 64 )
l_data_loader_test  = torch.utils.data.DataLoader( l_data_test,
                                                   batch_size = 64 )

# set up model, loss function and optimizer
print( 'setting up model, loss function and optimizer' )
l_model = eml.mlp.model.Model()
l_loss_func = torch.nn.CrossEntropyLoss()
l_optimizer = torch.optim.SGD( l_model.parameters(),
                               lr = 1E-3 )
print( l_model )

# train for the given number of epochs
l_n_epochs = 25
for l_epoch in range( l_n_epochs ):
  print( 'training epoch #' + str(l_epoch+1) )
  l_loss_train = eml.mlp.trainer.train( l_loss_func,
                                        l_data_loader_train,
                                        l_model,
                                        l_optimizer )
  print( '  training loss:', l_loss_train )

  l_loss_test, l_n_correct_test = eml.mlp.tester.test( l_loss_func,
                                                       l_data_loader_test,
                                                       l_model )
  l_accuracy_test = l_n_correct_test / len(l_data_loader_test.dataset)
  print( '  test loss:', l_loss_test )
  print( '  test accuracy:', l_accuracy_test )

  # visualize results of intermediate model every 10 epochs
  if( (l_epoch+1) % 10 == 0 ):
    l_file_name =  'test_dataset_epoch_' + str(l_epoch+1) + '.pdf'
    print( '  visualizing intermediate model w.r.t. test dataset: ' + l_file_name )
    eml.vis.fashion_mnist.plot( 0,
                                250,
                                l_data_loader_test,
                                l_model,
                                l_file_name )

# visualize results of final model
l_file_name = 'test_dataset_final.pdf'
print( 'visualizing final model w.r.t. test dataset:', l_file_name )
eml.vis.fashion_mnist.plot( 0,
                            250,
                            l_data_loader_test,
                            l_model,
                            l_file_name )

print( "#############" )
print( "# Finished! #" )
print( "#############" )