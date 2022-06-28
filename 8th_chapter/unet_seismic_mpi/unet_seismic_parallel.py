#!/usr/bin/python3
import torch
import torchvision.datasets
import torchvision.transforms
import torch.utils.data

import numpy as np
import time

import eml.data.seismic
import eml.unet.model
import eml.unet.trainer
import eml.unet.tester


def print_rank(input):
    if l_rank == 0:
         print(input)

#Init MPI
torch.distributed.init_process_group("mpi")
l_rank  = torch.distributed.get_rank()
l_size = torch.distributed.get_world_size()

start_time = time.time()

# config
# 'n_init_channels':   64,
# 'kernel_size':        3,
# 'n_layers_per_block': 2,
# 'n_levels':           4

l_config = { 'unet': { 'n_init_channels':   32,
                       'kernel_size':        3,
                       'n_layers_per_block': 2,
                       'n_levels':           2 },
             'train': { 'data':           { 'seismic':      'data/data_train.npz',
                                            'labels':       'data/labels_train.npz',
                                            # data size is reduced for three levels in unet
                                            'sample_shape': (1004, 1, 588),
                                            'subset':       ( (0, 1004),
                                                              (0,  750),
                                                              (0,  588) )  },
                        'n_epochs':       1000,
                        'n_epochs_print': 5,
                        'n_batch_abort':  5000,
                        'batch_size':     2*l_size},
             'test':  { 'data':           { 'seismic':      'data/data_train.npz',
                                            'labels':       'data/labels_train.npz',
                                             # data size is reduced for three levels in unet
                                            'sample_shape': (1004, 1, 588),
                                            'subset':       ( (  0, 1004),
                                                              ( 751, 782),
                                                              (  0,  588) ) },
                        'batch_size':      1*l_size} }

print_rank( "##############################################" )
print_rank( "# Welcome to EML's U-Net for seismic example #" )
print_rank( "##############################################" )
if( torch.cuda.is_available() ):
  l_n_cuda_devices = torch.cuda.device_count();
  print_rank( 'CUDA devices:', l_n_cuda_devices )
  for l_de in range(l_n_cuda_devices):
    print_rank( '  ', torch.cuda.get_device_name(l_de) )
else:
  print_rank( 'could not find a CUDA device' )

print_rank( 'printing configuration:' )
try:
  import json
  print_rank( json.dumps( l_config,
                     indent = 2 ) )
except:
  print_rank( '  json module missing, continuing' )

print_rank( '********************')
print_rank( '* assembling U-Net *')
print_rank( '********************')

# construct U-Net and print info
l_unet2d = eml.unet.model.Unet2d( i_n_init_channels    = l_config['unet']['n_init_channels'],
                                  i_kernel_size        = l_config['unet']['kernel_size'],
                                  i_n_layers_per_block = l_config['unet']['n_layers_per_block'],
                                  i_n_levels           = l_config['unet']['n_levels'] )

if( torch.cuda.is_available() ):
  l_unet2d = l_unet2d.to( torch.device('cuda') )

print_rank( l_unet2d )

# set U-Net to training mode
l_unet2d.train()

# loss function and optimizer
l_loss_func = torch.nn.CrossEntropyLoss()
l_optimizer = torch.optim.Adam( l_unet2d.parameters(),
                                lr=1E-2 )

print_rank( '*****************')
print_rank( '* prepping data *')
print_rank( '*****************')
# training dataset
print_rank( 'loading training dataset' )


l_data_set_train = eml.data.seismic.SeismicDataSet( l_config['train']['data']['seismic'],
                                                    l_config['train']['data']['labels'],
                                                    i_item_shape = l_config['train']['data']['sample_shape'],
                                                    i_subset     = l_config['train']['data']['subset'],
                                                    l_rank = l_rank )

print_rank( 'loading test data set' )
l_data_set_test = eml.data.seismic.SeismicDataSet( l_config['test']['data']['seismic'],
                                                   l_config['test']['data']['labels'],
                                                   i_item_shape = l_config['test']['data']['sample_shape'],
                                                   i_subset     = l_config['test']['data']['subset'],
                                                   l_rank = l_rank )

print_rank( 'deriving mean and standard deviation of training data' )

l_mean_train = l_data_set_train.getMean()
l_std_dev_train = l_data_set_train.getStdDev()

print_rank( '  mean: '+ str(l_mean_train)    )
print_rank( '  std: '+  str(l_std_dev_train) )
print_rank( 'normalizing training and test data' )

l_data_set_train.normalize( l_mean_train,
                            l_std_dev_train )
l_data_set_test.normalize( l_mean_train,
                           l_std_dev_train )

#_________________________________________________________________training dataloader__________________________________________________________________________

l_dist_sampler_train = torch.utils.data.DistributedSampler(
    l_data_set_train,
    num_replicas=l_size,
    rank=l_rank,
    shuffle=True,
    drop_last=False
)

l_dist_sampler_test = torch.utils.data.DistributedSampler(
    l_data_set_test,
    num_replicas=l_size,
    rank=l_rank,
    shuffle=False,
    drop_last=False
)


MICRO_BATCH_SIZE_TRAIN = l_config['train']['batch_size'] // l_size
MICRO_BATCH_SIZE_TEST  = l_config['test']['batch_size'] // l_size

# 2 batch sampler for train and test data with micro batch size
# @batch_size, we use our micro batch size for the nodes
# @drop_last, drops last batch if smaller than micro_batch_size

l_batch_sampler_train = torch.utils.data.BatchSampler(
    sampler=l_dist_sampler_train,
    batch_size=MICRO_BATCH_SIZE_TRAIN,
    drop_last=False
)

l_batch_sampler_test = torch.utils.data.BatchSampler(
    sampler=l_dist_sampler_test,
    batch_size=MICRO_BATCH_SIZE_TEST,
    drop_last=False
)

print_rank( 'initializing data loaders' )

l_data_loader_train = torch.utils.data.DataLoader( l_data_set_train,
                                                   batch_sampler = l_batch_sampler_train )

l_data_loader_test = torch.utils.data.DataLoader( l_data_set_test,
                                                  batch_sampler = l_batch_sampler_test )

print_rank( '************')
print_rank( '* training *')
print_rank( '************')

# train for the given number of epochs
for l_epoch in range( l_config['train']['n_epochs'] ):
  
  print_rank( 'training epoch'+ str(l_epoch+1) )
  #print_rank(len(l_data_loader_train))
  l_loss_train = eml.unet.trainer.train( l_loss_func,
                                         l_data_loader_train,
                                         l_unet2d,
                                         l_optimizer,
                                         l_size = l_size,
                                         l_rank = l_rank,
                                         i_n_batches_abort = l_config['train']['n_batch_abort'] )
  
  print_rank( '  training loss:'+ str(l_loss_train) )
  print_rank( 'applying net to test data' )

  l_loss_test, l_n_correct_test, l_n_total_test = eml.unet.tester.test( l_loss_func,
                                                                        l_data_loader_test,
                                                                        l_unet2d,
                                                                        l_size = l_size )
  #if l_epoch == 0:  
  #  print("l_n_correct_test: ", l_n_correct_test)
  #  print("l_n_total_test: ", l_n_total_test)
  
  #100 * correct_train / total_train
  l_accuracy_test = l_n_correct_test / l_n_total_test
  
  
  print_rank( '  test loss:'+ str(l_loss_test) )
  print_rank( '  test accuracy:'+ str(l_accuracy_test) )

  # do an intermediate evaluation on the test data
  if( (l_epoch+1) % l_config['train']['n_epochs_print'] == 0 ):
    l_unet2d.eval()
    with torch.no_grad():
      l_data_raw = l_data_set_test.m_data[:,0,:].squeeze().reshape( 1, 1, 1004, 588 )
      l_data_raw = torch.Tensor( l_data_raw )
      if( torch.cuda.is_available() ):
        l_data_raw = l_data_raw.to( torch.device('cuda') )
      l_prediction = l_unet2d.forward( l_data_raw )
      l_prediction = l_prediction.argmax(1).squeeze()
      l_prediction = l_prediction.to('cpu')

      # pad invalid values, i.e., -1, to match input data
      l_pad = l_unet2d.m_padding
      l_prediction = torch.nn.functional.pad( l_prediction,
                                              (l_pad, l_pad, l_pad, l_pad),
                                              value = -1 )

    #import matplotlib.pyplot as plt
    #from matplotlib.backends.backend_pdf import PdfPages

    #l_pdf_pages = PdfPages('epoch_' + str(l_epoch+1) + '.pdf')
    #plt.figure( figsize=(12, 10) )
    #plt.subplot(1, 3, 1)
    #plt.imshow( l_data_set_test.m_data[:,10,:],
    #            cmap = 'gray' )
    #plt.subplot(1, 3, 2)
    #plt.imshow( l_data_set_test.m_data[:,10,:],
    #            cmap = 'gray' )
    #plt.imshow( l_prediction,
    #            alpha=0.5,
    #            vmin=-1,
    #            cmap = 'plasma' )
    #plt.subplot(1, 3, 3)
    #plt.imshow( l_data_set_test.m_data[:,10,:],
    #            cmap = 'gray' )
    #plt.imshow( l_data_set_test.m_labels[:,10,:],
    #            alpha=0.5,
    #            vmin=-1,
    #            cmap = 'plasma' )
    #l_pdf_pages.savefig()
    #l_pdf_pages.close()

    l_unet2d.train()

print_rank(f"\n... Run took {time.time() - start_time:.0f} seconds.\n")
