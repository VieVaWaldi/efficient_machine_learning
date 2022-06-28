import torch

## Trains the given model.
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied (single epoch).
#  @param io_model model which is trained.
#  @param io_optimizer used optimizer.
#  @param i_n_batches_abort abort the training after the given number of batches.
#  @return summed loss over all training samples.
def train( i_loss_func,
           io_data_loader,
           io_model,
           io_optimizer,
           l_size,
           l_rank,
           i_n_batches_abort = None ):
  # get padding
  l_pad = io_model.m_padding

  # switch model to train mode
  io_model.train()

  l_loss_total = 0

  for l_batch_id, (l_x, l_y) in enumerate( io_data_loader ):
    if( torch.cuda.is_available() ):
      l_x = l_x.to( torch.device('cuda') )
      l_y = l_y.to( torch.device('cuda') )

    # compute prediction and loss
    l_prediction = io_model( l_x )

    # remove border of labels which are not predicted by u-net
    l_y = l_y[:,:,l_pad:-l_pad,l_pad:-l_pad]

    # remove spatial dimensions which are 1
    if( l_y.size()[0] == 1 ):
      l_y[0] = l_y[0].squeeze()
    else:
      l_y = l_y.squeeze()

    # compute loss
    l_loss = i_loss_func( l_prediction, l_y )
    #l_loss_total += l_loss.item()
    
    l_loss_total = torch.tensor(l_loss.item())
    torch.distributed.all_reduce(l_loss_total, op = torch.distributed.ReduceOp.SUM)
    l_loss_total = l_loss_total / float(l_size)

    # backprop
    io_optimizer.zero_grad()
    l_loss.backward()
    io_optimizer.step()

    # inform user on status
    if( (l_batch_id+1) % 250 == 0 ):
      print_rank( '  reached batch #'+ str(l_batch_id+1) )
      print_rank( '    loss: '+ str(l_loss.item()) )
  
    if( i_n_batches_abort != None ):
      if( l_batch_id+1 >= i_n_batches_abort ):
        print_rank( '  aborting epoch early due to upper limit on #batches' )
        break

  print_rank( l_rank,'  processed '+ str(l_batch_id+1)+ ' batches' )

  return l_loss_total

def print_rank(l_rank,input):
    if l_rank == 0:
         print(input)
