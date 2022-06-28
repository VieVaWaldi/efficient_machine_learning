import torch

## Tests the model
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied (single epoch).
#  @param io_model model which is tested.
#  @return summed loss over all test samples, number of correctly predicted points, number of total predicted points.
def test( i_loss_func,
          io_data_loader,
          io_model ):
  # get padding
  l_pad = io_model.m_padding

  # switch model to eval-mode
  io_model.eval()

  l_loss_total = 0
  l_n_correct = 0
  l_n_total = 0

  with torch.no_grad():
    for (l_x, l_y) in io_data_loader:
      if( torch.cuda.is_available() ):
        l_x = l_x.to( torch.device('cuda') )
        l_y = l_y.to( torch.device('cuda') )

      # compute prediction
      l_prediction = io_model( l_x )

      # remove border of labels which are not predicted by u-Net
      l_y = l_y[:,:,l_pad:-l_pad,l_pad:-l_pad]

      # remove spatial dimensions which are 1
      # extra complexity keeps batch dimension alive
      for l_di in range( 1, len(l_y.size()) ):
        if( l_di < len(l_y.size()) ):
          l_y = l_y.squeeze(l_di)

      # compute loss
      l_loss = i_loss_func( l_prediction, l_y )

      # sum total loss and derive number of correct predictions
      l_loss_total += l_loss.item()

      # dim 1 holds the probabilities of the classes
      l_n_correct += (l_prediction.argmax(1) == l_y).type(torch.float).sum().item()
      l_n_total += l_y.numel()

  return l_loss_total, l_n_correct, l_n_total
