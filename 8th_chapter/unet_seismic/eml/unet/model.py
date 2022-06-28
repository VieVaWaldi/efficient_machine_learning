import torch.nn

## @package eml.unet.model.Unet2d
#  Unet in two dimensions.
class Unet2d( torch.nn.Module ):
  ## Generates a single block in the U-Net, i.e., a set of convolutional layers followed by ReLU respectively.
  #
  # @param i_in_channels number of input channels.
  # @param i_out_channels number of output channels.
  # @param i_kernel_size kernel size used in the convolutions.
  # @param i_n_layers number of layers in the blocks, e.g., 3 means 3x(conv + relu).
  @staticmethod
  def generateBlock( i_channels_in,
                     i_channels_out,
                     i_kernel_size,
                     i_n_layers = 3 ):
    l_block = [ torch.nn.Conv2d( in_channels  = i_channels_in,
                                 out_channels = i_channels_out,
                                 kernel_size  = i_kernel_size ),
                torch.nn.BatchNorm2d( i_channels_out ),
                torch.nn.ReLU( inplace = True ) ]
    for _ in range(i_n_layers-1):
      l_block.append( torch.nn.Conv2d( in_channels  = i_channels_out,
                                       out_channels = i_channels_out,
                                       kernel_size  = i_kernel_size ) )
      l_block.append( torch.nn.BatchNorm2d( i_channels_out ) ),
      l_block.append( torch.nn.ReLU( inplace = True ) )

    return torch.nn.Sequential( *l_block )

  ## Derives the required padding for the net.
  #  @param self object pointer.
  #  @return padding.
  def getPadding( self ):
    l_tmp = torch.zeros( 1, 1, 1004, 1004 )
    l_tmp = self.forward( l_tmp )
    l_tmp = 1004 - l_tmp.shape[-1]
    l_tmp = l_tmp // 2

    return l_tmp

  ## Initializes the class.
  #  @param self object pointer.
  #  @param i_n_init_channels number of initial channels (used in first block).
  #  @param i_kernel_size kernels size of the convolution.
  #  @param i_n_layers_per_block number of layers per block.
  #  @param i_n_levels number of levels.
  def __init__( self,
                i_n_init_channels    = 32,
                i_kernel_size        =  3,
                i_n_layers_per_block =  3,
                i_n_levels           =  2 ):
    super( Unet2d, self ).__init__()

    self.m_n_levels = i_n_levels
    self.m_encoder = []
    self.m_crop = []
    self.m_decoder = []

    # iterate over levels of encoder
    for l_le in range( i_n_levels-1 ):
      self.m_encoder.append( self.generateBlock( 1 if (l_le == 0) else i_n_init_channels*pow(2,l_le-1),
                                                 i_n_init_channels*pow(2,l_le),
                                                 i_kernel_size,
                                                 i_n_layers_per_block ) )
    
    # max pooling
    self.m_max_pooling = torch.nn.MaxPool2d( 2 )

    # cropping
    # we start at the bottleneck layer
    l_n_crop = (i_kernel_size - 1) * i_n_layers_per_block
    self.m_crop.append( torch.nn.ZeroPad2d( -l_n_crop ) )
    # in every step:
    #   1) multiply old crop with 2x due to pooling
    #   2) add contribution of convolutions in encoder and decoder
    for l_le in range( 1, i_n_levels-1 ):
      l_n_crop = l_n_crop*2
      l_n_crop = l_n_crop + (i_kernel_size - 1) * i_n_layers_per_block * 2
      self.m_crop.append( torch.nn.ZeroPad2d( -l_n_crop ) )
    self.m_crop.reverse()

    # bottleneck is last level
    self.m_bottleneck = self.generateBlock( i_n_init_channels*pow(2,i_n_levels-2),
                                            i_n_init_channels*pow(2,i_n_levels-2),
                                            i_kernel_size,
                                            i_n_layers_per_block )

    # up sampling
    self.m_up_sampling = torch.nn.Upsample( scale_factor = 2,
                                            mode = 'bilinear',
                                            align_corners = True )

    # iterate over levels of decoder
    for l_le in range( i_n_levels-1 ):
      self.m_decoder.append( self.generateBlock( i_n_init_channels*pow(2,l_le+1),
                                                 i_n_init_channels*pow(2,max(l_le-1, 0)),
                                                 i_kernel_size,
                                                 i_n_layers_per_block ) )

    # final output in eight classes
    self.m_classification = torch.nn.Conv2d( i_n_init_channels, 8, 1 )

    self.eval()
    self.m_padding = self.getPadding()

    # transfer layers to CUDA device if available
    if( torch.cuda.is_available() ):
      for l_en in range( len( self.m_encoder ) ):
        self.m_encoder[l_en] = self.m_encoder[l_en].to( torch.device('cuda') )
      for l_de in range( len( self.m_decoder ) ):
        self.m_decoder[l_de] = self.m_decoder[l_de].to( torch.device('cuda') )
      for l_en in range( len( self.m_crop ) ):
        self.m_crop[l_en] = self.m_crop[l_en].to( torch.device('cuda') )
      self.m_bottleneck = self.m_bottleneck.to( torch.device('cuda') )
      self.m_up_sampling = self.m_up_sampling.to( torch.device('cuda') )
      self.m_max_pooling = self.m_max_pooling.to( torch.device('cuda') )
      self.m_classification = self.m_classification.to( torch.device('cuda') )

  ## String representation of the Unet2d class.
  #  @param self object pointer.
  def __str__(self):
    l_str = "encoder:"
    for l_en in self.m_encoder:
      l_str = l_str + "\n" + str(l_en)
    l_str = l_str + "\nmax_pooling:\n"
    l_str = l_str + str(self.m_max_pooling)
    l_str = l_str + "\ncrop:"
    for l_en in self.m_crop:
      l_str = l_str + "\n" + str(l_en)
    l_str = l_str + "\nbottleneck:\n"
    l_str = l_str + str(self.m_bottleneck)
    l_str = l_str + "\nup_sampling:\n"
    l_str = l_str + str(self.m_up_sampling)
    l_str = l_str + "\ndecoder:"
    for l_de in self.m_decoder:
      l_str = l_str + "\n" + str(l_de)
    l_str = l_str + "\nclassification:\n"
    l_str = l_str + str(self.m_classification)

    return l_str

  ## Forward pass with the given input.
  #  @param self object pointer.
  #  @param i_input input for the forward pass.
  #  @return output of the U-Net.
  def forward( self,
               i_input ):
    l_tmp_encoder = []

    l_tmp = i_input

    # encoder
    for l_le in range(0, self.m_n_levels-1):
      # eval convolutional block
      l_tmp = self.m_encoder[l_le]( l_tmp )
      # save cropped result for skip connection
      l_tmp_encoder.append( self.m_crop[l_le]( l_tmp ) )
      # apply max pooling
      l_tmp = self.m_max_pooling( l_tmp )

    # bottleneck
    l_tmp = self.m_bottleneck( l_tmp )

    # decoder
    for l_le in range(self.m_n_levels-2, -1, -1):
      l_tmp = self.m_up_sampling( l_tmp )
      l_tmp = torch.cat( (l_tmp, l_tmp_encoder[l_le]),
                         dim = 1 )
      l_tmp = self.m_decoder[l_le]( l_tmp )

    # classification
    l_tmp = self.m_classification( l_tmp )

    return l_tmp
