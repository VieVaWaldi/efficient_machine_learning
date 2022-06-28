import numpy as np
import torch.utils.data

## @package eml.SeismicDataSet
#  Seismic data set which reads raw data and processes it.
class SeismicDataSet( torch.utils.data.Dataset ):
  ## Initializes the data set.
  #  @param i_file_data file containing the stacked seismic data.
  #  @param i_file_labels file containing the labels.
  #  @param i_item_shape shape of a single item.
  #  @param i_subset optional subset of the data set.
  def __init__( self,
                i_file_data,
                i_file_labels,
                i_item_shape,
                l_rank,
                i_subset = None ):
    self.m_data = np.load( i_file_data,
                           mmap_mode='r',
                           allow_pickle = True )['data']

    self.m_labels = np.load( i_file_labels,
                             mmap_mode='r',
                             allow_pickle = True )['labels']

    if( i_subset != None ):
      self.m_data = self.m_data[ i_subset[0][0]:i_subset[0][1],
                                 i_subset[1][0]:i_subset[1][1],
                                 i_subset[2][0]:i_subset[2][1] ]
      self.m_labels = self.m_labels[ i_subset[0][0]:i_subset[0][1],
                                     i_subset[1][0]:i_subset[1][1],
                                     i_subset[2][0]:i_subset[2][1] ]

    self.m_shape_padded = self.m_data.shape
    self.m_shape = np.subtract( self.m_shape_padded, i_item_shape )
    self.m_shape = tuple( np.add( self.m_shape, 1 ) )
    self.m_item_shape = i_item_shape
    self.l_rank = l_rank

    # TODO: delete
    self.print_rank( 'shape padded:'+ str(self.m_shape_padded) )
    self.print_rank( 'shape:'+ str(self.m_shape))

  def print_rank(self,input):
    if self.l_rank == 0:
         print(input)

  ## Gets the mean of the internal data.
  #  @return mean.
  def getMean( self ):
    return self.m_data.mean()

  ## Gets the standard deviation of the internal data.
  #  @return standard deviation.
  def getStdDev( self ):
    return self.m_data.std()

  ## Normalizes the internal data by the given mean and standard deviation.
  #  This means: internal = (internal - mean) / stddev
  #  @param i_mean used mean.
  #  @param i_std_dev used standard deviation.
  def normalize( self,
                 i_mean,
                 i_std_dev ):
    self.m_data = self.m_data - i_mean
    self.m_data = self.m_data / i_std_dev

  ## Returns the length of the data set, i.e., the number of inner points.
  #  @param self object pointer.
  def __len__( self ):
    return np.prod( self.m_shape )

  ## Gets a single "item" based on a one-dimensional id.
  #  The method infers the respective 3d id and returns the patch (see shape) starting a this position.
  #
  #  @param self object pointer.
  #  @param i_idx one-dimensional id.
  def __getitem__( self,
                   i_idx ):
    # derive ids in every dimension.
    l_id0 = i_idx // ( self.m_shape[1] * self.m_shape[2] )
    l_id1 = i_idx % ( self.m_shape[1] * self.m_shape[2] )
    l_id1 = l_id1 // self.m_shape[2]
    l_id2 = i_idx % self.m_shape[2]

    # get chunks
    l_data  = self.m_data[ l_id0:l_id0 + self.m_item_shape[0],
                           l_id1:l_id1 + self.m_item_shape[1],
                           l_id2:l_id2 + self.m_item_shape[2] ]

    l_labels = self.m_labels[ l_id0:l_id0 + self.m_item_shape[0],
                              l_id1:l_id1 + self.m_item_shape[1],
                              l_id2:l_id2 + self.m_item_shape[2] ]

    # squeeze dimensions of size 1
    l_data = l_data.squeeze()
    l_labels = l_labels.squeeze()

    # add a feature dimension
    l_data = np.expand_dims( l_data,
                             axis = 0 )
    l_labels = np.expand_dims( l_labels,
                               axis = 0 )

    # convert to fp32 torch tensor
    l_data = torch.from_numpy( l_data )
    l_data = l_data.type( torch.float32 )
    l_labels = torch.from_numpy( l_labels )
    l_labels = l_labels.type( torch.long )

    return l_data, l_labels