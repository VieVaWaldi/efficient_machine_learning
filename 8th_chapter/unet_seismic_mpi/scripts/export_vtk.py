#!/usr/bin/env python3
import numpy
from pyevtk.hl import gridToVTK

l_path_in = 'data'
l_path_out = 'out'

# iterate over all files
for l_base in [ 'data_test_1',
                'data_test_2',
                'data_train',
                'labels_train',
                'sample_submission_1',
                'sample_submission_2' ]:
  print( 'working on ' + l_base )

  l_key = 'data'
  if 'labels' in l_base:
    l_key = 'labels'
  elif 'sample' in l_base:
    l_key = 'prediction'

  # read data
  print( '  reading..' )
  l_data = numpy.load( l_path_in + '/' + l_base + '.npz',
                       allow_pickle=True,
                       mmap_mode='r' )
  l_data = l_data[l_key]

  # reorder dimensions
  print( '  reordering..' )
  l_data = l_data.transpose( (1, 2, 0) )
  l_data = numpy.ascontiguousarray( l_data )
  l_shape = l_data.shape

  l_x = numpy.arange( 0, l_shape[0]+1 )
  l_y = numpy.arange( 0, l_shape[1]+1 )
  l_z = numpy.arange( 0, l_shape[2]+1 )

  # write data as vtk
  print( '  writing..' )
  gridToVTK( l_path_out + '/' + l_base,
             x = l_x,
             y = l_y,
             z = l_z,
             cellData = { l_key: l_data } )
