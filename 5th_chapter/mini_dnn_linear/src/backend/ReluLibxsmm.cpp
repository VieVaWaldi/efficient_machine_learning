#include "ReluLibxsmm.h"
#include <libxsmm.h>

at::Tensor mini_dnn::backend::ReluLibxsmm::forward( at::Tensor i_input)
{
	// get 3d-tensor view to potentially higher dimensional tensor
	at::Tensor l_input_3d = i_input.squeeze().flatten(0,-3);
	
	c10::IntArrayRef l_sizes_input_3d = l_input_3d.size();
	c10::IntArrayRef l_strides_input_3d = l_input_3d.strides();
	
	//make surr that we have enough dimensions and strides-1 access w.r.t.
	MINI_DNN_CHECK_GE (l_strides_input_3d.size(),3)
	MINI_DNN_CHECK_EQ (l_strides_input_3d[2] ,1)
	
	//generate kernels
	libxsmm_meltw_unary_shape l_unary_shape = libxsmm_create_meltw_unary_shape( l_sizes_input_3d[2],
																				l_sizes_input_3d[1],
																				l_strides_input_3d[1],
																				l_strides_input_3d[1],
																				LIBXSMM_DATATYPE_F32,
																				LIBXSMM_DATATYPE_F32,
																				LIBXSMM_DATATYPE_F32 );
	
	libxsmm_meltwfunction_unary l_relu = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU,
																		  l_unary_shape,
																		  LIBXSMM_MELTW_FLAG_NONE );
	
	//prepare data  for LIBXSMM calls
	at::Tensor l_output = at::empty( i_input.size());
	
	#pragma omp parallel for
	for( int64_t l_i = 0; l_i < l_sizes_input_3d[0]; l_i++ )
	{
		libxsmm_meltw_unary_param l_param;
		l_param.in.primary = (float *) i_input.data_ptr() + l_strides_input_3d[0] * l_i;
		l_param.out.primary = (float *) l_output.data_ptr() l_strides_input_3d[0] * l_i;
		
		//call kernel
		l_relu( &l_param);
	}
	
	return l_output;
}
																	