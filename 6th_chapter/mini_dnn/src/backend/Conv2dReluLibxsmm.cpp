#include "Conv2dReluLibxsmm.h"
#include <libxsmm.h>

at::Tensor mini_dnn::backend::Conv2dReluLibxsmm::forward( at::Tensor i_x,
                                                          at::Tensor i_w ) {
  // get involved sizes
  Conv2d::Sizes l_sizes = Conv2d::getSizes( i_x,
                                            i_w );
  // create LIBXSMM kernel (isolated)
  libxsmm_gemm_shape l_shape_brgemm;
  libxsmm_bitfield l_flags_brgemm = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags_brgemm = 0;
    
  // libxsmm_blasint l_n = l_sizes.bk;
  // libxsmm_blasint l_k = l_sizes.bc;

  // libxsmm_blasint l_lda = l_n;
  // libxsmm_blasint l_ldb = l_k;

  libxsmm_blasint l_m = 1;
  libxsmm_blasint l_n = l_sizes.bk;
  libxsmm_blasint l_k = l_sizes.bc;

  libxsmm_blasint l_lda = l_m;
  libxsmm_blasint l_ldb = l_k;
  libxsmm_blasint l_ldc = l_m;
  
//   l_shape_brgemm = libxsmm_create_gemm_shape( l_m,
//                                               l_n,
//                                               l_k,
//                                               l_lda,
//                                               l_ldb,
//                                               l_ldc,
//                                               LIBXSMM_DATATYPE_F32,
//                                               LIBXSMM_DATATYPE_F32,
//                                               LIBXSMM_DATATYPE_F32,
//                                               LIBXSMM_DATATYPE_F32 );
//   //generate kernels
//   libxsmm_meltw_unary_shape l_unary_shape = libxsmm_create_meltw_unary_shape( l_n,
// 	 																		  l_k,
// 																		    l_lda,
// 																			  l_ldc,
// 																			  LIBXSMM_DATATYPE_F32,
// 																			  LIBXSMM_DATATYPE_F32,
// 																			  LIBXSMM_DATATYPE_F32 );
																				  
//   libxsmm_meltwfunction_unary l_relu = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU,
// 																		l_unary_shape,
// 																		LIBXSMM_GEMM_FLAG_NONE );
//   libxsmm_gemm_batch_reduce_config l_brconfig;
//   l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
//   l_brconfig.br_stride_a_hint = 0;
//   l_brconfig.br_stride_b_hint = 0;
//   l_brconfig.br_unroll_hint = 0;

//   libxsmm_xmmfunction l_kernel_forward;
//   l_kernel_forward.gemm = libxsmm_dispatch_brgemm_v2( l_shape_brgemm,
//                                                       l_flags_brgemm,
//                                                       l_prefetch_flags_brgemm,
//                                                       l_brconfig );
//   libxsmm_gemm_param l_param_relu;
//   libxsmm_gemm_param l_param;
//   memset( &l_param_relu,0,sizeof(libxsmm_gemm_param));
//   memset( &l_param,
//           0,
//           sizeof(libxsmm_gemm_param) );

//   // prepare data for blocked Aten calls
  at::Tensor l_y = at::zeros( {l_sizes.n,l_sizes.kb, l_sizes.p, l_sizes.cb, l_sizes.r,l_sizes.s} );
  
//   c10::IntArrayRef l_strides_a = i_x.strides();
//   c10::IntArrayRef l_strides_b = i_w.strides();
//   c10::IntArrayRef l_strides_c = l_y.strides();

//   float * l_ptr_a = (float*) i_x.data_ptr();
//   float * l_ptr_b = (float*) i_w.data_ptr();
//   float * l_ptr_c = (float*) l_y.data_ptr();

//   std::cout << "TESSTTTT LIBXSMM" << std::endl;
    
//   for( int64_t l_n = 0; l_n < l_sizes.n; l_n++ ) {
// 	for( int64_t l_kb = 0; l_kb < l_sizes.kb; l_kb++ ) {
// 		for( int64_t l_p = 0; l_p < l_sizes.p; l_p++ ) {
			
// 			int64_t l_offset_output  = l_n   * l_strides_c[0];
// 			        l_offset_output += l_kb  * l_strides_c[1];
//               l_offset_output += l_p   * l_strides_c[2];

// 			for( int64_t l_cb = 0; l_cb < l_sizes.cb; l_cb++ ) {
// 				for( int64_t l_r = 0; l_r < l_sizes.r; l_r++ ) {
// 					for( int64_t l_s = 0; l_s < l_sizes.s; l_s++ ) {
// 						int64_t l_offset_weight  = l_kb * l_strides_b[0];
// 						l_offset_weight         += l_cb * l_strides_b[1];
// 						l_offset_weight         += l_r  * l_strides_b[2];
// 						l_offset_weight         += l_s  * l_strides_b[3];
						
// 						int64_t l_h = l_p + l_r;
// 						int64_t l_w = 0   + l_s;
						
// 						int64_t l_offset_input   = l_n  * l_strides_a[0];
// 						l_offset_input  += l_cb * l_strides_a[1];
// 						l_offset_input  += l_h  * l_strides_a[2];
// 						l_offset_input  += l_w  * l_strides_a[3];
						
// 						l_param.a.primary = l_ptr_b + l_offset_weight;
// 						l_param.b.primary = l_ptr_a + l_offset_input;
// 						l_param.c.primary = l_ptr_c + l_offset_output;
						
// 						l_kernel_forward.gemm(&l_param);
//            }
//          }
//        }
	   
// 	   // apply Relu
// 	   libxsmm_meltw_unary_param = l_param_relu;
// 	   l_param_relu.in.primary   = l_ptr_c + l_offset_output;
// 	   l_param_relu.out.primary  = l_ptr_c + l_offset_output;
// 	   l_relu( &l_param_relu );
	   
//      }
//    }
//  }
  return l_y;
}