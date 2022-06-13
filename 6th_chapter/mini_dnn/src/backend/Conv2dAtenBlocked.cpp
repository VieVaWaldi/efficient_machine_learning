#include "Conv2dAtenBlocked.h"

at::Tensor mini_dnn::backend::Conv2dAtenBlocked::forward( at::Tensor i_x,
                                                          at::Tensor i_w ) {
  // get involved sizes
  Conv2d::Sizes l_sizes = Conv2d::getSizes( i_x,
                                            i_w );

  // prepare data for blocked Aten calls
  at::Tensor l_output = at::zeros( {l_sizes.n, l_sizes.kb, l_sizes.p, l_sizes.q, l_sizes.bk} );
  
  int64_t l_w = l_sizes.w;

  for( int64_t l_n = 0; l_n < l_sizes.n; l_n++ ) {
    for( int64_t l_kb = 0; l_kb < l_sizes.kb; l_kb++ ) {
      for( int64_t l_p = 0; l_p < l_sizes.p; l_p++ ) {
        for( int64_t l_cb = 0; l_cb < l_sizes.cb; l_cb++ ) {
          for( int64_t l_r = 0; l_r < l_sizes.r; l_r++ ) {
            for( int64_t l_s = 0; l_s < l_sizes.s; l_s++ ) {

              int64_t l_h = l_p + l_r;
              int64_t l_q = l_w - l_s;
              
              l_output[l_n][l_kb][l_p][l_q] += at::matmul( i_w[l_kb][l_cb][l_r][l_s],
                                                           i_x[l_n][l_cb][l_h][l_w] );
            }
          }
        }
      }
    }
  }
  return l_output;
}