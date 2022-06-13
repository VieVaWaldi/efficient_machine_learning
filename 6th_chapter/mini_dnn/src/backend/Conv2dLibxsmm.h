#ifndef MINI_DNN_BACKEND_CONV2D_LIBXSMM_H
#define MINI_DNN_BACKEND_CONV2D_LIBXSMM_H

#include "Conv2d.hpp"
#include <ATen/ATen.h>

namespace mini_dnn {
  namespace backend {
    class Conv2dLibxsmm;
  }
}

/**
 * Matmul backend using LIBXSMM.
 **/
class mini_dnn::backend::Conv2dLibxsmm: public Conv2d {
  private:
  public:
    /**
     * Perform the forward pass, i.e., Y = XW.
     *
     * @param i_x matrix X.
     * @param i_w matrix W.
     * @return output of the matmul, i.e., Y.
     **/
    at::Tensor forward( at::Tensor i_x,
                        at::Tensor i_w );
};

#endif