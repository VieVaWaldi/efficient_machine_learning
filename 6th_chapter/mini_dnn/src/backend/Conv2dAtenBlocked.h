#ifndef MINI_DNN_BACKEND_CONV2D_ATEN_BLOCKED_H
#define MINI_DNN_BACKEND_CONV2D_ATEN_BLOCKED_H

#include <ATen/ATen.h>
#include "Conv2d.hpp"

namespace mini_dnn {
  namespace backend {
    class Conv2dAtenBlocked;
  }
}

/**
 * Conv2d Blocked backend using the Aten library.
 **/
class mini_dnn::backend::Conv2dAtenBlocked: public Conv2d {
  private:
  public:
    /**
     * Perform the forward pass.
     *
     * @param i_input input.
     * @param i_weight weights.
     * @return output of the convolution.
     **/
    at::Tensor forward( at::Tensor i_input,
                        at::Tensor i_weight );
};

#endif