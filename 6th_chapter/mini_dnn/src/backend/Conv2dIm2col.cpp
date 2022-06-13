#include "Conv2dIm2col.h"

at::Tensor mini_dnn::backend::Conv2dIm2col::forward(at::Tensor i_input,
                                                    at::Tensor i_weight)
{
  // get involved sizes
  Conv2d::Sizes l_sizes = Conv2d::getSizes(i_input,
                                           i_weight);

  // check that we are not having batched data
  MINI_DNN_CHECK_EQ(l_sizes.bc, 1);
  MINI_DNN_CHECK_EQ(l_sizes.bk, 1);

  // DONE: finish implementation

  // siehe conv2d.hpp -> r und s kernel sizes
  // kernel size, dilation, padding, stride als parameter definieren
  std::vector<int64_t> l_kernel_sizes = {l_sizes.r, l_sizes.s};
  std::vector<int64_t> l_dilations = {1, 1};
  std::vector<int64_t> l_paddings = {0, 0};
  std::vector<int64_t> l_strides = {1, 1};

  // im2Col aufruf
  // 45 kommt aus dem r*s*c Block von w
  // 60 aus dem c*w von x
  std::cout << "i_input: " << i_input.sizes() << std::endl;
  at::Tensor l_input = at::im2col(i_input, l_kernel_sizes, l_dilations, l_paddings, l_strides);
  std::cout << "l_input: " << l_input.sizes() << std::endl;

  // Nur ändern der Sicht auf die weights
  std::cout << "i_weight: " << i_weight.sizes() << std::endl;
  at::Tensor l_weight = at::flatten(i_weight, 1);
  std::cout << "l_weight: " << l_weight.sizes() << std::endl;

  // Und jetzt wo x und w richtig liegen können wir rechnen
  // 3 batch size, 4 channel_input, 60 p*q
  at::Tensor l_output = at::matmul(l_weight, l_input);
  std::cout << "l_output: " << l_output.sizes() << std::endl;

  // Output Sicht aendern fuer pytorch
  // Die 60 wird zu einer 6*10 umgewandelt
  l_output = l_output.view({l_sizes.n, l_sizes.kb, l_sizes.p, l_sizes.q});

  return l_output;
}
