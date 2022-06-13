#include <catch2/catch.hpp>
#include "Conv2dAtenBlocked.h"

TEST_CASE("Tests the convolution operator going through blocked at::matmul.",
          "[conv2d][im2col][forward]")
{
    int64_t l_size_n = 3;  // n = batch_size
    int64_t l_size_h = 8;  // h = x_height
    int64_t l_size_w = 12; // w = x_width
    int64_t l_size_c = 5;  // c = x_channels && w_channels

    int64_t l_size_k = 4; // k = y_channels
    int64_t l_size_r = 3; // r = w_height
    int64_t l_size_s = 3; // s = w_width

    int64_t l_size_bc = 1; // bc = block input_channels
    int64_t l_size_bk = 1; // bk = block output_channels

    int64_t l_size_cb = l_size_c / l_size_bc;
    int64_t l_size_kb = l_size_k / l_size_bk;

    // construct input and weight tensors
    at::Tensor l_x = at::rand({l_size_n, l_size_c, l_size_h, l_size_w});
    at::Tensor l_w = at::rand({l_size_k, l_size_c, l_size_r, l_size_s});

    // Construct blocked tensors
    // Xb: n,  cb, h, w, bc
    // Wb: kb, cb, r, s, bc, bk
    // Yw: n,  kb, p, q, bk

    // X_blocked ???
    at::Tensor l_x_blocked = l_x.view({l_size_n, l_size_cb, l_size_h, l_size_w, l_size_bc});
    l_x_blocked = l_x_blocked.permute({0, 1, 2, 3, 4}).contiguous();

    // W_blocked ???
    at::Tensor l_w_blocked = l_w.view({l_size_kb, l_size_cb, l_size_r, l_size_s, l_size_bc, l_size_bk});
    l_w_blocked = l_w_blocked.permute({0, 1, 2, 3, 4, 5}).contiguous();

    // compute solution
    mini_dnn::backend::Conv2dIm2col l_conv2d;
    at::Tensor l_y_blocked = l_conv2d.forward(l_x_blocked,
                                              l_w_blocked);

    // compute reference
    at::Tensor l_y = at::conv2d(l_x,
                                l_w);

    // check solution
    REQUIRE(at::allclose(l_y_blocked, l_y));
}