#include <iostream>
#include <catch2/catch.hpp>
#include "Conv2dAtenBlocked.h"

TEST_CASE("Tests the convolution operator going through blocked at::matmul.",
          "[conv2d][conv2dAtenBlocked][forward]")
{
    std::cout<< "DEBUG TEST CONV2DATENBLOCKED" << std::endl;

    // M: N (batch size)
    // K: C (in features)
    // N: K (out features)

    // input sizes
    int64_t l_size_n = 3;  // n = batch_size
    int64_t l_size_h = 8;  // h = x_height
    int64_t l_size_w = 12; // w = x_width
    int64_t l_size_c = 6;  // c = x_channels && w_channels

    int64_t l_size_k = 4; // k = y_channels
    int64_t l_size_r = 3; // r = w_height
    int64_t l_size_s = 3; // s = w_width

    int64_t l_size_bc = 1; // bc = block input_channels
    int64_t l_size_bk = 1; // bk = block output_channels

    int64_t l_size_cb = l_size_c / l_size_bc;
    int64_t l_size_kb = l_size_k / l_size_bk;

    int64_t l_size_p = 6; // p = y_height
    int64_t l_size_q = 10; // q = y_width

    // construct input and weight tensors
    at::Tensor l_x = at::rand({l_size_n, l_size_c, l_size_h, l_size_w});
    at::Tensor l_w = at::rand({l_size_k, l_size_c, l_size_r, l_size_s});

    // Construct blocked tensors
    // Xb: n,  cb, h, w, bc
    // Wb: kb, cb, r, s, bc, bk
    // Yb: n,  kb, p, q, bk

    // X_blocked                        0           1           2           3       4                 
    at::Tensor l_x_blocked = l_x.view({l_size_n, l_size_cb, l_size_bc, l_size_h, l_size_w });
    l_x_blocked = l_x_blocked.permute({0, 1, 3, 4, 2}).contiguous();

    // W_blocked                        0           1           2           3       4           5
    at::Tensor l_w_blocked = l_w.view({l_size_kb, l_size_bk, l_size_cb, l_size_bc, l_size_r, l_size_s});
    l_w_blocked = l_w_blocked.permute({0, 2, 4, 5, 3, 1}).contiguous();

    std::cout<< "DEBUG AUFRUF FORWARD CONV2DATENBLOCKED" << std::endl;

    // compute solution
    mini_dnn::backend::Conv2dAtenBlocked l_conv2dAtenBlocked;
    at::Tensor l_y_blocked = l_conv2dAtenBlocked.forward(l_x_blocked,
                                                         l_w_blocked);

    // reverse blocking
    at::Tensor l_y = l_y_blocked.permute({0, 1, 4, 2, 3}).contiguous();
    l_y = l_y.view({l_size_n, l_size_k, l_size_p, l_size_q});

    // compute reference
    at::Tensor l_y_reference = at::conv2d(l_x,
                                          l_w);

    std::cout << "l_y " << l_y.sizes() << std::endl;
    std::cout << l_y << std::endl;

    std::cout << "l_y_reference " << l_y_reference.sizes() << std::endl;
    std::cout << l_y_reference << std::endl;

    // check solution
    REQUIRE(at::allclose(l_y, l_y_reference, 1E-5, 1E-6));
}