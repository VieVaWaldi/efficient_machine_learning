#include <catch2/catch.hpp>
#include "Conv2dLibxsmm.h"

TEST_CASE("Tests the Conv2dLibxsmm",
          "[conv2d][conv2dAtenBlocked][forward]")
{
    // input sizes
    int64_t l_size_n = 3;  // n = batch_size
    int64_t l_size_h = 8;  // h = x_height
    int64_t l_size_w = 12; // w = x_width
    int64_t l_size_c = 5;  // c = x_channels && w_channels

    int64_t l_size_k = 4; // k = y_channels
    int64_t l_size_r = 3; // r = w_height
    int64_t l_size_s = 3; // s = w_width

    int64_t l_size_p = 6; // p = y_height
    int64_t l_size_q = 10; // q = y_width

    int64_t l_size_bc = 1; // bc = block input_channels
    int64_t l_size_bk = 2; // bk = block output_channels

    int64_t l_size_cb = l_size_c / l_size_bc;
    int64_t l_size_kb = l_size_k / l_size_bk;

    // construct input and weight tensors
    at::Tensor l_input = at::rand({l_size_n, l_size_c, l_size_h, l_size_w});
    at::Tensor l_weight = at::rand({l_size_k, l_size_c, l_size_r, l_size_s});
    l_input -= 0.5f;
    l_weight -= 0.5f;

    // Construct blocked tensors
    // Xb: n,  cb, h, w, bc
    // Wb: kb, cb, r, s, bc, bk
    // Yb: n,  kb, p, q, bk

    // X_blocked                                0           1           2           3       4                 
    at::Tensor l_input_blocked = l_input.view({l_size_n, l_size_cb, l_size_bc, l_size_h, l_size_w });
    l_input_blocked = l_input_blocked.permute({0, 1, 3, 4, 2}).contiguous();

    // W_blocked                                    0           1           2           3       4           5
    at::Tensor l_weight_blocked = l_weight.view({l_size_kb, l_size_bk, l_size_cb, l_size_bc, l_size_r, l_size_s});
    l_weight_blocked = l_weight_blocked.permute({0, 2, 4, 5, 3, 1}).contiguous();

    // compute solution
    mini_dnn::backend::Conv2dReluLibxsmm l_kernel;
    at::Tensor l_output_blocked = l_kernel.forward(l_input_blocked,
                                                   l_weight_blocked);

    // reverse blocking
    at::Tensor l_output = l_output_blocked.permute( {0, 1, 4, 2, 3} ).contiguous();
    l_output = l_output.view( { l_size_n, l_size_k, l_size_p, l_size_q } );

    // compute reference
    at::Tensor l_output_reference = at::conv2d(l_input,
                                               l_weight);

    // check solution
    REQUIRE(at::allclose(l_output, 
                         l_output_reference,
                         1E-5,
                         1E-6));
}