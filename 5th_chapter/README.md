# 5th chapter

Hello, in here are the tasks for Chapter 5.

**/mini_dnn_linear:** Includes the solution for MatmulReluLibxsmm.

**/benchmarks_5_3:** Attached are the images to the benchmarks.

Ookami workflow
===============
module load pytorch/arm22/1.10

git clone https://github.com/libxsmm/libxsmm.git
cd libxsmm
make BLAS=0 -j
cd ..

mkdir catch2
wget https://github.com/catchorg/Catch2/releases/download/v2.13.9/catch.hpp -O ./catch2/catch.hpp

CXX=armclang++ make all
LIBXSMM_TARGET=a64fx ./build/test
LIBXSMM_TARGET=a64fx OMP_PLACES={0}:48:1 OMP_NUM_THREADS=48 ./build/performance_matmul
