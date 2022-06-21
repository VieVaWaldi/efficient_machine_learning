OMP_NUM_THREADS=12 mpiexec -n $1 --bind-to socket python mlp_fashion_mnist.py
