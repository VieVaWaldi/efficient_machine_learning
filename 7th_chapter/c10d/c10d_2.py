import torch
import torch.distributed

"""
    Braucht den Compute Knoten und pytorch env.

    Das Programm muss mehrmals gestartet werden, wobei
    das wrapper können $ mpiexec -n 2 python c10d.py

    Ranks sind Prozesse mit ID. Oben haben wir 2 Ranks
    gestartet. Die ID fragen wir unten ab.
"""

torch.distributed.init_process_group("mpi")

l_rank = torch.distributed.get_rank()
l_size = torch.distributed.get_world_size()
print("I am rank: " + l_rank + " of size " + l_size)

if (l_rank == 0):
    l_data = torch.tensor([[1.5, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]])

if (l_rank == 1):
    l_data = torch.tensor([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]], dtype=torch.float32)

torch.distributed.all_reduce(l_data,
                             torch.distributed.ReduceOp.SUM)

print("I am rank: " + l_rank + " with data " + l_data)
