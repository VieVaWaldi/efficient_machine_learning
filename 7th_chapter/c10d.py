import os
import torch
import torch.distributed

"""
    Braucht den Compute Knoten und pytorch env.

    Das Programm muss mehrmals gestartet werden, wobei
    das wrapper können $ mpiexec -n 2 python c10d.py

    Ranks sind Prozesse mit ID. Oben haben wir 2 Ranks
    gestartet. Die ID fragen wir unten ab.
"""

# 1. ProzessGruppe initialisieren mit mpi backend
torch.distributed.init_process_group("mpi")

l_rank = torch.distributed.get_rank()
print("I am rank: " + l_rank)

l_size = torch.distributed.get_world_size()
print("l_size: " + l_size)

# 2.
