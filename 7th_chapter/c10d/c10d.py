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

# 2. Get Daten

if (l_rank == 0):
    l_data = torch.ones(3, 4)  # [3, 4]
else:
    l_data = torch.zeros(3, 4)  # [3, 4]

print("Before send, I am rank: " + l_rank + " with data " + l_data)

# 3. Daten verschicken

# rank 0 sendet an rang 1, 1 received von 0
# returned nicht bevor kommunikatino abgeschlossen ist -> Blockiert
if (l_rank == 0):
    torch.distributed.send(tensor=l_data, dst=1)
else:
    torch.distributed.recv(tensor=l_data, dst=0)

# 4. Ausgeben der daten

if (l_rank == 1):
    print("After receive, I am rank: " + l_rank + " with data " + l_data)

# 5. Unblockierte kommunikation -> Async

if (l_rank == 0):
    l_req = torch.distributed.isend(tensor=l_data, dst=1)
else:
    l_req = torch.distributed.irecv(tensor=l_data, dst=0)

# Doch warten bis request fertig ist
if (l_rank < 2):
    l_req.wait()

if (l_rank == 1):
    print("After isend/irecv " + l_data)
