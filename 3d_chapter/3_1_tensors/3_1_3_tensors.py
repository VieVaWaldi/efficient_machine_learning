import torch
import ctypes

"""
    Task 1: Create tensor T and print size, stride and other attributes.
"""
print("### Task 1 ###\n")


def show_tensor(tensor):
    print(tensor)
    print(
        f"With shape {tensor.size()}, stride {tensor.stride()}, dtype {tensor.dtype}, ",
        f"layout {tensor.layout} and device {tensor.device}\n")


t0 = [[0, 1, 2],
      [3, 4, 5]]
t1 = [[6, 7, 8],
      [9, 10, 11]]
t2 = [[12, 13, 14],
      [15, 16, 17]]
t3 = [[18, 19, 20],
      [21, 22, 23]]

T = [t0, t1, t2, t3]
T_tensor = torch.tensor(T)

show_tensor(T_tensor)

"""
    Task 2: New tensor and change dtype.
"""
print("### Task 2 ###\n")

l_tensor_float = torch.tensor(T, dtype=torch.float32)
print(l_tensor_float)
print(f"with dtype {l_tensor_float.dtype}\n")

"""
    Task 3: What attributes change if we fix the 2nd dim of l_tensor_float.
    - shape from [4, 2, 3] to [4, 3], excluding the 2nd dim
    - stride from (6, 3, 1) to (6, 1), again excluding 2nd dim
    - rest stayed the same
"""
print("### Task 3 ###\n")

# Dimensionen von rechts nach links, die äußerste zur innersten.
l_tensor_fixed = l_tensor_float[:, 0, :]

show_tensor(l_tensor_float)
show_tensor(l_tensor_fixed)

"""
    Task 4: Even more complex view.
    -> An sich machen wir das gleiche, aber machen in der ersten Dimension 2er Spruenge,
    wodurch die Haelft fehlt. In der 2ten Dimension aendern wir den Index um 1 nach oben.
    - shape from [4, 2, 3] to [2, 3], halbiert sich durchs springen in 1
    - stride from (6, 3, 1) to (12, 1), stride verdoppelt sich auf der ersten dim
    - rest stayed the same
"""
print("### Task 4 ###\n")

l_tensor_comp_view = l_tensor_float[::2, 1, :]

show_tensor(l_tensor_float)
show_tensor(l_tensor_comp_view)

"""
    Task 5: Contigous function.
    (As from the internet)
    - A view doesnt not change the position of elements in memory.
    - Applying the contigous function changes the data in memory so that the map 
    from elements in the new tensor to memory is canonical.
    - The new stride is therefore not excluding indices,
    here from (12, 1) to (3, 1)
"""
print("### Task 5 ###\n")

l_tensor_comp_view_contigous = l_tensor_comp_view.contiguous()

show_tensor(l_tensor_comp_view)
show_tensor(l_tensor_comp_view_contigous)

"""
    Task 6: Internal storage of a tensor by printing corresponding internal data.
    - ???
"""
print("### Task 6 ###\n")

simple_tensor = torch.tensor(t0, dtype=torch.float32)

# l_data_raw = (ctypes.c_float).from_address(simple_tensor)
