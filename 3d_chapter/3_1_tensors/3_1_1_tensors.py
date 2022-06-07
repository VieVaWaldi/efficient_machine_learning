import torch
import numpy as np

"""
    Task 1: Try different tensor generating functions.
"""
print("### Task 1 ###\n")


def show_tensor(tensor):
    print(tensor)
    print(
        f"with shape {tensor.shape}, stride {tensor.stride()}, min {tensor.min()}, and max {tensor.max()}\n")


a_tensor = torch.zeros([2, 2])
b_tensor = torch.ones([2, 2, 2])
c_tensor = torch.ones_like(b_tensor)
d_tensor = torch.rand([2, 2, 2, 2])

show_tensor(a_tensor)
show_tensor(b_tensor)
show_tensor(c_tensor)
show_tensor(d_tensor)

"""
    Task 2: Use a list of lists to alloc tensor with 4,2,3, init it to 3.1.1
"""
print("### Task 2 ###\n")

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
    Task 3: Turn T into np.array, then into a tensor
"""
print("### Task 3 ###\n")

T_numpy = np.array(T)
T_tensor = torch.tensor(T_numpy)

print(T_numpy)
show_tensor(T_tensor)
