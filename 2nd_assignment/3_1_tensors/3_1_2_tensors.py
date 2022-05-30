import torch

"""
    Task 1: Elemtwise operation on P and Q with add and mul, also overload.
"""
print("### Task 1 ###\n")

p = [[0, 1, 2],
     [3, 4, 5]]
q = [[6, 7, 8],
     [9, 10, 11]]

p_tensor = torch.Tensor(p)
q_tensor = torch.Tensor(q)

print(torch.add(p_tensor, q_tensor))
print(p_tensor+q_tensor)

print(torch.mul(p_tensor, q_tensor))
print(p_tensor*q_tensor)
print("")

"""
    Task 2: Matrix matrix product of P and Qt, also overload.
"""
print("### Task 2 ###\n")

q_tensor_trans = torch.transpose(q_tensor, 0, 1)

print(q_tensor_trans)
print(torch.matmul(p_tensor, q_tensor_trans))
print(p_tensor @ q_tensor_trans)
print("")

"""
    Task 3: Reduction operations.
"""
print("### Task 3 ###\n")

print(f"min of q = {q_tensor.min()}")
print(f"max of q = {q_tensor.max()}")
print(f"sum of q = {q_tensor.sum()}\n")

"""
    Task 4: explain the difference.
    - In both cases the output is a tensor with the same shape, but all entries set to 0.
    - However in the 2nd case, the original tensor is untouched and a new one created.
    - In the first case the operation to set all entries to 0 affects the original tensor,
    - as it should. The 2nd tensor got detached.
"""
print("### Task 4 ###\n")

l_tensor_0 = torch.tensor(q)
l_tensor_1 = torch.tensor(q)

l_tmp = l_tensor_0
l_tmp[:] = 0
print(l_tensor_0)
print(l_tmp)

l_tmp = l_tensor_1.clone().detach()
l_tmp[:] = 0
print(l_tensor_1)
print(l_tmp)
