import torch

# Define two 2D tensors (matrices)
A = torch.tensor([[1, 2], 
                  [3, 4]])
B = torch.tensor([[5, 6], 
                  [7, 8]])

# Matrix multiplication
C = A @ B

print(C)