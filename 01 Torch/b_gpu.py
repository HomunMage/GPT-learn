import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# Get the current device
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# Display GPU information if available
if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Perform a simple computation on GPU
x = torch.rand(3, 3).to(device)
y = torch.rand(3, 3).to(device)
z = x + y
w = x @ y

print("Computation result (x + y):")
print(z)
print(f"Result is on device: {z.device}")
