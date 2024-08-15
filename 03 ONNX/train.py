# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.onnx

# Define a simpler neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to sample points within an ellipse
def sample_points_within_ellipse(num_samples, a, b):
    points = []
    while len(points) < num_samples:
        x, y = np.random.uniform(-a, a), np.random.uniform(-b, b)
        if (x**2 / a**2 + y**2 / b**2) <= 1:
            points.append([x, y])
    return np.array(points)

# Generate data points
a, b = 3.0, 2.0  # Semi-major and semi-minor axes
num_samples = 1000
points = sample_points_within_ellipse(num_samples, a, b)

# Compute z values (distance from center, with sign based on quadrant)
z_values = points[:, 1]  # For simplicity, use y coordinate

# Convert dataset to PyTorch tensors
points_tensor = torch.tensor(points, dtype=torch.float32)
z_tensor = torch.tensor(z_values, dtype=torch.float32).view(-1, 1)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# Get the current device
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# Initialize the simple model
model = SimpleNet().to(device)  # Move model to GPU if available
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Move data to GPU if available
points_tensor = points_tensor.to(device)
z_tensor = z_tensor.to(device)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()
    outputs = model(points_tensor)
    loss = criterion(outputs, z_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:  # Print every 100 epochs
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

print("Training complete.")

# Save the trained model to ONNX format
onnx_file_path = "simplenet_model.onnx"
dummy_input = torch.randn(1, 2, device=device)  # Dummy input for exporting the model
torch.onnx.export(model, dummy_input, onnx_file_path, input_names=['input'], output_names=['output'], opset_version=11)

print(f"Model saved to {onnx_file_path}")
