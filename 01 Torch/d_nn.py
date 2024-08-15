import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer to hidden layer
        self.fc2 = nn.Linear(4, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.fc2(x)          # Output layer (no activation function here)
        return x

# Instantiate the neural network and move it to the appropriate device
model = SimpleNN().to(device)

# Create a sample input tensor and move it to the appropriate device
input_tensor = torch.tensor([1.0, 2.0]).to(device)

# Pass the input through the model
output = model(input_tensor)

# Print the output
print("Output:", output)
