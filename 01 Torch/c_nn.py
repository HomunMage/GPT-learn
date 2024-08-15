import torch
import torch.nn as nn

# Define the neural network class
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define the layers of the network
        self.fc1 = nn.Linear(10, 5)  # Input layer to hidden layer
        self.fc2 = nn.Linear(5, 1)   # Hidden layer to output layer

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.fc2(x)              # Output layer
        return x

# Create an instance of the network
net = SimpleNet()

# Create some random input data
input_data = torch.randn(1, 10)  # Batch size of 1, 10 features

# Perform a forward pass through the network
output = net(input_data)

# Print the output
print("Output:", output)
