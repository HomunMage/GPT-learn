import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Load the tokenizer and set embedding dimension
tokenizer = AutoTokenizer.from_pretrained("gpt2")
embed_dim = 768  # Use the GPT-2 embedding dimension

# Define the State Space Model
class StateSpaceModel(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim):
        super(StateSpaceModel, self).__init__()
        
        # Linear layers for state transition and observation
        # A: State transition matrix
        # B: Control input matrix
        # C: Observation matrix
        # D: Feedthrough matrix
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(state_dim, input_dim))
        self.C = nn.Parameter(torch.randn(output_dim, state_dim))
        self.D = nn.Parameter(torch.randn(output_dim, input_dim))
        
        # Optionally, define noise scales for process and measurement noise
        self.process_noise_std = nn.Parameter(torch.ones(state_dim))
        self.measurement_noise_std = nn.Parameter(torch.ones(output_dim))

    def forward(self, x_t, u_t):
        """
        Apply the state space model equations:

        1. State transition equation:
            x_{t+1} = A * x_t + B * u_t + w_t
        2. Observation equation:
            y_t = C * x_{t+1} + D * u_t + v_t
        
        where:
        - x_t: Current state (batch_size, state_dim)
        - u_t: Control input (batch_size, input_dim)
        - w_t: Process noise (batch_size, state_dim)
        - v_t: Measurement noise (batch_size, output_dim)

        Returns:
        - x_{t+1}: Next state (batch_size, state_dim)
        - y_t: Observation (batch_size, output_dim)
        """
        # State transition equation
        x_t1 = torch.matmul(x_t, self.A.t()) + torch.matmul(u_t, self.B)
        
        # Add process noise
        process_noise = self.process_noise_std * torch.randn_like(x_t1)
        x_t1 = x_t1 + process_noise
        
        # Observation equation
        y_t = torch.matmul(x_t1, self.C.t()) + torch.matmul(u_t, self.D)
        
        # Add measurement noise
        measurement_noise = self.measurement_noise_std * torch.randn_like(y_t)
        y_t = y_t + measurement_noise
        
        return x_t1, y_t


# Initialize the model
state_dim = 4
input_dim = 768  # Corresponds to the GPT-2 embedding dimension
output_dim = 3

model = StateSpaceModel(state_dim, input_dim, output_dim)

# Function to train the model on a text file
def train_model(model, tokenizer, file_path, epochs=100, lr=0.001, batch_size=2, seq_len=50):
    # Load the text file
    with open(file_path, 'r') as f:
        text_data = f.read()

    # Tokenize the text data
    input_ids = tokenizer.encode(text_data, add_special_tokens=False)
    input_tensor = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    
    # Initialize state tensor (randomly for simplicity)
    x_t = torch.randn(state_dim).unsqueeze(0)
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Prepare inputs and targets
        batch_input = torch.randn(batch_size, seq_len, input_dim)
        batch_target = torch.randn(batch_size, seq_len, output_dim)
        
        # Forward pass
        x_t1, y_t = model(x_t, batch_input)
        
        # Compute loss
        loss = criterion(y_t, batch_target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

    print("Training completed.")

# Function to perform inference on a new text input
def infer(model, tokenizer, text):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_tensor = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    
    # Initialize state tensor (randomly for simplicity)
    x_t = torch.randn(1, state_dim)
    
    # Perform inference
    with torch.no_grad():
        x_t1, y_t = model(x_t, input_tensor.float())
        predicted_ids = torch.argmax(y_t, dim=-1).squeeze().tolist()
    
    # Decode the output to text
    decoded_text = tokenizer.decode(predicted_ids)
    print("Decoded Text:", decoded_text)

# Function to export the model to ONNX
def export_to_onnx(model, tokenizer, text, file_name="state_space_model.onnx"):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    dummy_input = torch.randn(1, len(input_ids), embed_dim)

    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, file_name, input_names=['input'], output_names=['output'])
    print(f"Model has been converted to ONNX format as {file_name}.")

# Main execution
if __name__ == "__main__":
    # Train the model on a text file (replace 'data.txt' with your actual file path)
    train_model(model, tokenizer, 'data.txt')

    # Perform inference on the text "how are you"
    infer(model, tokenizer, "how are you")

    # Export the trained model to ONNX format
    export_to_onnx(model, tokenizer, "how are you")
