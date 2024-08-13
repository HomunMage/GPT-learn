import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Load the tokenizer and set embedding dimension
tokenizer = AutoTokenizer.from_pretrained("gpt2")
embed_dim = 768  # Use the GPT-2 embedding dimension

# Define the Attention Model
class AttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(AttentionModel, self).__init__()
        
        # Linear layers to project input to Query, Key, and Value spaces
        # Q: Query matrix
        # K: Key matrix
        # V: Value matrix
        self.query_linear = nn.Linear(vocab_size, embed_dim)
        self.key_linear = nn.Linear(vocab_size, embed_dim)
        self.value_linear = nn.Linear(vocab_size, embed_dim)
        
        # Linear layer to project output back to vocab_size
        self.output_linear = nn.Linear(embed_dim, vocab_size)

    # Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        if mask is not None:
            scores += mask
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

    def forward(self, x, mask=None):
        # Project input x to Query, Key, and Value
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)
        
        # Compute attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Project attention output back to vocab_size
        output = self.output_linear(attention_output)
        
        return output

# Initialize the model
vocab_size = tokenizer.vocab_size
model = AttentionModel(vocab_size, embed_dim)

# Function to train the model on a text file with B, T, C format
def train_model(model, tokenizer, file_path, epochs=100, lr=0.001, batch_size=2, seq_len=50):
    # Load the text file
    with open(file_path, 'r') as f:
        text_data = f.read()

    # Tokenize the text data
    input_ids = tokenizer.encode(text_data, add_special_tokens=False)

    # Split the input into batches and sequences
    num_batches = len(input_ids) // (batch_size * seq_len)
    input_ids = input_ids[:num_batches * batch_size * seq_len]  # Trim the input to fit the batch size and seq_len

    # Reshape into (B, T) format
    input_tensor = torch.tensor(input_ids).view(batch_size, num_batches * seq_len)

    # Target tensor is the same as input tensor for next-token prediction
    target_tensor = input_tensor.clone()

    # Convert input_tensor to one-hot encoding (B, T, C)
    input_tensor_one_hot = torch.nn.functional.one_hot(input_tensor, num_classes=vocab_size).float()

    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for i in range(num_batches):
            batch_input = input_tensor_one_hot[:, i * seq_len: (i + 1) * seq_len, :]
            batch_target = target_tensor[:, i * seq_len: (i + 1) * seq_len]

            # Forward pass
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output.view(-1, vocab_size), batch_target.view(-1))
            
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

    # Convert to one-hot encoding
    input_tensor_one_hot = torch.nn.functional.one_hot(input_tensor, num_classes=vocab_size).float()

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor_one_hot)
        predicted_ids = torch.argmax(output, dim=-1).squeeze().tolist()

    # Decode the output to text
    decoded_text = tokenizer.decode(predicted_ids)
    print("Decoded Text:", decoded_text)

# Function to export the model to ONNX
def export_to_onnx(model, tokenizer, text, file_name="attention_model.onnx"):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    dummy_input = torch.randn(1, len(input_ids), vocab_size)

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
