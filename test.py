# test.py

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Define the model with one attention layer
class OneLayerAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(OneLayerAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Add embedding layer
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, vocab_size)  # Output dimension should be vocab_size
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        device = Q.device
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=device))
        if mask is not None:
            scores += mask
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def forward(self, x, mask=None):
        x = self.embedding(x)  # Convert token IDs to embeddings
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.out_linear(attention_output)
        return output

# Inference function
def generate_text(model, tokenizer, initial_prompt, num_generations=10, device='cpu'):
    model.to(device)  # Move model to the specified device
    model.eval()
    input_ids = tokenizer.encode(initial_prompt, return_tensors='pt').to(device)  # Move input_ids to device

    for _ in range(num_generations):
        with torch.no_grad():
            output = model(input_ids.squeeze(0))  # Remove batch dimension if present

        # Handle different output shapes
        if len(output.shape) == 3:
            logits = output[0, -1, :]  # Shape [batch_size, seq_length, vocab_size]
        elif len(output.shape) == 2:
            logits = output[-1, :]  # Shape [seq_length, vocab_size]
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        predicted_token_id = torch.argmax(logits, dim=-1).item()

        input_ids = torch.cat((input_ids, torch.tensor([[predicted_token_id]], device=device)), dim=1)

        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

# Main script to load checkpoint and generate text
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Define model parameters
    vocab_size = tokenizer.vocab_size
    embed_dim = 768
    
    # Initialize model and move to device
    model = OneLayerAttentionModel(vocab_size=vocab_size, embed_dim=embed_dim).to(device)
    
    # Load from the checkpoint
    checkpoint_path = 'checkpoint_step_320000.pth'
    if os.path.exists(checkpoint_path):
        print(f'Loading model from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f'Checkpoint not found: {checkpoint_path}')
        exit(1)

    # Generate text with the initial prompt
    initial_prompt = "she is warrior"
    generate_text(model, tokenizer, initial_prompt, num_generations=10, device=device)
