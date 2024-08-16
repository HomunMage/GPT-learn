# test.py

import os
import torch
from transformers import AutoTokenizer
from train import OneLayerAttentionModel  # Import the model class from train.py

# Inference function
def generate_text(model, tokenizer, initial_prompt, num_generations=10, device='cpu'):
    model.to(device)
    model.eval()
    input_ids = tokenizer.encode(initial_prompt, return_tensors='pt').to(device)

    for _ in range(num_generations):
        with torch.no_grad():
            output = model(input_ids.squeeze(0))

        if len(output.shape) == 3:
            logits = output[0, -1, :]
        elif len(output.shape) == 2:
            logits = output[-1, :]
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        predicted_token_id = torch.argmax(logits, dim=-1).item()
        input_ids = torch.cat((input_ids, torch.tensor([[predicted_token_id]], device=device)), dim=1)

        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

# Main script to load checkpoint and generate text
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    vocab_size = tokenizer.vocab_size
    embed_dim = 768
    
    model = OneLayerAttentionModel(vocab_size=vocab_size, embed_dim=embed_dim).to(device)
    
    checkpoint_path = 'checkpoint_step_320000.pth'
    if os.path.exists(checkpoint_path):
        print(f'Loading model from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f'Checkpoint not found: {checkpoint_path}')
        exit(1)

    initial_prompt = "she is warrior"
    generate_text(model, tokenizer, initial_prompt, num_generations=10, device=device)
