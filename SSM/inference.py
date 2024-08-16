import os
import torch
from transformers import AutoTokenizer
from train import SSMTransformerModel  # Import the correct model class from train.py

# Inference function
def generate_text(model, tokenizer, initial_prompt, num_generations=10, device='cpu'):
    model.to(device)
    model.eval()
    
    # Encode the initial prompt
    input_ids = tokenizer.encode(initial_prompt, return_tensors='pt').to(device)
    window_size = model.window_size  # Ensure this matches the model's expected window size

    # Ensure the input sequence length matches the window size
    if input_ids.size(1) < window_size:
        # Pad the input with value 12 if it's shorter than the window size
        padding_length = window_size - input_ids.size(1)
        padding = torch.full((1, padding_length), 12, dtype=torch.long, device=device)
        input_ids = torch.cat((padding, input_ids), dim=1)
    elif input_ids.size(1) > window_size:
        # Truncate the input if it's longer than the window size
        input_ids = input_ids[:, -window_size:]

    for _ in range(num_generations):
        with torch.no_grad():
            output = model(input_ids)
        
        # Assuming output shape is [batch_size, seq_len, vocab_size]
        logits = output[0, -1, :]  # Get logits for the last token position
        
        # Get the predicted token ID
        predicted_token_id = torch.argmax(logits, dim=-1).item()
        
        # Append the predicted token ID to the input
        input_ids = torch.cat((input_ids, torch.tensor([[predicted_token_id]], device=device)), dim=1)
        
        # Ensure the new sequence length matches the window size
        if input_ids.size(1) > window_size:
            input_ids = input_ids[:, -window_size:]

        # Decode the generated tokens to text
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

# Main script to load checkpoint and generate text
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    vocab_size = tokenizer.vocab_size
    embed_dim = 768
    window_size = 64  # Example value; adjust as needed
    block_size = 32   # Example value; adjust as needed
    
    model = SSMTransformerModel(vocab_size=vocab_size, embed_dim=embed_dim, window_size=window_size, block_size=block_size).to(device)
    
    checkpoint_path = 'checkpoint_step_300000.pth'
    if os.path.exists(checkpoint_path):
        print(f'Loading model from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f'Checkpoint not found: {checkpoint_path}')
        exit(1)

    initial_prompt = "In dusk’s light soft hues drift Pink orange sky’s gift Clouds move slow and kind Echoes of twilight unwind You have made fair work"
    generate_text(model, tokenizer, initial_prompt, num_generations=10, device=device)
