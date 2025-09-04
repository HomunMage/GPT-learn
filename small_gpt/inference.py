# inference.py

import torch
import torch.nn.functional as F
from train import BigramLanguageModel, decode, block_size, device

def load_model(checkpoint_path):
    model = BigramLanguageModel()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    return model

def generate_text(model, max_new_tokens=2000):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens)
    return decode(generated[0].tolist())

if __name__ == "__main__":
    checkpoint_path = "model_final.pth"  # Update this path to the desired checkpoint
    model = load_model(checkpoint_path)
    text = generate_text(model)
    print(text)

