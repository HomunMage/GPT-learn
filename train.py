# train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


window_size = 128


# training data
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
def load_input_txt():
    with open('input.txt', 'r') as file:
        return file.read()


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
        # Ensure the sqrt operation is on the correct device

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=Q.device))
        if mask is not None:
            scores += mask
        attention_weights = F.softmax(scores, dim=-1)
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

# Custom dataset class for rolling window
class RollingWindowDataset(Dataset):
    def __init__(self, input_ids, window_size):
        self.input_ids = input_ids
        self.window_size = window_size

    def __len__(self):
        return len(self.input_ids) - self.window_size + 1

    def __getitem__(self, idx):
        x_ids = self.input_ids[idx:idx + self.window_size]
        y_ids = self.input_ids[idx + 1:idx + self.window_size + 1]
        x = torch.tensor(x_ids, dtype=torch.long)  # Use long for token IDs
        y = torch.tensor(y_ids, dtype=torch.long)  # Use long for target token IDs
        return x, y



# Training function and checkpoint saving
def train_model(model, dataloader, optimizer, criterion, device='cpu', print_interval=100, checkpoint_interval=10000, start_step=0):
    model.to(device)  # Move model to the specified device
    model.train()

    step = start_step
    running_loss = 0.0

    # Iterate through DataLoader and skip batches until reaching the desired step
    for epoch in range(start_step, len(dataloader) + start_step):
        if step < start_step:
            # Skip batches until we reach the start_step
            for _ in range(start_step - step):
                next(iter(dataloader))  # Move to next batch
                step += 1
                if step >= start_step:
                    break

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move data to the specified device
            optimizer.zero_grad()

            output = model(x_batch)  # Use the batch dimension correctly

            output = output.view(-1, output.size(-1))  # Flatten for loss calculation
            y_batch = y_batch.view(-1)
            loss = criterion(output, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % print_interval == 0:
                avg_loss = running_loss / print_interval
                print(f'Epoch {step // len(dataloader) + 1}, Step {step}/{len(dataloader) + start_step}, Loss: {avg_loss:.4f}')
                running_loss = 0.0

            # Save checkpoint
            if step % checkpoint_interval == 0:
                checkpoint_path = f'checkpoint_step_{step}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step
                }, checkpoint_path)
                print(f'Saved checkpoint: {checkpoint_path}')

    # Save final model
    final_checkpoint_path = 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }, final_checkpoint_path)
    print(f'Saved final model: {final_checkpoint_path}')




# Main script
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data and tokenizer
    input_txt = load_input_txt()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(input_txt)

    # Print the length of tokens
    print(f'Token length after tokenization: {len(input_ids)}')

    # Define window size and create dataset
    dataset = RollingWindowDataset(input_ids, window_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Define model and move it to the device
    vocab_size = tokenizer.vocab_size  # Get vocab size from tokenizer
    embed_dim = 768
    model = OneLayerAttentionModel(vocab_size=vocab_size, embed_dim=embed_dim).to(device)

    # Initialize optimizer after moving the model to the device
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Optionally load from a checkpoint
    start_step = 0  # Set this to the last checkpoint step you want to resume from
    checkpoint_path = f'checkpoint_step_{start_step}.pth'
    if os.path.exists(checkpoint_path):
        print(f'Loading model and optimizer from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
    else:
        start_step = 0

    # Training the model
    checkpoint_interval = 20000  # Set the checkpoint interval
    train_model(model, dataloader, optimizer, criterion, device=device, print_interval=100, checkpoint_interval=checkpoint_interval, start_step=start_step)
