# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# training data
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
def load_input_txt():
    with open('input.txt', 'r') as file:
        return file.read()

# Define the model
class OneLayerAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size):
        super(OneLayerAttentionModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.query_linear = nn.Linear(embed_dim, embed_dim)  # Added for Q calculation

        self.fc = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=Q.device))
        if mask is not None:
            scores += mask
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def forward(self, x):
        batch_size, seq_length = x.size()

        # Ensure sequence length doesn't exceed block size
        if seq_length > self.block_size:
            raise ValueError(f"Sequence length ({seq_length}) exceeds block size ({self.block_size})")

        token_embeddings = self.token_embedding_table(x)
        position_ids = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding_table(position_ids)
        
        x = token_embeddings + position_embeddings

        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)
        x = self.scaled_dot_product_attention(Q, K, V)
        
        x = self.fc(x)
        return x

# Custom dataset
class RollingWindowDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Training function and checkpoint saving
def train_model(model, dataloader, optimizer, criterion, device='cpu', print_interval=100, checkpoint_interval=10000, start_step=0):
    model.to(device)
    model.train()
    step = start_step
    running_loss = 0.0

    for epoch in range(start_step, len(dataloader) + start_step):
        if step < start_step:
            # Skip batches until we reach the start_step
            for _ in range(start_step - step):
                next(iter(dataloader))
                step += 1
                if step >= start_step:
                    break

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            output = output.view(-1, output.size(-1))
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


    input_txt = load_input_txt()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(input_txt)
    print(f'Token length after tokenization: {len(input_ids)}')

    # Define block size and create dataset
    block_size = 32  # Should match the block size used in your model
    dataset = RollingWindowDataset(input_ids, block_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Define model and move it to the device
    vocab_size = tokenizer.vocab_size
    embed_dim = 768
    model = OneLayerAttentionModel(vocab_size=vocab_size, embed_dim=embed_dim, block_size=block_size).to(device)

    # Initialize optimizer after moving the model to the device
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Optionally load from a checkpoint
    start_step = 0
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
    checkpoint_interval = 20000
    train_model(model, dataloader, optimizer, criterion, device=device, print_interval=100, checkpoint_interval=checkpoint_interval, start_step=start_step)
