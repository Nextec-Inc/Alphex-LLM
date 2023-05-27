import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from src.model import Alphex
import src.tokenizer as tokenizer
from src.train import train , evaluate
from tqdm import tqdm
import requests 
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size = 50000  # Vocabulary size
hidden_size = 256  # Hidden size for embeddings and model layers
layers = 6  # Number of  encoder layers
heads = 8  
max_sequence_len = 1024 # Maximum sequence length for input and output

# Set up data loader
seq_length = 64  # Desired sequence length

def download_file(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

base_url = 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/'
train_url = base_url + 'train.txt'
valid_url = base_url + 'valid.txt'
test_url = base_url + 'test.txt'

# Download the dataset files
train_dataset  = download_file(train_url)
valid_dataset = download_file(valid_url)
test_dataset = download_file(test_url)
tokenizer.BPEncoder.fit(train_dataset)
#Encode data 
train_data = tokenizer.BPEncoder.encode(train_dataset)
val_data = tokenizer.BPEncoder.encode(valid_dataset)
test_data = tokenizer.BPEncoder.encode(test_dataset)

batch_size = 32
bptt_len = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


# Initialize the model
mod = Alphex(vocab_size, hidden_size, layers, heads, max_sequence_len).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
weight_decay = 1e-5
optimizer = optim.AdamW(mod.parameters(), lr=3e-5, weight_decay=weight_decay)

if __name__ == '__main__':
# Training
    num_epochs = 5
    print(f"Alphex model parameter count :{sum(p.numel() for p in mod.parameters() if p.requires_grad)}")
    for epoch in range(num_epochs):

      train_loss = train(mod, train_loader, optimizer, criterion, weight_decay, vocab_size, device)
      val_loss = evaluate(mod, val_loader, criterion, vocab_size, device)

      print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Testing
    test_loss = evaluate(mod, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    o = mod.generate(input_ids.to(device))
    print(tokenizer.BPEncoder.decode(o))
    torch.save(mod ,'Alphex.pt')
