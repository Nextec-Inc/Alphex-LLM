import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from src.model import Alphex
from src.tokenizer import BpeTokenizer as Tokenizer 
from src.train import train , evaluate
from tqdm import tqdm
import requests 
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BpeTokenizer = Tokenizer()
class Config:
    def __init__(self, hidden_size, num_layers, num_heads, dropout_rate=0.1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

# Define different configurations
config_100M = Config(hidden_size=512, num_layers=6, num_heads=8, dropout_rate=0.1)
config_350M = Config(hidden_size=1024, num_layers=12, num_heads=16, dropout_rate=0.1)
config_2B = Config(hidden_size=2048, num_layers=24, num_heads=16, dropout_rate=0.1)
config_10B = Config(hidden_size=4096, num_layers=32, num_heads=32, dropout_rate=0.1)

config = config_2B
#Hyperparameters
vocab_size = 50000  # Vocabulary size
hidden_size = config.hidden_size # Hidden size for embeddings and model layers
layers = config.num_layers   # Number of  encoder layers
heads = config.num_heads
max_sequence_len = 1024 # Maximum sequence length for input and outpuoutput

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
 
train_data = BpeTokenizer.encode(train_dataset)
val_data = BpeTokenizer.encode(valid_dataset)
test_data = BpeTokenizer.encode(test_dataset)
batch_size = 32
bptt_len = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


# Initialize the model
model = Alphex(vocab_size, hidden_size, layers, heads, max_sequence_len).to(device)
mod = torch.compile(model)
mod.to(device)
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
    print(BpeTokenizer.decode(o))
    torch.save(mod ,'Alphex.pt')
