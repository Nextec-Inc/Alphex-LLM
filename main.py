import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import model 
from tqdm import tqdm
from torchtext.datasets import WikiText2
from transformers import BertTokenizer
import requests 
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size = 50000  # Vocabulary size
hidden_size = 256  # Hidden size for embeddings and model layers
layers = 6  # Number of  encoder layers
heads = 8  
max_sequence_len = 1024 # Maximum sequence length for input and output

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# Set up data loaders
def data_process(text, seq_length, stride):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = []

    for start in range(0, len(tokens), stride):
        end = min(start + seq_length, len(tokens))
        chunk = tokens[start:end]
        chunk = chunk + [0] * (seq_length - len(chunk))  # Pad the chunk to match the desired sequence length
        chunks.append(chunk)

    # Convert chunks to a PyTorch tensor
    tensor = torch.tensor(chunks)
    return tensor
def decode(tensor):
    tensor = tensor.tolist()
    decoded_text = []

    for chunk in tensor:
        # Remove padding (0) and convert integers to words
        words = [tokenizer.decode([index]) for index in chunk if index != 0]
        # Join the words to form the decoded string
        decoded_text.append(' '.join(words))

    return decoded_text

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
#Encode data 
train_data = data_process(train_dataset , seq_length, 128)
val_data = data_process(valid_dataset , seq_length, 128)
test_data = data_process(test_dataset, seq_length, 128)

batch_size = 32
bptt_len = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


# Initialize the model
mod = model.Alphex(vocab_size, hidden_size, layers, heads, max_sequence_len).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
weight_decay = 1e-5
optimizer = optim.AdamW(mod.parameters(), lr=3e-5, weight_decay=weight_decay)

# Training loop
def train(mod, iterator, optimizer, criterion):
    mod.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(iterator):
        optimizer.zero_grad()
        batch = batch.to(device)

        # Forward pass 
        logits = mod(batch[:,:-1].contiguous())
        targets = batch[:, 1:].contiguous().view(-1)
        loss = criterion(logits.view(-1, vocab_size), targets)

        # L2 regularization
        l2_loss = 0.0
        for param in mod.parameters():
            l2_loss += torch.norm(param, p=2)

        loss += weight_decay * l2_loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_tokens += targets.size(0)

    return total_loss / total_tokens

# Evaluation loop
def evaluate(mod, iterator, criterion):
    mod.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(iterator):
            batch = batch.to(device)

            # Forward pass
            logits = mod(batch[:, :-1])
            targets = batch[:, 1:].contiguous().view(-1)
            loss = criterion(logits.view(-1, vocab_size), targets)

            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

    return total_loss / total_tokens


if __name__ == '__main__':
# Training
    num_epochs = 5
    print(f"Alphex model parameter count :{sum(p.numel() for p in mod.parameters() if p.requires_grad)}")
    for epoch in range(num_epochs):
      train_loss = train(mod, train_loader, optimizer, criterion)
      val_loss = evaluate(mod, val_loader, criterion)
      print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Testing
    test_loss = evaluate(mod, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    o = mod.generate(input_ids.to(device))
    print(decode(o))
    torch.save(mod ,'Alphex.pt')
