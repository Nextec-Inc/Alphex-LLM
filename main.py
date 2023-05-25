import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import model 
from tqdm import tqdm
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertTokenizer
import datasets
import urllib
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

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filename = "tinyshakespeare.txt"
if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)

# Read the dataset
with open(filename, 'r') as f:
    text = f.read()

# Split the dataset into train, validation, and test sets
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

num_chars = len(text)
train_cutoff = int(num_chars * train_frac)
val_cutoff = int(num_chars * (train_frac + val_frac))

train_dataset , valid_dataset ,  test_dataset = WikiText2()

# Build vocabulary from the training set
vocab = build_vocab_from_iterator(map(tokenizer, train_dataset))

# Set up data loaders
def data_process(text, seq_length):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    # Truncate or pad the sequence to a fixed length
    tokens = tokens[:seq_length] if len(tokens) > seq_length else tokens + [0] * (seq_length - len(tokens))
    # Convert tokens to a PyTorch tensor
    tensor = torch.tensor(tokens)
    return tensor
    

def decode(tensor):

    tensor = tensor.tolist()
    # Remove padding (0) and convert integers to words
    words = [tokenizer.decode([index]) for index in tensor if index != 0]
    # Join the words to form the decoded string
    decoded_text = ' '.join(words)
    return decoded_text

seq_length = 64  # Desired sequence length

train_data = data_process(train_dataset , seq_length)
val_data = data_process(valid_dataset , seq_length)
test_data = data_process(test_dataset, seq_length)

batch_size = 32
bptt_len = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Alphex model n


# Initialize the model
model = model.Alphex(vocab_size, hidden_size, layers, heads, max_sequence_len).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(iterator):
        optimizer.zero_grad()
        batch = batch.to(device)

        # Forward pass 
        logits = model(batch[:,:-1].contiguous())
        targets = batch[:, 1:].contiguous().view(-1)
        loss = criterion(logits.view(-1, vocab_size), targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_tokens += targets.size(0)

    return total_loss / total_tokens

# Evaluation loop
def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(iterator):
            batch = batch.to(device)

            # Forward pass
            logits = model(batch[:, :-1])
            targets = batch[:, 1:].contiguous().view(-1)
            loss = criterion(logits.view(-1, vocab_size), targets)

            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

    return total_loss / total_tokens

if __name__ == '__main__':
# Training
    num_epochs = 3

    for epoch in range(num_epochs):
      train_loss = train(model, train_loader, optimizer, criterion)
      val_loss = evaluate(model, val_loader, criterion)
      print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Testing
    test_loss = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    o = model.generate(input_ids.to(device))
    print(decode(o))
    torch.save(model ,'Alphex-BiDEN-Pretrained.pt')
