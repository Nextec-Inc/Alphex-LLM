import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import model 
from tqdm import tqdm
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size = 50000  # Vocabulary size
hidden_size = 256  # Hidden size for embeddings and model layers
layers = 6  # Number of  encoder layers
heads = 8  
max_sequence_len = 128  # Maximum sequence length for input and output

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Load WikiText2 dataset
train_dataset, valid_dataset, test_dataset = WikiText2()

# Build vocabulary from the training set
vocab = build_vocab_from_iterator(map(tokenizer, train_dataset))

# Set up data loaders
def data_process(raw_text_iter):
    data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                         dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_data = data_process(train_dataset)
val_data = data_process(valid_dataset)
test_data = data_process(test_dataset)

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
        logits = model(batch[:,:-1])
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
    num_epochs = 10

    for epoch in range(num_epochs):
      train_loss = train(model, train_loader, optimizer, criterion)
      val_loss = evaluate(model, val_loader, criterion)
      print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Testing
    test_loss = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    torch.save('Alphex-BiDEN-Pretrained.pt')