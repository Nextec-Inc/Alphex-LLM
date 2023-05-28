from tqdm import tqdm
import torch
# Training loop
def train(mod, iterator, optimizer, criterion, weight_decay, vocab_size, device):
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
def evaluate(mod, iterator, criterion, vocab_size, device):
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






