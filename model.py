import torch
import torch.nn as nn
import torch.nn.functional as F


class Alphex(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_sequence_len, dropout_rate=0.1):
        super(Alphex, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = ContextualEncoder(hidden_size, num_layers, num_heads, dropout_rate)
        self.decoder = TextDecoder(hidden_size, num_layers, num_heads, dropout_rate)
        self.decoder_linear = nn.Linear(hidden_size, vocab_size)
        self.max_sequence_len = max_sequence_len

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        encoded = self.encoder(embedded)
        decoded = self.decoder(encoded)
        logits = self.decoder_linear(decoded)
        return logits[:, -self.max_sequence_len:, :]

    def generate(self, input_ids, max_length=20, temperature=1.0):
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.size(0)
            output_ids = input_ids

            for _ in range(max_length):
                logits = self.forward(output_ids)[:, -1, :] / temperature
                next_token_ids = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                output_ids = torch.cat((output_ids, next_token_ids), dim=1)

            return output_ids

    def chat(self, prompt, max_length=20, temperature=1.0):
        self.eval()
        with torch.no_grad():
            prompt_ids = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
            generated_ids = self.generate(prompt_ids, max_length=max_length, temperature=temperature)
            generated_text = ' '.join([tokenizer.decode(token_id.item()) for token_id in generated_ids[0]])
            return generated_text

class ContextualEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout_rate=0.1):
        super(ContextualEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class TextDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout_rate=0.1):
        super(TextDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        residual = inputs
        inputs = self.layer_norm1(inputs)
        attended = self.multihead_attention(inputs, inputs, inputs)[0]
        inputs = residual + self.dropout(attended)

        residual = inputs
        
        

        inputs = self.layer_norm2(inputs)
        feed_forward_output = self.feed_forward(inputs)
        inputs = residual + self.dropout(feed_forward_output)

        return inputs
