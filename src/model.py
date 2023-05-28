import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

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

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(MultiheadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query_projection = nn.Linear(hidden_size, hidden_size * num_heads)
        self.key_value_projection = nn.Linear(hidden_size, hidden_size * num_heads)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(hidden_size * num_heads, hidden_size)
        
    def forward(self, inputs):
        batch_size, seq_len, hidden_size = inputs.size()
        
        # Project inputs to queries, keys, and values
        queries = self.query_projection(inputs)
        keys_values = self.key_value_projection(inputs)
        
        # Reshape queries, keys, and values to incorporate multiple heads
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys_values = keys_values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Split keys and values
        keys = keys_values[:, :, :, :self.head_dim]  # Shape: (batch_size, num_heads, seq_len, head_dim)
        values = keys_values[:, :, :, self.head_dim:]  # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores and apply dropout
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        
        # Apply attention scores to values and concatenate heads
        attended_values = torch.matmul(attention_scores, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Project the attended values and apply dropout
        outputs = self.output_projection(attended_values)
        outputs = self.dropout(outputs)
        
        return outputs

class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.activation = SwiGLU()
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        
    def forward(self, inputs):
        outputs = self.dropout(inputs)
        outputs = self.fc1(outputs)
        outputs = self.activation(outputs)
        outputs = self.fc2(outputs)
        
        return outputs

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_sequence_len):
        super(RotaryPositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.max_sequence_len = max_sequence_len
        self.frequency_bands = hidden_size // 2
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_size, 2) / hidden_size))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, inputs):
        seq_len = inputs.size(1)
        pos_enc = torch.arange(seq_len, dtype=torch.float32, device=inputs.device)
        sinusoid_inp = torch.einsum("i,j->ij", pos_enc, self.inv_freq)
        emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        
        if seq_len < self.max_sequence_len:
            emb = F.pad(emb, (0, self.max_sequence_len - seq_len, 0, 0))
        
        return emb.unsqueeze(0)
                                                                                                                    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1, max_sequence_len=512):
        super(TransformerBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_value_projection = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.rotary_positional_encoding = RotaryPositionalEncoding(hidden_size, max_sequence_len)
        
    def forward(self, inputs):
        residual = inputs
        inputs = self.layer_norm1(inputs)
        
        # Add rotary positional encodings
        seq_len = inputs.size(1)
        rotary_pos_enc = self.rotary_positional_encoding(inputs[:, :, :self.hidden_size])
        inputs = inputs + rotary_pos_enc
        
        # Project inputs to queries, keys, and values
        queries = self.query_projection(inputs)
        keys_values = self.key_value_projection(inputs)
        
        # Reshape queries, keys, and values to incorporate multiple heads
        queries = queries.view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys_values = keys_values.view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Split keys and values
        keys = keys_values[:, :, :, :self.head_dim]
        values = keys_values[:, :, :, self.head_dim:]
        
        # Compute attention scores and apply dropout
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        
        # Apply attention scores to values and concatenate heads
        attended_values = torch.matmul(attention_scores, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(-1, seq_len, self.hidden_size)
        
        # Project the attended values and apply dropout
        attended_values = self.output_projection(attended_values)
        attended_values = self.dropout(attended_values)
        
        # Residual connection and layer normalization
        inputs = residual + attended_values
        inputs = self.layer_norm2(inputs)
        
        return inputs