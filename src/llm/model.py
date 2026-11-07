
import tiktoken

import torch

import torch.nn as nn

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads = 2, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)



        context_vec = (attn_weights @ values)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

torch.manual_seed(123)

# Can change to change the model size
GPT_CONFIG_124_M = {"vocab_size": 50257, "context_length": 1024, "emb_dim": 768, "n_heads": 12, "n_layers": 12, "drop_rate": 0.1, "qkv_bias": False
}





# Layer Normalization implementation
class LayerNorm(nn.Module):
    def __init__(self, embedding_dimension):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized_x + self.shift

class Transformer_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config["emb_dim"], config["emb_dim"], config["context_length"], config["drop_rate"], config["n_heads"], config["qkv_bias"])

        self.ffn = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
            nn.Dropout(config["drop_rate"])
        )
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # shortcut connxn for attn block
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x= self.drop_shortcut(x)
        x = x + shortcut


        # shortcut connxn for ffn block
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_embedding = nn.Dropout(config["drop_rate"])


        self.transformer_blocks = nn.Sequential(*[Transformer_Block(config) for _ in range(config["n_layers"])])


        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_length = x.shape
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.pos_embedding(torch.arange(seq_length, device=x.device))

        y = token_embeddings + position_embeddings
        y = self.drop_embedding(y)
        y = self.transformer_blocks(y)
        y = self.final_norm(y)
        logits = self.out_head(y)
        return logits

torch.manual_seed(123)

model = GPTBlock(GPT_CONFIG_124_M)
output = model(batch)



# def generate(model, idx, max_new_tokens, context_size):
#     for _ in range(max_new_tokens):
#         idx_cond = idx[:, -context_size:]
#         logits = model(idx_cond)
#         logits = logits[:, -1, :]
#         probs = torch.softmax(logits, dim=-1)
#         idx_next = torch.argmax(probs, dim=-1, keepdim=True)
#         idx = torch.cat((idx, idx_next), dim=1)
#     return idx


