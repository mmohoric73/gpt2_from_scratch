# import re

text = "Hello World. This is an arbitrary test thing with many great words and cool special characters like ? and , and ] and :"

# result = re.split(r'([,.:;?_!()\']|--|\s)', text)
# result = [item.strip() for item in result if item.strip()]
# preprocessed = result

# all_words = sorted(set(result))

# vocab = {token:integer for integer,token in enumerate(all_words)}


# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s, i in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!()\']|--|\s)', text)

#         preprocessed = [item.strip() for item in preprocessed if item.strip()]

#         preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i]for i in ids])
#         # replace spaces before the specified punctuations
#         text = re.sub(r'([,.:;?_!()\']|--|\s)', r'\1', text)
#         return text
# tokenizer = SimpleTokenizerV1(vocab)
# tokenizer.encode(text)


# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# vocab = {token:integer for integer,token in enumerate(all_tokens)}




# tokenizer = tiktoken.get_encoding("gpt2")
# tokenizer.encode(text, allowed_special = {"<|endoftext|>"})



# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# enc_text = tokenizer.encode(raw_text)

# context_size = 4

# x = enc_text[:context_size]
# y = enc_text[1:]


# import tiktoken

# import torch

# from torch.utils.data import Dataset, DataLoader


# class GPTDatasetV1 (Dataset):
#     def __init__(self, txt, tokenizer, max_length, stride):
#         self.input_ids = []
#         self.target_ids = []
#         # will run out of memory if dataset is too large (cap is ballpark a few gigabytes of input data)

#         # Tokenize entire text
#         token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

#         # using a sliding window to chunk the book into overlapping sequences of max_length
#         for i in range(0, len(token_ids)) - max_length, stride:
#             input_chunk = token_ids[i:i+max_length]
#             target_chunk = token_ids[i + 1: i + max_length +1]
#             self.input_ids.appnd(torch.tensor(input_chunk))
#             self.target_ids.append(torch.tensor(target_chunk))

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         return self.input_ids[idx], self.target_ids[idx]


# def create_dataloader_v1(txt, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0):
#     # Initialize tiktokenizer
#     tokenizer = tiktoken.get_encoding("gpt2")

#     # Create dataset
#     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

#     # Create dataloader
#     dataloader = DataLoader(
#         dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers
#     )

#     return dataloader


# vocab_size = 50257
# output_dim = 256

# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# context_length = max_length
# pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)


# example of computing attn context vectors for a single entry in a set of inputs
# inputs = torch.tensor(
#     [
#         [0.43, 0.15, 0.89],[0.55, 0.87, 0.66], [0.57, 0.85, 0.64], [0.22, 0.58, 0.33], [0.77, 0.25, 0.10], [0.05, 0.80, 0.55]
#     ]
# )
# input_query = inputs[1]

# attn_scores_2 = torch.empty(inputs.shape[0])
# for idx, x_i in enumerate(inputs):
#     attn_scores_2[idx] = torch.dot(x_i, input_query)

# attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# query = inputs[1]

# context_vec_2 = torch.zeros(query.shape)
# for i,x_i in enumerate (inputs):
#     # print(f'{attn_weights_2[i]} ----> {inputs[i]}')
#     context_vec_2 += attn_weights_2[i]*inputs[i]
# print(context_vec_2)



# generalizing attn context vector computation

# attn_scores = inputs @ inputs.T
# attn_weights = torch.softmax(attn_scores, dim=1)
# all_context_vecs = attn_weights @ inputs



# # example of computing context vector using queries keys and values
# # x_2 = inputs[1]
# d_in = inputs.shape[1]
# d_out = 2

# # torch.manual_seed(123)

# # W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
# # W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
# # W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

# # query_2 = x_2 @ W_query

# # keys = inputs @ W_key
# # value = inputs @ W_value

# # keys_2 = keys[1]
# # attn_scores_22 = torch.dot(query_2, keys_2)



# # attn_scores_2 = query_2 @ keys.T




# # attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)

# # context_vec_2 = attn_weights_2 @ value



# # import torch.nn as nn

# # class SelfAttention_v1(nn.Module):

# #     def __init__(self, d_in, d_out):
# #         super().__init__()
# #         self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
# #         self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
# #         self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
# #     def forward(self, x):
# #         queries = inputs @ W_query
# #         keys = inputs @ W_key
# #         values = inputs @ W_value

# #         attn_scores = queries @ keys.T
# #         attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

# #         context_vec = attn_weights @ value

# #         return context_vec
# # torch.manual_seed(123)
# # sa_v1 = SelfAttention_v1(d_in, d_out)


# # Preventing overfitting by masking random positions in attn_weights
# torch.manual_seed(123)
# layer = torch.nn.Dropout(0.5)

# batch = torch.stack((inputs, inputs), dim = 0)

# # mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
# # masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

# m = torch.nn.Linear(2,3)

# import torch.nn as nn

# class Causal_Attention(nn.Module):
#     def __init__(self, d_in, d_out, context_length, dropout_rate, q_kv_bias = False):
#         super().__init__()
#         self.W_query = torch.nn.Linear(d_in, d_out, bias = q_kv_bias)
#         self.W_key = torch.nn.Linear(d_in, d_out, bias = q_kv_bias)
#         self.W_value = torch.nn.Linear(d_in, d_out, bias = q_kv_bias)
#         self.dropout = torch.nn.Dropout(dropout_rate)
#         self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal =1))



#     def forward(self, x):
#         b, num_tokens, d_in = x.shape
#         queries = self.W_query(x)
#         keys = self.W_key(x)
#         values = self.W_value(x)

#         attn_scores = queries @ keys.transpose(1,2)
#         attn_scores.masked_fill_(
#             self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

#         attn_weights = torch.softmax(
#             attn_scores / keys.shape[-1]**0.5, dim=-1
#             )

#         context_vec = attn_weights @ values

#         return context_vec

# torch.manual_seed(789)

# context_length = batch.shape[1]
# dropout = 0.0
# ca = Causal_Attention(d_in, d_out, context_length, dropout)


# class MultiHeadAttentionWrapper(nn.Module):
#     def __init__(self, d_in, d_out, context_length, dropout, num_heads =2, qkv_bias=False):
#         super().__init__()
#         self.heads = nn.ModuleList([Causal_Attention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

#     def forward(self, x):
#         return torch.cat([head(x) for head in self.heads], dim=-1)

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



def generate(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
out = generate(model, torch.tensor(encoded).unsqueeze(0), max_new_tokens=10, context_size=1024)
print(tokenizer.decode(out.squeeze(0).tolist()))
