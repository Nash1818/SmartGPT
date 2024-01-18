import torch
import torch.nn as nn
import torch.nn.functional as F
import mmap
import random
import pickle
import argparse

# inserting through command line arguments:::

parser = argparse.ArgumentParser(description = "DemoScript")
# adding an argument to the parser:
parser.add_argument('-batch_size', type = str, required = True, help = "Provide a batch_size")
args = parser.parse_args()
print(f'batch_size: {args.batch_size}')

# HyperParameters:
block_size = 64
# batch_size = 128
batch_size = args.batch_size
epochs = 1000
learning_rate = 3e-4
val_epochs = 100
n_embd = 384
n_layer = 4
n_head = 4
dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

chars = ""
with open('vocab.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
    chars = sorted(set(text))
vocab_size = len(chars)
vocab_size

# encoding and decoding:
string_to_int = {ch: i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# memory map for using small snippets of text from a single file of any size:
"""def get_random_chunk(split):
    filename = 'train_split.txt' if split == "train" else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as mm:
            # determine the file size and a random position to start reading:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)
            
            # seek to the random position and read the block of text:
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            
            # decode the block to a string, ignoring invalid byte sequences:
            # just wrap around and continue:
            decoded_block = block.decode('utf-8', errors = 'ignore').replace('\r', '')
            
            # Train and Test splits:
            data = torch.tensor(encode(decoded_block), dtype = torch.long)
    return data


# get_batch function:
def get_batch(split):
    #data = train_data if split == "train" else val_data
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+ block_size] for i in ix])
    y = torch.stack([data[i+1:i+ block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y"""

# initial no gradient:
"""@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train","val"]:
        losses = torch.zeros(10)
        for k in range(10):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out"""

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd,head_size, bias = False)
        # register buffer saves computation and helps save training time: 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input size (batch, time-step, channels)
        # output size(batch, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores:
        wei = q @ k.transpose(-2,-1) * k.shape[-1] **-0.5 
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim = -1) # along the last dimension (B,T,T)
        wei = self.dropout(wei)
        # perform weighted aggregation of values:
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # (B, T, C = F => [h1,h1,h1,h1,h2,h2,h2.....]-> dim = 1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Simple feed forward architecture of a linear layer with a non linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

# Let's create the decoder block class for the GPTModel:
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # no of features each head will be capturing is head_size: 384/4 = 96 features
        
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        # Post normalization used here as given in paper of transformer:
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

# main model:
class gptmodel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        # Now to make 4 decoder blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # for layer normalization: with nn.Linear
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Now apply initial weights:
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
    
    def forward(self, index, targets = None):
        # logits = self.token_embedding_table(index)
        # B, T, C = logits.shape
        B, T = index.shape
        # idx and targets are both (B,T) tensor of integers:
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self.forward(index)
            # focus only on last time step:
            logits = logits[:,-1,:]
            # apply softmax to get probabilities:
            probs = F.softmax(logits, dim = -1) # (B, C)
            # sample from the distribution:
            index_next = torch.multinomial(probs, num_samples = 1) #(B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim = 1) # (B, T+1)
        return index

model = gptmodel(vocab_size)
#m = model.to(device)

print("Loading....")
with open('model-01.pkl','rb') as f:
    model = pickle.load(f)
print("loaded")
m = model.to(device)

"""# create a training loop:
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(1000):
    if iter % 100 == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
    
    # sample a batch of data:
    xb, yb = get_batch("train")
    
    # evaluate the loss:
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
print(loss.item())

# pickle works only on one GPU core whereas torch.load can use multiple:
with open('model-01.pkl','wb') as f:
    pickle.dump(model, f)
print("model saved!")"""

"""context = torch.zeros((1,1), dtype = torch.long, device = device)
generated_chars = decode(m.generate(context, max_new_tokens = 500)[0].tolist())
print(generated_chars)"""