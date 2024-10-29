from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


device = "cuda"
batch_size = 32
learning_rate = 1e-3
learning_iters = 3000
eval_iters = 200
n_embed = 32
n_heads = 4
n_blocks = 4
dropout = 0.5
assert n_embed % n_heads == 0
torch.manual_seed(1337)

text: str = (Path(__file__).parent / "tiny-shakespeare.txt").read_text()
chars = sorted(list(set(text)))
vocab_size = len(chars)  # =65

char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}


def encode_string(string: str) -> list[int]:
    return [char_to_int[ch] for ch in string]


def decode_ints(ints: list[int]) -> str:
    return "".join(int_to_char[i] for i in ints)


def decode_batch(x: torch.Tensor) -> list[str]:
    return [decode_ints(example) for example in x.tolist()]


def seed_token_for_generation() -> torch.Tensor:
    return torch.tensor(encode_string("\n"), device=device).long().reshape(1, 1)


full_data = torch.LongTensor(encode_string(text)).to(device)
n = int(0.9 * len(full_data))
train_data, val_data = full_data[:n], full_data[n:]
assert len(train_data) + len(val_data) == len(full_data)

# EXAMPLE ABOUT CONTEXT AND TARGET
block_size = 8
x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    ctx = x[: t + 1]
    tgt = y[t]
# END EXAMPLE


def get_batch(data):
    ixs = torch.randint(len(data) - block_size, (batch_size,)).to(device)
    xs = torch.stack([data[i : i + block_size] for i in ixs])
    ys = torch.stack([data[i + 1 : i + block_size + 1] for i in ixs])
    return xs, ys


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    res = {}
    for name, split in [("train", train_data), ("validation", val_data)]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        res[name] = sum(losses) / len(losses)
    model.train()
    return res


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for steps in range(learning_iters):
        xb, yb = get_batch(train_data)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        if steps % 100 == 0:
            loss_est = estimate_loss(model)
            print(
                f"Step: {steps}, train_loss={loss_est['train']:.2f}, val_loss={loss_est['validation']:.2f}"
            )


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_tbl = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_tbl = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_blocks)],
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets):
        logits = self.logits(x)  # (B,T,V) where V=vocab_size
        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def logits(self, x):
        B, T = x.shape
        token_embed = self.token_embedding_tbl(x)  # (B,T,E)
        pos_embed = self.pos_embedding_tbl(torch.arange(T, device=device))  # (T,E)
        out = token_embed + pos_embed
        out = self.blocks(out)
        logits = self.lm_head(out)  # (B,T,V)
        return logits

    def generate(self, x, max_new_tokens: int) -> torch.Tensor:
        # x: (B,T)
        for _ in range(max_new_tokens):
            x_truncated = x[:, -block_size:]  # (B,T)
            idx_next = self._generate_step(x_truncated)
            x = torch.cat([x, idx_next], dim=1)  # (B,T+1)
        return x

    def _generate_step(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        logits = self.logits(x)  # (B,T,V)
        last_logits = logits[:, -1, :]  # (B,V)
        probs = F.softmax(last_logits, dim=1)  # (B,V)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
        return idx_next


class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.head_size = head_size

        tril = torch.triu(torch.ones(block_size, block_size), 1)
        self.register_buffer("tril", tril)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,H)
        q = self.query(x)  # (B,T,H)
        v = self.value(x)  # (B,T,H)
        normalization = self.head_size**-0.5
        wei = torch.einsum("bth, bTh -> btT", q, k) * normalization  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T].bool(), float("-inf"))
        wei = F.softmax(wei, dim=2)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class MultiHeadAttn(nn.Module):
    def __init__(self, n_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa_head = MultiHeadAttn(n_heads=n_heads, head_size=n_embed // n_heads)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


xb, yb = get_batch(train_data)
m = BigramLM()
m.to(device)
logits, loss = m(xb, yb)

train(m)


x = seed_token_for_generation()
outs = m.generate(x=x, max_new_tokens=100)
gen_text = decode_ints(outs[0].cpu().tolist())
print(gen_text)


# B, T, C = 4, 8, 32
# x = torch.randn(B, T, C, device=device)

# head_size = 16


# print(q.var())
# print(k.var())
# print(wei.var())
