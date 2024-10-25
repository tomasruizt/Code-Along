from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


device = "cuda"
batch_size = 4
learning_rate = 0.001
eval_iters = 100
n_embed = 32

text: str = (Path(__file__).parent / "tiny-shakespeare.txt").read_text()
chars = sorted(list(set(text)))
vocab_size = len(chars)  # =65

char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}


def encode_string(string: str) -> torch.Tensor:
    return [char_to_int[ch] for ch in string]


def decode_ints(ints: list[int]) -> str:
    return "".join(int_to_char[i] for i in ints)


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

torch.manual_seed(42)


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
    for steps in range(5000):
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
        logits = self.lm_head(token_embed + pos_embed)  # (B,T,V)
        return logits

    def generate(self, x, max_new_tokens: int) -> torch.Tensor:
        # x: (B,T)
        for _ in range(max_new_tokens):
            idx_next = self._generate_step(x)
            x = torch.cat([x, idx_next], dim=1)  # (B,T+1)
        return x

    def _generate_step(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        logits = self.logits(x)  # (B,T,V)
        last_logits = logits[:, -1, :]  # (B,V)
        probs = F.softmax(last_logits, dim=1)  # (B,V)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
        return idx_next


xb, yb = get_batch(train_data)
m = BigramLM()
m.to(device)
logits, loss = m(xb, yb)


# TRAIN
# train(model=m)
x_ = torch.randn(4, 8, 2)  # (B,T,V)
wei = torch.tril(torch.ones(8, 8))  # (T,T)
wei = wei / wei.sum(dim=1, keepdim=True)  # (B,T,V)
print((wei @ x_).shape)
