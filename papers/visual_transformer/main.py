import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms

P = 16  # Patch size
D = 256  # Hidden dimension
C = 3  # channels
MAX_CTX_LEN = 1024
N_BLOCKS = 4
NUM_CLASSES = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-3


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.mha = nn.MultiheadAttention(
            embed_dim=D, num_heads=D // 64, dropout=DROPOUT
        )
        self.mlp = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.ReLU(),
            nn.Linear(4 * D, D),
        )

    def forward(self, x):
        x_ = self.ln1(x)
        x = self.mha(key=x_, query=x_, value=x_)[0] + x
        x = self.mlp(self.ln2(x)) + x
        return x


class ViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Linear(P**2 * C, D)
        self.class_token = nn.Embedding(num_embeddings=1, embedding_dim=D)
        self.pos_embed = nn.Embedding(num_embeddings=MAX_CTX_LEN, embedding_dim=D)
        self.mha_blocks = nn.ModuleList([Block() for _ in range(N_BLOCKS)])
        self.classification_head = nn.Linear(D, NUM_CLASSES)

    def forward(self, flattened):  # N,D
        N, D = flattened.shape
        pos_embeds = self.pos_embed(torch.arange(N + 1))  # N+1,D
        patch_embeds = self.project(flattened)  # N,D
        class_token = self.class_token(torch.LongTensor([0]))  # 1,D
        patch_embeds = torch.cat([class_token, patch_embeds])  # N+1,D
        x = patch_embeds + pos_embeds
        for block in self.mha_blocks:
            x = block(x)
        logits = self.classification_head(x[0, :])
        return logits

    def patch_and_flatten(self, pil_img: Image.Image) -> torch.Tensor:
        img_w, img_h = pil_img.size
        num_patches_w, num_patches_h = img_w // P, img_h // P
        print("num patches: %d by %d" % (num_patches_w, num_patches_h))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((num_patches_w * P, num_patches_h * P)),
            ]
        )
        img: torch.Tensor = preprocess(pil_img)  # C,W,H
        C, W, H = img.shape
        patched = img.view(-1, C, P, P)  # desired: N,C,P,P
        flattened = patched.flatten(start_dim=1)  #  desired: N,P^2 * C
        return flattened


model = ViT()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
pil_img = Image.open("ocean.jpg")
flattened = model.patch_and_flatten(pil_img)
target = torch.tensor([1.0, 0])

for _ in range(3):
    opt.zero_grad()
    logits = model(flattened)  # D
    loss = loss_fn(logits, target)
    print(loss.item())
    loss.backward()
    opt.step()
