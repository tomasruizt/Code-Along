import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from torchvision.transforms import ToTensor
from datasets import load_dataset, Dataset

to_tensor = ToTensor()

img_size = 28


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(
                in_features=64 * (img_size - 4) * (img_size - 4), out_features=10
            ),
        )

    def forward(self, x):
        return self.layers(x)


def loss_fn(logits, labels):
    # logits: (N, 10)
    # labels: (N, 1)
    return nn.functional.cross_entropy(logits, labels)


def mean_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean()


def train_model(seed: int, max_n_steps: int | None = None) -> dict:
    device = "cuda"
    model = CNNModel()
    model.to(device, non_blocking=True)

    torch.manual_seed(seed)
    train_ds, test_ds = load_mnist_train_and_test()
    batch_isze = 1024
    loader = DataLoader(
        train_ds, batch_size=batch_isze, shuffle=True, num_workers=10, pin_memory=True
    )
    val_imgs = test_ds["image"].to(device, non_blocking=True)
    val_labels = test_ds["label"].to(device, non_blocking=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses_idxs = []
    losses = []

    train_accs = []
    train_accs_idxs = []

    val_accs = []
    val_accs_idxs = []

    for i, batch in enumerate(tqdm.tqdm(loader)):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach())
        losses_idxs.append(i)
        train_accs.append(mean_accuracy(logits, labels).detach())
        train_accs_idxs.append(i)

        if i % (len(loader) // 10) == 0 or i == len(loader) - 1:
            with torch.no_grad():
                val_logits = model(val_imgs)
                val_acc = mean_accuracy(val_logits, val_labels)
                val_accs.append(val_acc.detach())
                val_accs_idxs.append(i)

        if max_n_steps is not None and i >= max_n_steps:
            print("Stopping training early")
            break

    return {
        "train_accs": [acc.item() for acc in train_accs],
        "train_accs_idxs": train_accs_idxs,
        "val_accs": [acc.item() for acc in val_accs],
        "val_accs_idxs": val_accs_idxs,
        "train_losses": [loss.item() for loss in losses],
        "train_losses_idxs": losses_idxs,
    }


def load_mnist_train_and_test() -> tuple[Dataset, Dataset]:
    ds = load_dataset("mnist")
    train_ds = ds["train"].with_transform(transform=transform)
    test_ds = ds["test"].with_transform(transform=transform)
    return train_ds, test_ds


def transform(d: dict[str, Any]) -> dict[str, Any]:
    if "image" in d:
        d["image"] = torch.cat([to_tensor(img) for img in d["image"]])  # (N, W, H)
        d["image"] = d["image"][:, None, :, :]  # (N, 1, W, H)
    if "label" in d:
        d["label"] = torch.tensor(d["label"])
    return d


def joint_df(results: dict) -> pd.DataFrame:
    train_accs_df = pd.DataFrame(
        {
            "train_accs": results["train_accs"],
            "step": results["train_accs_idxs"],
        }
    )
    val_accs_df = pd.DataFrame(
        {
            "val_accs": results["val_accs"],
            "step": results["val_accs_idxs"],
        }
    )
    train_losses_df = pd.DataFrame(
        {
            "train_losses": results["train_losses"],
            "step": results["train_losses_idxs"],
        }
    )

    df = train_accs_df.merge(val_accs_df, on="step", how="outer").merge(
        train_losses_df, on="step", how="outer"
    )
    return df


def plot_loss_and_accuracy(df: pd.DataFrame):
    # plot train loss and train/val accuracy side by side
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(df["step"], df["train_losses"], label="train loss", alpha=0.5)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(df["step"], df["train_accs"], label="train accuracy", alpha=0.5)
    plt.scatter(df["step"], df["val_accs"], label="val accuracy")
    plt.legend()
    plt.show()


def train_and_save(seed: int):
    print("Starting training for seed: ", seed)
    results = train_model(seed)
    df = joint_df(results)
    print("Final train accuracy: %.4f" % df["train_accs"].iloc[-1])
    print(
        "Final val accuracy: %.4f" % df.query("val_accs.notna()")["val_accs"].iloc[-1]
    )
    df = df.assign(seed=seed)
    filname = "data/mnist_results.csv"
    df.to_csv(filname, mode="a", index=False, header=not os.path.exists(filname))


def plot_many_train_and_val_accuracy(df: pd.DataFrame):
    data = df.groupby("step").agg(["mean", "std"])
    data_nonna = data[data["val_accs"]["mean"].notna()]

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(data.index, data["train_losses"]["mean"], label="train loss mean")
    plt.fill_between(
        data.index,
        data["train_losses"]["mean"] - data["train_losses"]["std"],
        data["train_losses"]["mean"] + data["train_losses"]["std"],
        alpha=0.5,
    )
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(
        data_nonna.index,
        data_nonna["train_accs"]["mean"],
        label="train accuracy mean",
        marker="o",
    )
    plt.fill_between(
        data_nonna.index,
        data_nonna["train_accs"]["mean"] - data_nonna["train_accs"]["std"],
        data_nonna["train_accs"]["mean"] + data_nonna["train_accs"]["std"],
        alpha=0.5,
    )
    plt.plot(
        data_nonna.index,
        data_nonna["val_accs"]["mean"],
        label="val accuracy mean",
        marker="x",
    )
    plt.fill_between(
        data_nonna.index,
        data_nonna["val_accs"]["mean"] - data_nonna["val_accs"]["std"],
        data_nonna["val_accs"]["mean"] + data_nonna["val_accs"]["std"],
        alpha=0.5,
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
