from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from torchvision.transforms import ToTensor, transforms
from datasets import load_dataset, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_float32_matmul_precision("high")

to_tensor = ToTensor()


class CNNForMnist(nn.Module):
    def __init__(self, img_channels: int, img_size: int, n_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=64, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(
                in_features=64 * (img_size - 4) * (img_size - 4),
                out_features=n_classes,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class ImprovedCNNForCifar100(nn.Module):
    def __init__(self, img_channels: int, img_size: int, n_classes: int):
        super().__init__()
        
        def conv_block(in_channels, out_channels, dropout=0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            )

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        # Main blocks with residual connections
        self.block1 = conv_block(64, 128, dropout=0.2)
        self.block2 = conv_block(128, 256, dropout=0.3)
        self.block3 = conv_block(256, 512, dropout=0.4)
        self.block4 = conv_block(512, 1024, dropout=0.4)

        # Residual connections
        self.res1 = nn.Conv2d(64, 128, kernel_size=1)
        self.res2 = nn.Conv2d(128, 256, kernel_size=1)
        self.res3 = nn.Conv2d(256, 512, kernel_size=1)
        self.res4 = nn.Conv2d(512, 1024, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Enhanced classifier (keeping bias for linear layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        # Initial convolution
        x = self.initial(x)

        # Block 1 with residual
        identity = self.res1(nn.functional.avg_pool2d(x, 2))
        x = self.block1(x)
        x = x + identity

        # Block 2 with residual
        identity = self.res2(nn.functional.avg_pool2d(x, 2))
        x = self.block2(x)
        x = x + identity

        # Block 3 with residual
        identity = self.res3(nn.functional.avg_pool2d(x, 2))
        x = self.block3(x)
        x = x + identity

        # Block 4 with residual
        identity = self.res4(nn.functional.avg_pool2d(x, 2))
        x = self.block4(x)
        x = x + identity

        x = self.pool(x)
        x = self.classifier(x)
        return x


def loss_fn(logits, labels):
    # logits: (B, n_classes)
    # labels: (B, 1)
    # output: (1,)
    return nn.functional.cross_entropy(logits, labels)


def mean_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean()


@dataclass
class TrainConfig:
    dataset: str
    train_ds: Dataset
    test_ds: Dataset
    model: nn.Module
    n_epochs: int


def get_train_conf(dataset: str) -> TrainConfig:
    if dataset == "mnist":
        train_ds, test_ds, img_channels, img_size, n_classes = (
            load_mnist_train_and_test()
        )
        model = CNNForMnist(
            img_channels=img_channels, img_size=img_size, n_classes=n_classes
        )
        n_epochs = 2
        return TrainConfig(dataset, train_ds, test_ds, model, n_epochs)

    if dataset == "cifar100":
        train_ds, test_ds, img_channels, img_size, n_classes = (
            load_cifar100_train_and_test()
        )
        model = ImprovedCNNForCifar100(
            img_channels=img_channels, img_size=img_size, n_classes=n_classes
        )
        model = torch.compile(model)
        n_epochs = 50
        return TrainConfig(dataset, train_ds, test_ds, model, n_epochs)

    raise ValueError(f"Unknown dataset: {dataset}")


def train_model(seed: int, conf: TrainConfig, max_n_steps: int | None = None) -> dict:
    print("Starting training for seed: %d" % seed)
    torch.manual_seed(seed)
    device = "cuda"

    model = conf.model
    train_ds = conf.train_ds
    test_ds = conf.test_ds
    n_epochs = conf.n_epochs

    model.to(device, non_blocking=True)

    batch_size = 256
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True
    )
    val_imgs = test_ds["image"].to(device, non_blocking=True)
    val_labels = test_ds["label"].to(device, non_blocking=True)
    weight_decay = 5e-4  # claude suggestion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5, verbose=True
    )

    losses_idxs = []
    losses = []

    train_accs = []
    train_accs_idxs = []

    val_accs = []
    val_accs_idxs = []

    for epoch in range(n_epochs):
        for i, batch in enumerate(tqdm.tqdm(loader)):
            step = i + epoch * len(loader)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach())
            losses_idxs.append(step)
            train_accs.append(mean_accuracy(logits, labels).detach())
            train_accs_idxs.append(step)

            if max_n_steps is not None and i >= max_n_steps:
                print("Stopping training early")
                return

        print("Finished epoch %d, evaluating model" % (epoch + 1))
        val_acc, val_loss = evaluate(model, val_imgs, val_labels)
        val_accs.append(val_acc.detach())
        val_accs_idxs.append(step)
        scheduler.step(val_loss)

    return {
        "train_accs": [acc.item() for acc in train_accs],
        "train_accs_idxs": train_accs_idxs,
        "val_accs": [acc.item() for acc in val_accs],
        "val_accs_idxs": val_accs_idxs,
        "train_losses": [loss.item() for loss in losses],
        "train_losses_idxs": losses_idxs,
    }


@torch.inference_mode()
def evaluate(model, val_imgs, val_labels):
    val_logits = model(val_imgs)
    val_acc = mean_accuracy(val_logits, val_labels)
    val_loss = loss_fn(val_logits, val_labels)
    return val_acc, val_loss


def load_mnist_train_and_test() -> tuple[Dataset, Dataset, int, int, int]:
    print("Loading MNIST")
    ds = load_dataset("mnist")
    train_ds = ds["train"].with_transform(transform=vanilla_transform)
    test_ds = ds["test"].with_transform(transform=vanilla_transform)
    img_channels = 1
    img_size = 28
    n_classes = 10
    return train_ds, test_ds, img_channels, img_size, n_classes


def load_cifar100_train_and_test() -> tuple[Dataset, Dataset, int, int, int]:
    print("Loading CIFAR100")
    ds = load_dataset("uoft-cs/cifar100")
    new_names = {"img": "image", "fine_label": "label"}
    train_ds = (
        ds["train"]
        .rename_columns(new_names)
        .with_transform(transform=data_augmentation_transform)
    )
    test_ds = (
        ds["test"].rename_columns(new_names).with_transform(transform=vanilla_transform)
    )
    img_channels = 3
    img_size = 32
    n_classes = 100
    return train_ds, test_ds, img_channels, img_size, n_classes


data_augmentation = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ),  # CIFAR-100 specific
    ]
)


def data_augmentation_transform(d: dict[str, Any]) -> dict[str, Any]:
    if "image" in d:
        d["image"] = torch.stack(
            [data_augmentation(img) for img in d["image"]]
        )  # (B, C, W, H)
    if "label" in d:
        d["label"] = torch.tensor(d["label"])
    return d


def vanilla_transform(d: dict[str, Any]) -> dict[str, Any]:
    if "image" in d:
        d["image"] = torch.stack([to_tensor(img) for img in d["image"]])  # (B, C, W, H)
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


def plot_loss_and_accuracy(df: pd.DataFrame, alpha: float = 0.5):
    # plot train loss and train/val accuracy side by side
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(df["step"], df["train_losses"], label="train loss", alpha=alpha)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(df["step"], df["train_accs"], label="train accuracy", alpha=alpha)
    plt.scatter(df["step"], df["val_accs"], label="val accuracy")
    plt.legend()
    plt.show()


def train_and_save(seed: int, conf: TrainConfig):
    results = train_model(seed, conf)
    df = joint_df(results)
    print("Final train accuracy: %.4f" % df["train_accs"].iloc[-1])
    print(
        "Final val accuracy: %.4f" % df.query("val_accs.notna()")["val_accs"].iloc[-1]
    )
    df = df.assign(seed=seed)
    filename = f"data/{conf.dataset}_results.csv"
    df.to_csv(filename, mode="a", index=False, header=not os.path.exists(filename))


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
