"""Small MNIST task shared by the honest and attacker ClientApps."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class Net(nn.Module):
    """A deliberately small model so one workstation can run five clients."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    max_samples: int,
) -> tuple[DataLoader, DataLoader]:
    """Load one deterministic modulo partition of MNIST."""

    root = Path(os.environ.get("DATA_ROOT", "/data"))
    transform = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_indices = list(range(partition_id, len(train), num_partitions))
    test_indices = list(range(partition_id, len(test), num_partitions))
    if max_samples > 0:
        train_indices = train_indices[:max_samples]
        test_indices = test_indices[: max(1, max_samples // 5)]

    return (
        DataLoader(Subset(train, train_indices), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(test, test_indices), batch_size=batch_size, shuffle=False),
    )


def train_model(
    model: Net,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> float:
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    batches = 0
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
    return total_loss / max(1, batches)


def evaluate_model(
    model: Net, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += float(criterion(outputs, labels).item()) * len(labels)
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total += len(labels)
    return total_loss / max(1, total), correct / max(1, total)

