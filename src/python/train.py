"""Excerpt from the training pipeline (mlp_train.py) - the training loop.
Author: Hadi Sarhangi Fard

Standard PyTorch training on MNIST with the Adam optimizer and cross-entropy
loss. What makes this loop specific to the visualizer is that it does not
only save one final checkpoint: it calls record_snapshot() at a list of
"milestones" (measured in images seen) so the app's training timeline can
scrub through the network's whole learning history, not just its end state.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallMLP

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Accuracy on a held-out loader (used for the timeline metrics)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def train(hidden_dims=(1024, 784, 484), batch_size=128, lr=1e-3, epochs=70, device="cpu"):
    device = torch.device(device)
    model = SmallMLP(28 * 28, hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),  # same normalization used at inference time
    ])
    train_loader = DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("data", train=False, download=True, transform=transform),
        batch_size=512, shuffle=False,
    )

    images_seen = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_images = 0.0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()

            images_seen += data.size(0)
            epoch_loss += loss.item() * data.size(0)
            epoch_images += data.size(0)

            # In the full script, advance_milestones() checks `images_seen`
            # here on every batch and calls record_snapshot() (see export.py)
            # whenever a timeline threshold has just been crossed.

        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} - avg loss: {epoch_loss / epoch_images:.4f} - test accuracy: {accuracy * 100:.2f}%")

    return model
