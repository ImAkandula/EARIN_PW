# train_utils.py

import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(criterion, (nn.MSELoss, nn.L1Loss)):
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
            labels_onehot = labels_onehot.to(device)
            loss = criterion(outputs, labels_onehot)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            images = images.view(images.size(0), -1)
            outputs = model(images)

            if isinstance(criterion, (nn.MSELoss, nn.L1Loss)):
                labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
                labels_onehot = labels_onehot.to(device)
                loss = criterion(outputs, labels_onehot)
            else:
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def save_experiment_results(config, train_losses, val_losses, train_accuracies, val_accuracies, filename):
    os.makedirs("results", exist_ok=True)

    with open(f"results/{filename}", "w") as f:
        f.write("Experiment Configuration:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\nTraining Losses per Epoch:\n")
        f.write(", ".join([f"{loss:.4f}" for loss in train_losses]) + "\n")
        f.write("\nValidation Losses per Epoch:\n")
        f.write(", ".join([f"{loss:.4f}" for loss in val_losses]) + "\n")
        f.write("\nTraining Accuracies per Epoch:\n")
        f.write(", ".join([f"{acc:.4f}" for acc in train_accuracies]) + "\n")
        f.write("\nValidation Accuracies per Epoch:\n")
        f.write(", ".join([f"{acc:.4f}" for acc in val_accuracies]) + "\n")
