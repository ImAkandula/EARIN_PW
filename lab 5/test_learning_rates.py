# test_learning_rates.py

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import MLP
from train_utils import train, validate, save_experiment_results

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 64
    hidden_layers = [128, 64]
    learning_rates = [0.1, 0.01, 0.001]
    num_epochs = 10

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = MLP(hidden_layers=hidden_layers).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        config = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "hidden_layers": hidden_layers,
            "activation": "ReLU",
            "loss_function": "CrossEntropyLoss",
        }

        # Save results
        save_experiment_results(config, train_losses, val_losses, train_accuracies, val_accuracies, f"learning_rate_{lr}.txt")

        # Save plots
        os.makedirs("plots", exist_ok=True)

        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f"Learning Rate = {lr} - Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"plots/learning_rate_{lr}_loss.png")
        plt.close()

        plt.figure()
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title(f"Learning Rate = {lr} - Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f"plots/learning_rate_{lr}_accuracy.png")
        plt.close()
