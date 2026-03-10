import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label = 'train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label = 'val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

def train_epoch(model, optimizer, criterion, loader, device, pad_ind):
    model.train()
    epoch_loss = 0.0

    for source, target, source_len, target_len in tqdm(loader, desc='train', leave=True):
        source = source.to(device)
        target = target.to(device)
        source_len = source_len.to(device)
        target_len = target_len.to(device)

        target_input = target[:, :-1]
        target_output = target[:, 1:]

        source_pad_mask = source == pad_ind
        target_pad_mask = target_input == pad_ind
        optimizer.zero_grad()
        logits = model(source, target_input, source_pad_mask, target_pad_mask)

        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = target_output.reshape(-1)

        loss = criterion(logits_flat, target_flat)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * source.size(0)

    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def val_epoch(model, optimizer, criterion, loader, device, pad_ind):
    model.eval()
    epoch_loss = 0.0

    for source, target, source_len, target_len in tqdm(loader, desc='val', leave=True):
        source = source.to(device)
        target = target.to(device)
        source_len = source_len.to(device)
        target_len = target_len.to(device)

        target_input = target[:, :-1]
        target_output = target[:, 1:]
        source_pad_mask = source == pad_ind
        target_pad_mask = target_input == pad_ind

        logits = model(source, target_input, source_pad_mask, target_pad_mask)

        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = target_output.reshape(-1)
        loss = criterion(logits_flat, target_flat)

        epoch_loss += loss.item() * source.size(0)
    
    return epoch_loss / len(loader.dataset)

def train(model, optimizer, train_loader, val_loader, num_epochs, device, pad_ind):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_ind)
    train_losses = []
    val_losses = []

    for epochs in tqdm(range(1, num_epochs + 1), desc='Epoch'):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device, pad_ind)
        train_losses.append(train_loss)
        val_loss = val_epoch(model, optimizer, criterion, train_loader, device, pad_ind)
        val_losses.append(val_loss)

        plot_losses(train_losses, val_losses)