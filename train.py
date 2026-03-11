import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from dataset import Vocabulary
from model import Transformer
from torch.cuda.amp import autocast, GradScaler

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

def train_epoch(model, optimizer, criterion, loader, device, pad_ind, scaler):
    model.train()
    epoch_loss = 0.0

    for source, target, source_len, target_len in tqdm(loader, desc='train', leave=True):
        source = source.to(device)
        target = target.to(device)
        source_len = source_len.to(device)
        target_len = target_len.to(device)

        target_input = target[:, :-1]
        target_output = target[:, 1:]

        source_pad_mask = (source == pad_ind).to(device)
        target_pad_mask = (target_input == pad_ind).to(device)
        optimizer.zero_grad()

        with autocast():
            logits = model(source, target_input, source_pad_mask, target_pad_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_output.reshape(-1)
            loss = criterion(logits_flat, target_flat)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
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
        source_pad_mask = (source == pad_ind).to(device)
        target_pad_mask = (target_input == pad_ind).to(device)
        with autocast():
            logits = model(source, target_input, source_pad_mask, target_pad_mask)

            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_output.reshape(-1)
            loss = criterion(logits_flat, target_flat)

        epoch_loss += loss.item() * source.size(0)
    
    return epoch_loss / len(loader.dataset)

def train(model, optimizer, train_loader, val_loader, num_epochs, device, pad_ind, save_path = './best_model.pt'):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_ind).to(device)
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    scaler = GradScaler()

    for epochs in tqdm(range(1, num_epochs + 1), desc='Epoch'):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device, pad_ind, scaler)
        train_losses.append(train_loss)
        val_loss = val_epoch(model, optimizer, criterion, val_loader, device, pad_ind)
        val_losses.append(val_loss)

        if epochs >= 5 and val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)

        plot_losses(train_losses, val_losses)

@torch.no_grad()
def translate(model : Transformer, source_sentence, source_vocab, target_vocab : Vocabulary, device, max_len = 82):
    model.eval()
    source_tokens = source_vocab.encode(source_sentence)
    source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device)
    translation = [target_vocab.bos_ind]

    forbidden_tokens = [
        target_vocab.unk_ind,
        target_vocab.pad_ind,
        target_vocab.bos_ind,
    ]

    for _ in range(max_len):
        target_tensor = torch.tensor(translation).unsqueeze(0).to(device)
        logits = model(source_tensor, target_tensor, None, None)
        next_logits = logits[0, -1, :].clone()
        next_logits[forbidden_tokens] = -float('inf')
        next_token = next_logits.argmax().item()
        translation.append(next_token)
        if next_token == target_vocab.eos_ind:
            break

    return target_vocab.decode(translation[1: -1] if translation[-1] == target_vocab.eos_ind else translation[1:])