import os
import torch
import subprocess
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from dataset import Vocabulary
from model import Transformer
from torch.cuda.amp import autocast, GradScaler

def compute_val_bleu(model, source_vocab, target_vocab, device, val_de_path='data/val.de-en.de', val_en_path='data/val.de-en.en'):
    model.eval()
    temp_pred_path1 = 'temp_val1.en'
    temp_pred_path2 = 'temp_val2.en'
    with open(val_de_path, 'r', encoding='utf-8') as f:
        val_sentences = [line.strip() for line in f]
    
    translations = []
    for sent in tqdm(val_sentences[::2], desc='Translating val', leave=False):
        en_translation = translate(
            model,
            sent,
            source_vocab,
            target_vocab,
            device, 
            beam_size = 3
        )
        translations.append(en_translation)

    with open(temp_pred_path1, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')

    with open(val_en_path, 'r', encoding='utf-8') as f:
        val_en_sentences = [line.strip() for line in f]

    with open(temp_pred_path2, 'w', encoding='utf-8') as f:
        for sent in val_en_sentences[::2]:
            f.write(sent + '\n')

    with open(temp_pred_path1, 'r', encoding='utf-8') as pred_file:
        result = subprocess.run(
            ['sacrebleu', temp_pred_path2, '--tokenize', 'none', '--width', '2', '-b'],
            stdin=pred_file,
            capture_output=True,
            text=True,
            check=True
        )

    bleu = float(result.stdout.strip())
    os.remove(temp_pred_path1)
    os.remove(temp_pred_path2)
    return bleu


def plot_losses(train_losses, val_losses, val_bleus):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label='train')
    axes[0].plot(epochs, val_losses, label='val')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(range(1, len(val_bleus) + 1), val_bleus, label='val BLEU')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('BLEU')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
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

        with autocast(dtype=torch.bfloat16):
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
        with autocast(dtype=torch.bfloat16):
            logits = model(source, target_input, source_pad_mask, target_pad_mask)

            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_output.reshape(-1)
            loss = criterion(logits_flat, target_flat)

        epoch_loss += loss.item() * source.size(0)
    
    return epoch_loss / len(loader.dataset)

def train(model, optimizer, train_loader, val_loader, num_epochs, device, pad_ind, source_vocab, target_vocab, save_path = './best_model.pt'):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_ind, label_smoothing=0.1).to(device)
    train_losses = []
    val_losses = []
    val_bleus = []
    best_bleu = -float('inf')
    scaler = GradScaler()

    for epochs in tqdm(range(1, num_epochs + 1), desc='Epoch'):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device, pad_ind, scaler)
        train_losses.append(train_loss)
        val_loss = val_epoch(model, optimizer, criterion, val_loader, device, pad_ind)
        val_losses.append(val_loss)
        val_bleu = compute_val_bleu(model, source_vocab, target_vocab, device)
        val_bleus.append(val_bleu)

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bleu': val_bleu,
            }, save_path)

        plot_losses(train_losses, val_losses, val_bleus)

@torch.no_grad()
def translate(model: Transformer, source_sentence, source_vocab, target_vocab: Vocabulary, device, max_len=82, beam_size=3):
    model.eval()

    source_tokens = source_vocab.encode(source_sentence)
    source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device)

    beams = [([target_vocab.bos_ind], 0.0)]

    forbidden_tokens = [
        target_vocab.unk_ind,
        target_vocab.pad_ind,
        target_vocab.bos_ind,
    ]

    def sort_func(x):
        return x[1]

    for _ in range(max_len):
        all_candidates = []

        for translation, score in beams:
            if translation[-1] == target_vocab.eos_ind:
                all_candidates.append((translation, score))
                continue

            target_tensor = torch.tensor(translation).unsqueeze(0).to(device)

            logits = model(source_tensor, target_tensor, None, None)
            next_logits = logits[0, -1, :].clone()
            next_logits[forbidden_tokens] = -float('inf')

            log_probs = torch.log_softmax(next_logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            for log_prob, token_id in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                new_translation = translation + [token_id]
                new_score = score + log_prob
                all_candidates.append((new_translation, new_score))

        all_candidates.sort(key=sort_func, reverse=True)
        beams = all_candidates[:beam_size]

        if all(translation[-1] == target_vocab.eos_ind for translation, _ in beams):
            break

    best_translation, _ = max(beams, key=sort_func)

    result = best_translation[1:]
    if target_vocab.eos_ind in result:
        result = result[:result.index(target_vocab.eos_ind)]

    return target_vocab.decode(result)