"""
Training script for Custom TTS Model.
Supports mixed precision, distributed training, gradient accumulation, and low-end hardware optimization.
Optimized for CPU, low-end GPUs (2GB VRAM), mid-range, and high-end GPUs.
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml
import numpy as np

from models.tts_model import create_model, get_hardware_optimized_config
from utils.logger import TrainingLogger
from utils.loss import TTSLoss


class TTSDataset(Dataset):
    """Dataset for TTS training."""
    
    def __init__(self, metadata_path: str, feature_dir: str):
        with open(metadata_path, 'r') as f:
            self.samples = json.load(f)
        self.feature_dir = feature_dir
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        feature_path = sample['feature_path']
        
        # Load features
        features = np.load(feature_path, allow_pickle=True).item()
        
        text_sequence = torch.tensor(features['text_sequence'], dtype=torch.long)
        mel_spec = torch.tensor(features['mel_spec'], dtype=torch.float32)
        pitch = torch.tensor(features['pitch'], dtype=torch.float32)
        energy = torch.tensor(features['energy'], dtype=torch.float32)
        
        # Calculate durations from mel spectrogram and text lengths
        text_len = len(text_sequence)
        mel_len = len(mel_spec)
        
        # Simple duration calculation (can be improved with forced alignment)
        avg_duration = mel_len / text_len
        durations = torch.full((text_len,), avg_duration)
        
        return {
            'text': text_sequence,
            'mel_spec': mel_spec,
            'pitch': pitch,
            'energy': energy,
            'durations': durations,
            'text_length': text_len,
            'mel_length': mel_len
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    # Find max lengths
    max_text_len = max(item['text_length'] for item in batch)
    max_mel_len = max(item['mel_length'] for item in batch)
    
    # Initialize tensors
    batch_size = len(batch)
    text_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    mel_padded = torch.zeros(batch_size, max_mel_len, batch[0]['mel_spec'].size(-1))
    pitch_padded = torch.zeros(batch_size, max_mel_len)
    energy_padded = torch.zeros(batch_size, max_mel_len)
    durations_padded = torch.zeros(batch_size, max_text_len, dtype=torch.float32)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)
    mel_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        text_len = item['text_length']
        mel_len = item['mel_length']
        
        text_padded[i, :text_len] = item['text']
        mel_padded[i, :mel_len] = item['mel_spec']
        pitch_padded[i, :mel_len] = item['pitch']
        energy_padded[i, :mel_len] = item['energy']
        durations_padded[i, :text_len] = item['durations'].float()
        text_lengths[i] = text_len
        mel_lengths[i] = mel_len
    
    return {
        'text': text_padded,
        'mel_spec': mel_padded,
        'pitch': pitch_padded,
        'energy': energy_padded,
        'durations': durations_padded,
        'text_lengths': text_lengths,
        'mel_lengths': mel_lengths
    }


class NoamLR:
    """Noam learning rate scheduler."""
    
    def __init__(self, optimizer, model_size, warmup_steps):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.model_size ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def train_epoch(model, dataloader, criterion, optimizer, scheduler, 
                device, epoch, config, scaler, logger):
    """Train for one epoch with gradient accumulation support for low-end hardware."""
    model.train()
    total_loss = 0.0
    total_mel_loss = 0.0
    total_dur_loss = 0.0
    total_pitch_loss = 0.0
    total_energy_loss = 0.0
    
    # Gradient accumulation settings
    accum_steps = config['training'].get('gradient_accumulation_steps', 1)
    effective_batch_size = config['training']['batch_size'] * accum_steps
    print(f"Using gradient accumulation: {accum_steps} steps (effective batch size: {effective_batch_size})")
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        text = batch['text'].to(device)
        mel_spec = batch['mel_spec'].to(device)
        pitch = batch['pitch'].to(device)
        energy = batch['energy'].to(device)
        durations = batch['durations'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config['training']['mixed_precision'] and device.type == 'cuda'):
            outputs = model(
                text, text_lengths,
                mel_spec, pitch, energy, durations
            )
            
            # Calculate losses
            loss_dict = criterion(outputs)
            # Scale loss by accumulation steps
            scaled_loss = loss_dict['total_loss'] / accum_steps
        
        # Backward pass with gradient scaling
        scaler.scale(scaled_loss).backward()
        
        # Only update weights after accum_steps batches
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['clip_grad_norm']
            )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Zero gradients after update
            optimizer.zero_grad()
            
            # Learning rate schedule
            if scheduler is not None:
                scheduler.step()
        
        # Update statistics
        total_loss += loss_dict['total_loss'].item()
        total_mel_loss += loss_dict.get('mel_loss', 0.0)
        total_dur_loss += loss_dict.get('duration_loss', 0.0)
        total_pitch_loss += loss_dict.get('pitch_loss', 0.0)
        total_energy_loss += loss_dict.get('energy_loss', 0.0)
        
        # Update progress bar
        if batch_idx % config['logging']['log_interval'] == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'mel': f'{total_mel_loss/(batch_idx+1):.4f}',
                'dur': f'{total_dur_loss/(batch_idx+1):.4f}'
            })
            
            # Log to tensorboard
            global_step = epoch * len(dataloader) + batch_idx
            logger.log_scalar('train/total_loss', avg_loss, global_step)
            logger.log_scalar('train/mel_loss', total_mel_loss/(batch_idx+1), global_step)
            logger.log_scalar('train/duration_loss', total_dur_loss/(batch_idx+1), global_step)
    
    # Return average losses
    num_batches = len(dataloader)
    return {
        'total_loss': total_loss / num_batches,
        'mel_loss': total_mel_loss / num_batches,
        'duration_loss': total_dur_loss / num_batches,
        'pitch_loss': total_pitch_loss / num_batches,
        'energy_loss': total_energy_loss / num_batches
    }


def validate(model, dataloader, criterion, device, epoch, config, logger):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mel_loss = 0.0
    total_dur_loss = 0.0
    total_pitch_loss = 0.0
    total_energy_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move data to device
            text = batch['text'].to(device)
            mel_spec = batch['mel_spec'].to(device)
            pitch = batch['pitch'].to(device)
            energy = batch['energy'].to(device)
            durations = batch['durations'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            
            # Forward pass
            outputs = model(
                text, text_lengths,
                mel_spec, pitch, energy, durations
            )
            
            # Calculate losses
            loss_dict = criterion(outputs)
            
            total_loss += loss_dict['total_loss'].item()
            total_mel_loss += loss_dict.get('mel_loss', 0.0)
            total_dur_loss += loss_dict.get('duration_loss', 0.0)
            total_pitch_loss += loss_dict.get('pitch_loss', 0.0)
            total_energy_loss += loss_dict.get('energy_loss', 0.0)
    
    # Return average losses
    num_batches = len(dataloader)
    metrics = {
        'total_loss': total_loss / num_batches,
        'mel_loss': total_mel_loss / num_batches,
        'duration_loss': total_dur_loss / num_batches,
        'pitch_loss': total_pitch_loss / num_batches,
        'energy_loss': total_energy_loss / num_batches
    }
    
    # Log to tensorboard
    global_step = epoch * len(dataloader)
    logger.log_scalar('val/total_loss', metrics['total_loss'], global_step)
    logger.log_scalar('val/mel_loss', metrics['mel_loss'], global_step)
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, filename='checkpoint.pt'):
    """Save model checkpoint."""
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.optimizer.state_dict()
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.optimizer.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint.get('loss', float('inf'))
    
    return model, optimizer, scheduler, start_epoch, best_loss


def main():
    parser = argparse.ArgumentParser(description='Train Custom TTS Model')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Path to processed dataset')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--hardware', type=str, default='auto',
                       choices=['cpu', 'low_end_gpu', 'mid_gpu', 'high_gpu', 'auto'],
                       help='Hardware optimization preset (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Load base configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply hardware-optimized configuration if requested
    if args.hardware != 'manual':
        hw_config = get_hardware_optimized_config(args.hardware)
        # Merge hardware config with base config
        config['model'].update(hw_config['model'])
        config['training']['batch_size'] = hw_config['training']['batch_size']
        config['training']['gradient_accumulation_steps'] = hw_config['training']['gradient_accumulation_steps']
        config['training']['mixed_precision'] = hw_config['training']['mixed_precision']
        print(f"Applied {args.hardware} hardware optimization")
        print(f"  - Batch size: {config['training']['batch_size']}")
        print(f"  - Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
        print(f"  - Mixed precision: {config['training']['mixed_precision']}")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    elif config['device']['cuda'] and torch.cuda.is_available():
        device = torch.device(f'cuda:{config["device"]["gpu_ids"][0]}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = TTSDataset(
        os.path.join(args.data_dir, 'train_metadata.json'),
        os.path.join(args.data_dir, 'features')
    )
    val_dataset = TTSDataset(
        os.path.join(args.data_dir, 'val_metadata.json'),
        os.path.join(args.data_dir, 'features')
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['device']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['device']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = create_model(config).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function
    criterion = TTSLoss(config)
    
    # Create optimizer
    if config['optimizer']['type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=tuple(config['optimizer']['betas']),
            eps=config['optimizer']['eps'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=tuple(config['optimizer']['betas']),
            eps=config['optimizer']['eps']
        )
    
    # Create learning rate scheduler
    scheduler = None
    if config['training']['lr_scheduler'] == 'noam':
        scheduler = NoamLR(
            optimizer,
            model_size=config['model']['text_encoder']['embedding_dim'],
            warmup_steps=config['training']['warmup_steps']
        )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Create logger
    logger = TrainingLogger(config['logging']['tensorboard_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model, optimizer, scheduler, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        print(f"Resumed from epoch {start_epoch} with best loss {best_val_loss:.4f}")
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, config, scaler, logger
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, config, logger
        )
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics['total_loss'],
                config, 'best_model.pt'
            )
            print(f"Saved new best model with loss {best_val_loss:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics['total_loss'],
                config, f'checkpoint_epoch_{epoch + 1}.pt'
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, config['training']['num_epochs'],
        val_metrics['total_loss'], config, 'final_model.pt'
    )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
