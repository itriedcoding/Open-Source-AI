"""
Training Script for Custom TTS Model
Supports CPU, low-end GPU, and high-end GPU with automatic optimization
Includes gradient accumulation, mixed precision, and checkpointing
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import yaml
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models import CustomTTSModel, get_hardware_config


class TTSDataset(Dataset):
    """Dataset for TTS training"""
    
    def __init__(self, metadata_file, processed_dir):
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.processed_dir = Path(processed_dir)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        feature_path = self.processed_dir / item['feature_file']
        
        data = torch.load(feature_path, map_location='cpu')
        
        return {
            'tokens': data['tokens'],
            'mel': data['mel'],
            'pitch': data['pitch'],
            'energy': data['energy'],
            'duration': data['duration']
        }


def collate_fn(batch):
    """Collate function for batching with padding"""
    max_token_len = max(len(item['tokens']) for item in batch)
    max_mel_len = max(item['mel'].shape[0] for item in batch)
    
    tokens_padded = []
    mel_padded = []
    pitch_list = []
    energy_list = []
    durations_list = []
    token_lengths = []
    mel_lengths = []
    
    for item in batch:
        # Pad tokens
        tokens = item['tokens']
        padding = torch.zeros(max_token_len - len(tokens), dtype=tokens.dtype)
        tokens_padded.append(torch.cat([tokens, padding]))
        token_lengths.append(len(tokens))
        
        # Pad mel spectrogram
        mel = item['mel']
        mel_pad = torch.zeros(max_mel_len - mel.shape[0], mel.shape[1])
        mel_padded.append(torch.cat([mel, mel_pad], dim=0))
        mel_lengths.append(mel.shape[0])
        
        # Pad pitch and energy
        pitch = item['pitch']
        energy = item['energy']
        min_len = min(len(pitch), len(energy), mel.shape[0])
        
        pitch_list.append(pitch[:min_len])
        energy_list.append(energy[:min_len])
        durations_list.append(torch.ones(min_len))  # Simplified duration
    
    return {
        'tokens': torch.stack(tokens_padded),
        'mel': torch.stack(mel_padded),
        'pitch': torch.stack(pitch_list),
        'energy': torch.stack(energy_list),
        'durations': torch.stack(durations_list),
        'token_lengths': torch.tensor(token_lengths),
        'mel_lengths': torch.tensor(mel_lengths)
    }


class NoamScheduler:
    """Noam learning rate scheduler"""
    
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
        
        return lr


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, 
                grad_scaler, config, epoch):
    """Single training epoch"""
    model.train()
    total_loss = 0
    total_mel_loss = 0
    total_duration_loss = 0
    
    training_cfg = config.get('training', {})
    use_amp = training_cfg.get('use_mixed_precision', True)
    accum_steps = training_cfg.get('gradient_accumulation_steps', 1)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        tokens = batch['tokens'].to(device)
        mel_targets = batch['mel'].to(device)
        token_lengths = batch['token_lengths'].to(device)
        mel_lengths = batch['mel_lengths'].to(device)
        pitches = batch['pitch'].to(device)
        energies = batch['energy'].to(device)
        durations = batch['durations'].to(device)
        
        # Forward pass with mixed precision
        if use_amp and device.type == 'cuda':
            with autocast():
                output = model(
                    tokens, token_lengths, mel_targets, mel_lengths,
                    pitches=pitches, energies=energies, durations=durations
                )
                
                # Calculate losses
                mel_loss = criterion['mel'](
                    output['mel_output'][~output['mel_mask']], 
                    mel_targets[~output['mel_mask']]
                )
                
                duration_loss = criterion['duration'](
                    output['duration_pred'], 
                    durations
                )
                
                loss = mel_loss + 0.1 * duration_loss
                loss = loss / accum_steps
            
            # Backward pass with gradient scaling
            grad_scaler.scale(loss).backward()
            
            # Update weights every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        else:
            # Standard training (CPU or no AMP)
            output = model(
                tokens, token_lengths, mel_targets, mel_lengths,
                pitches=pitches, energies=energies, durations=durations
            )
            
            # Calculate losses
            mel_loss = criterion['mel'](
                output['mel_output'][~output['mel_mask']], 
                mel_targets[~output['mel_mask']]
            )
            
            duration_loss = criterion['duration'](
                output['duration_pred'], 
                durations
            )
            
            loss = mel_loss + 0.1 * duration_loss
            loss = loss / accum_steps
            
            loss.backward()
            
            # Update weights every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        
        # Track metrics
        total_loss += loss.item() * accum_steps
        total_mel_loss += mel_loss.item()
        total_duration_loss += duration_loss.item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'mel': f'{total_mel_loss / (batch_idx + 1):.4f}',
                'dur': f'{total_duration_loss / (batch_idx + 1):.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_mel_loss = total_mel_loss / len(dataloader)
    avg_duration_loss = total_duration_loss / len(dataloader)
    
    return avg_loss, avg_mel_loss, avg_duration_loss


def validate(model, dataloader, criterion, device, config):
    """Validation epoch"""
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    
    training_cfg = config.get('training', {})
    use_amp = training_cfg.get('use_mixed_precision', True)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            tokens = batch['tokens'].to(device)
            mel_targets = batch['mel'].to(device)
            token_lengths = batch['token_lengths'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            pitches = batch['pitch'].to(device)
            energies = batch['energy'].to(device)
            durations = batch['durations'].to(device)
            
            if use_amp and device.type == 'cuda':
                with autocast():
                    output = model(
                        tokens, token_lengths, mel_targets, mel_lengths,
                        pitches=pitches, energies=energies, durations=durations
                    )
                    
                    mel_loss = criterion['mel'](
                        output['mel_output'][~output['mel_mask']], 
                        mel_targets[~output['mel_mask']]
                    )
                    
                    duration_loss = criterion['duration'](
                        output['duration_pred'], 
                        durations
                    )
                    
                    loss = mel_loss + 0.1 * duration_loss
            else:
                output = model(
                    tokens, token_lengths, mel_targets, mel_lengths,
                    pitches=pitches, energies=energies, durations=durations
                )
                
                mel_loss = criterion['mel'](
                    output['mel_output'][~output['mel_mask']], 
                    mel_targets[~output['mel_mask']]
                )
                
                duration_loss = criterion['duration'](
                    output['duration_pred'], 
                    durations
                )
                
                loss = mel_loss + 0.1 * duration_loss
            
            total_loss += loss.item()
            total_mel_loss += mel_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mel_loss = total_mel_loss / len(dataloader)
    
    return avg_loss, avg_mel_loss


def main():
    parser = argparse.ArgumentParser(description='Train Custom TTS Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--hardware', type=str, default='auto',
                       choices=['auto', 'cpu', 'low_end_gpu', 'mid_gpu', 'high_gpu'],
                       help='Hardware type for optimization')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get hardware-specific configuration
    hw_config = get_hardware_config(args.hardware)
    
    # Merge hardware config with base config
    if 'model' in hw_config:
        config['model'].update(hw_config['model'])
    if 'training' in hw_config:
        config['training'].update(hw_config['training'])
    
    print(f"\n🔧 Hardware Configuration: {args.hardware}")
    print(f"   Model hidden dim: {config['model']['hidden_dim']}")
    print(f"   Encoder layers: {config['model']['encoder_layers']}")
    print(f"   Decoder layers: {config['model']['decoder_layers']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"   Mixed precision: {config['training']['use_mixed_precision']}")
    
    # Setup device
    if args.hardware == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("\n💻 Using CPU for training")
    else:
        device = torch.device('cuda')
        print(f"\n🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create checkpoint directory
    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary size from processed data
    vocab_file = Path(args.processed_dir) / 'vocab.json'
    if vocab_file.exists():
        with open(vocab_file, 'r') as f:
            vocab_info = json.load(f)
        config['vocab_size'] = vocab_info['vocab_size']
        print(f"\n📚 Vocabulary size: {config['vocab_size']}")
    else:
        print("\n⚠️  Warning: vocab.json not found. Using default vocab_size=100")
        config['vocab_size'] = 100
    
    # Initialize model
    model = CustomTTSModel(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Initialize datasets
    processed_path = Path(args.processed_dir)
    train_dataset = TTSDataset(processed_path / 'train_metadata.json', processed_path)
    val_dataset = TTSDataset(processed_path / 'val_metadata.json', processed_path)
    
    print(f"\n📁 Dataset sizes:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to >0 for faster loading on Linux
        pin_memory=device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    # Initialize loss functions
    criterion = {
        'mel': nn.MSELoss(),
        'duration': nn.MSELoss()
    }
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    scheduler = NoamScheduler(
        optimizer,
        model_size=config['model']['hidden_dim'],
        warmup_steps=config['training']['warmup_steps']
    )
    
    # Initialize gradient scaler for mixed precision
    grad_scaler = GradScaler() if config['training']['use_mixed_precision'] else None
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume and Path(args.resume).exists():
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   Resumed from epoch {start_epoch-1}")
    
    # Training loop
    num_epochs = args.epochs or config['training']['epochs']
    save_every = config['training'].get('save_every_epochs', 5)
    
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    print(f"   Checkpoints saved every {save_every} epochs")
    print(f"   Output directory: {checkpoint_path}")
    
    for epoch in range(start_epoch, num_epochs + 1):
        # Training
        train_loss, train_mel_loss, train_dur_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, grad_scaler, config, epoch
        )
        
        # Validation
        val_loss, val_mel_loss = validate(model, val_loader, criterion, device, config)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        print(f"Train Loss: {train_loss:.4f} (Mel: {train_mel_loss:.4f}, Dur: {train_dur_loss:.4f})")
        print(f"Val Loss:   {val_loss:.4f} (Mel: {val_mel_loss:.4f})")
        print(f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if epoch % save_every == 0 or val_loss < best_val_loss:
            checkpoint_file = checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': val_loss,
                'config': config
            }, checkpoint_file)
            
            print(f"✅ Checkpoint saved: {checkpoint_file}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_file = checkpoint_path / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': val_loss,
                    'config': config
                }, best_model_file)
                print(f"🏆 New best model saved: {best_model_file}")
        
        print()
    
    print("\n🎉 Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final checkpoint: {checkpoint_path / f'checkpoint_epoch_{num_epochs}.pt'}")
    print(f"Best model: {checkpoint_path / 'best_model.pt'}")


if __name__ == '__main__':
    main()
