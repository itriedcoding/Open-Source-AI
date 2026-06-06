"""
Loss functions for Custom TTS Model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TTSLoss(nn.Module):
    """Combined loss function for TTS training."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Mel spectrogram reconstruction loss
        self.mel_criterion = nn.L1Loss()
        
        # Duration prediction loss
        self.duration_criterion = nn.MSELoss()
        
        # Pitch prediction loss
        self.pitch_criterion = nn.MSELoss()
        
        # Energy prediction loss
        self.energy_criterion = nn.MSELoss()
        
        # Loss weights
        self.mel_weight = 1.0
        self.duration_weight = 1.0
        self.pitch_weight = 1.0
        self.energy_weight = 1.0
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate total loss and individual component losses.
        
        Args:
            outputs: Dictionary containing model predictions and targets
        
        Returns:
            Dictionary with total_loss and individual losses
        """
        # Extract predictions and targets
        mel_pred = outputs['mel_output']
        dur_pred = outputs['pred_durations']
        pitch_pred = outputs['pred_pitch']
        energy_pred = outputs['pred_energy']
        
        mel_target = outputs.get('mel_targets')
        dur_target = outputs.get('durations')
        pitch_target = outputs.get('pitch_targets')
        energy_target = outputs.get('energy_targets')
        
        # Initialize losses
        mel_loss = torch.tensor(0.0, device=mel_pred.device)
        duration_loss = torch.tensor(0.0, device=mel_pred.device)
        pitch_loss = torch.tensor(0.0, device=mel_pred.device)
        energy_loss = torch.tensor(0.0, device=mel_pred.device)
        
        # Calculate mel spectrogram loss
        if mel_target is not None:
            # Mask padding regions
            mel_lengths = outputs.get('mel_lengths')
            if mel_lengths is not None:
                max_len = mel_target.size(1)
                mask = torch.arange(max_len, device=mel_target.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).expand_as(mel_target)
                
                mel_pred_masked = mel_pred[mask]
                mel_target_masked = mel_target[mask]
                
                mel_loss = self.mel_criterion(mel_pred_masked, mel_target_masked)
            else:
                mel_loss = self.mel_criterion(mel_pred, mel_target)
        
        # Calculate duration loss
        if dur_target is not None:
            text_lengths = outputs.get('text_lengths')
            if text_lengths is not None:
                max_len = dur_target.size(1)
                mask = torch.arange(max_len, device=dur_target.device).unsqueeze(0) < text_lengths.unsqueeze(1)
                
                dur_pred_masked = dur_pred[mask]
                dur_target_masked = dur_target[mask]
                
                duration_loss = self.duration_criterion(dur_pred_masked, dur_target_masked)
            else:
                duration_loss = self.duration_criterion(dur_pred, dur_target)
        
        # Calculate pitch loss
        if pitch_target is not None:
            mel_lengths = outputs.get('mel_lengths')
            if mel_lengths is not None:
                max_len = pitch_target.size(1)
                mask = torch.arange(max_len, device=pitch_target.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
                
                pitch_pred_masked = pitch_pred[mask]
                pitch_target_masked = pitch_target[mask]
                
                pitch_loss = self.pitch_criterion(pitch_pred_masked, pitch_target_masked)
            else:
                pitch_loss = self.pitch_criterion(pitch_pred, pitch_target)
        
        # Calculate energy loss
        if energy_target is not None:
            mel_lengths = outputs.get('mel_lengths')
            if mel_lengths is not None:
                max_len = energy_target.size(1)
                mask = torch.arange(max_len, device=energy_target.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
                
                energy_pred_masked = energy_pred[mask]
                energy_target_masked = energy_target[mask]
                
                energy_loss = self.energy_criterion(energy_pred_masked, energy_target_masked)
            else:
                energy_loss = self.energy_criterion(energy_pred, energy_target)
        
        # Calculate total weighted loss
        total_loss = (
            self.mel_weight * mel_loss +
            self.duration_weight * duration_loss +
            self.pitch_weight * pitch_loss +
            self.energy_weight * energy_loss
        )
        
        return {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'duration_loss': duration_loss,
            'pitch_loss': pitch_loss,
            'energy_loss': energy_loss
        }
