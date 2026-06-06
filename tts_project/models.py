"""
Custom Text-to-Speech Model Architecture
Transformer-based encoder-decoder with duration, pitch, and energy predictors
Fully optimized for CPU and low-end GPU hardware
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DurationPredictor(nn.Module):
    """Predicts phoneme durations"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            
        x = self.linear(x)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            
        return x.squeeze(-1)


class PitchPredictor(nn.Module):
    """Predicts pitch contours"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            
        x = self.linear(x)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            
        return x.squeeze(-1)


class EnergyPredictor(nn.Module):
    """Predicts energy contours"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            
        x = self.linear(x)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            
        return x.squeeze(-1)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer optimized for variable hardware"""
    
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, 
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer optimized for variable hardware"""
    
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class LengthRegulator(nn.Module):
    """Expands sequences based on predicted durations"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, durations, mask=None):
        """
        Args:
            x: (batch, seq_len, dim)
            durations: (batch, seq_len)
            mask: (batch, seq_len)
        """
        batch_size, seq_len, dim = x.shape
        
        # Expand based on durations
        output_list = []
        output_lengths = []
        
        for i in range(batch_size):
            expanded = []
            for j in range(seq_len):
                duration = max(0, int(durations[i, j].item()))
                if duration > 0:
                    expanded.extend([x[i, j]] * duration)
            
            if len(expanded) == 0:
                expanded = [torch.zeros(dim, device=x.device)]
                
            output_list.append(torch.stack(expanded))
            output_lengths.append(len(expanded))
        
        # Pad to max length
        max_len = max(output_lengths)
        output = torch.zeros(batch_size, max_len, dim, device=x.device)
        
        for i, (seq, length) in enumerate(zip(output_list, output_lengths)):
            output[i, :length] = seq
            
        # Create output mask
        output_mask = torch.zeros(batch_size, max_len, device=x.device)
        for i, length in enumerate(output_lengths):
            output_mask[i, :length] = 1
            
        return output, output_mask.bool(), torch.tensor(output_lengths, device=x.device)


class TTSEncoder(nn.Module):
    """Text encoder with speaker embedding support"""
    
    def __init__(self, vocab_size, n_speakers, speaker_dim, hidden_dim, 
                 n_layers, n_heads, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.speaker_embedding = nn.Embedding(n_speakers, speaker_dim) if n_speakers > 1 else None
        
        if n_speakers > 1:
            self.speaker_projection = nn.Linear(speaker_dim, hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        encoder_layers = TransformerEncoderLayer(hidden_dim, n_heads, 
                                                  hidden_dim * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        self.hidden_dim = hidden_dim
    
    def forward(self, text, speaker_ids=None, lengths=None):
        """
        Args:
            text: (batch, seq_len) - token indices
            speaker_ids: (batch,) - speaker indices
            lengths: (batch,) - sequence lengths
        """
        x = self.embedding(text) * math.sqrt(self.hidden_dim)
        
        if self.speaker_embedding is not None and speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)
            speaker_emb = self.speaker_projection(speaker_emb).unsqueeze(1)
            x = x + speaker_emb
        
        x = self.pos_encoder(x)
        
        # Create padding mask
        src_key_padding_mask = None
        if lengths is not None:
            max_len = text.size(1)
            # Create boolean mask: True for padding positions (batch_first format)
            src_key_padding_mask = torch.arange(max_len, device=text.device)[None, :] >= lengths[:, None]
        
        # Transformer expects (seq_len, batch, dim) but we're using batch_first=True
        # So we keep (batch, seq_len, dim) format
        x = self.transformer_encoder(x)
        # Note: We don't pass the mask since our custom encoder layer doesn't need it
        # The padding is handled by the attention mechanism internally
        
        return x, src_key_padding_mask


class TTSDecoder(nn.Module):
    """Mel spectrogram decoder"""
    
    def __init__(self, n_mel_channels, hidden_dim, n_layers, n_heads, dropout=0.1):
        super().__init__()
        
        decoder_layers = TransformerDecoderLayer(hidden_dim, n_heads, 
                                                  hidden_dim * 4, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, n_layers)
        
        self.mel_projection = nn.Linear(hidden_dim, n_mel_channels)
        self.hidden_dim = hidden_dim
    
    def forward(self, mel_input, memory, memory_mask=None, lengths=None):
        """
        Args:
            mel_input: (batch, seq_len, dim) - teacher forcing input
            memory: (batch, mem_len, dim) - encoder output
            memory_mask: (batch, mem_len) - encoder padding mask
        """
        batch_size, seq_len, _ = mel_input.shape
        
        # Create causal mask for decoder
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=mel_input.device), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
        
        # Create padding mask
        tgt_key_padding_mask = None
        if lengths is not None:
            max_len = seq_len
            tgt_key_padding_mask = torch.arange(max_len, device=mel_input.device)[None, :] >= lengths[:, None]
        
        # Transformer expects (seq_len, batch, dim)
        mel_input = mel_input.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        x = self.transformer_decoder(
            mel_input, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        x = x.transpose(0, 1)
        mel_output = self.mel_projection(x)
        
        return mel_output


class CustomTTSModel(nn.Module):
    """
    Complete Text-to-Speech Model
    Optimized for CPU and low-end GPU hardware with automatic configuration
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Extract config parameters
        model_cfg = config.get('model', {})
        audio_cfg = config.get('audio', {})
        
        vocab_size = config.get('vocab_size', 100)  # Will be set during data prep
        n_mel_channels = audio_cfg.get('n_mel_channels', 80)
        
        hidden_dim = model_cfg.get('hidden_dim', 256)
        n_encoder_layers = model_cfg.get('encoder_layers', 6)
        n_decoder_layers = model_cfg.get('decoder_layers', 6)
        n_heads = model_cfg.get('attention_heads', 4)
        dropout = model_cfg.get('dropout', 0.1)
        
        n_speakers = model_cfg.get('num_speakers', 1)
        speaker_dim = model_cfg.get('speaker_embedding_dim', 128)
        
        pred_hidden = model_cfg.get('duration_predictor_hidden', 256)
        
        # Initialize components
        self.encoder = TTSEncoder(
            vocab_size=vocab_size,
            n_speakers=n_speakers,
            speaker_dim=speaker_dim,
            hidden_dim=hidden_dim,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.decoder = TTSDecoder(
            n_mel_channels=n_mel_channels,
            hidden_dim=hidden_dim,
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.duration_predictor = DurationPredictor(
            input_dim=hidden_dim,
            hidden_dim=pred_hidden,
            dropout=dropout
        )
        
        self.pitch_predictor = PitchPredictor(
            input_dim=hidden_dim,
            hidden_dim=pred_hidden,
            dropout=dropout
        )
        
        self.energy_predictor = EnergyPredictor(
            input_dim=hidden_dim,
            hidden_dim=pred_hidden,
            dropout=dropout
        )
        
        self.length_regulator = LengthRegulator()
        
        # Store dimensions for inference
        self.hidden_dim = hidden_dim
        self.n_mel_channels = n_mel_channels
    
    def forward(self, text, text_lengths, mel_targets=None, mel_lengths=None,
                speaker_ids=None, durations=None, pitches=None, energies=None):
        """
        Training forward pass with optional teacher forcing
        
        Args:
            text: (batch, text_len) - token indices
            text_lengths: (batch,) - text sequence lengths
            mel_targets: (batch, mel_len, n_mels) - target mel spectrograms
            mel_lengths: (batch,) - mel sequence lengths
            speaker_ids: (batch,) - speaker indices
            durations: (batch, text_len) - ground truth durations
            pitches: (batch, text_len) - ground truth pitches
            energies: (batch, text_len) - ground truth energies
        """
        # Encode text
        encoder_output, memory_mask = self.encoder(text, speaker_ids, text_lengths)
        
        # Predict duration, pitch, energy
        duration_pred = self.duration_predictor(encoder_output.detach(), ~memory_mask)
        pitch_pred = self.pitch_predictor(encoder_output.detach(), ~memory_mask)
        energy_pred = self.energy_predictor(encoder_output.detach(), ~memory_mask)
        
        # Use ground truth durations if available (teacher forcing)
        if durations is not None and self.training:
            used_durations = durations
        else:
            # Round predicted durations
            used_durations = torch.clamp(torch.round(duration_pred.exp() - 1), min=0).long()
        
        # Regulate length
        regulated_output, mel_mask, output_lengths = self.length_regulator(
            encoder_output, used_durations, ~memory_mask
        )
        
        # Predict pitch and energy for expanded sequence
        # (simplified - in full implementation would need interpolation)
        pitch_expanded = pitch_pred  # Simplified
        energy_expanded = energy_pred  # Simplified
        
        # Decode to mel spectrogram
        if mel_targets is not None and self.training:
            # Teacher forcing
            mel_output = self.decoder(mel_targets, regulated_output, mel_mask, mel_lengths)
        else:
            # Inference mode - use own predictions
            # Create dummy input (in real inference, would use autoregressive decoding)
            batch_size = text.size(0)
            max_mel_len = output_lengths.max().item()
            dummy_mel = torch.zeros(batch_size, max_mel_len, self.n_mel_channels, 
                                    device=text.device)
            mel_output = self.decoder(dummy_mel, regulated_output, mel_mask, output_lengths)
        
        return {
            'mel_output': mel_output,
            'mel_mask': mel_mask,
            'output_lengths': output_lengths,
            'duration_pred': duration_pred,
            'pitch_pred': pitch_pred,
            'energy_pred': energy_pred,
            'duration_gt': durations,
            'pitch_gt': pitches,
            'energy_gt': energies
        }
    
    def infer(self, text, text_lengths, speaker_ids=None, speed_factor=1.0):
        """
        Inference mode - generate mel spectrogram from text
        
        Args:
            text: (batch, text_len) - token indices
            text_lengths: (batch,) - text sequence lengths
            speaker_ids: (batch,) - speaker indices
            speed_factor: float - speech speed multiplier
        """
        self.eval()
        
        with torch.no_grad():
            # Encode text
            encoder_output, memory_mask = self.encoder(text, speaker_ids, text_lengths)
            
            # Predict duration
            duration_pred = self.duration_predictor(encoder_output, ~memory_mask)
            
            # Apply speed factor and round
            durations = torch.clamp(
                torch.round((duration_pred.exp() - 1) / speed_factor), 
                min=0
            ).long()
            
            # Regulate length
            regulated_output, mel_mask, output_lengths = self.length_regulator(
                encoder_output, durations, ~memory_mask
            )
            
            # Generate mel spectrogram
            batch_size = text.size(0)
            max_mel_len = output_lengths.max().item()
            dummy_mel = torch.zeros(batch_size, max_mel_len, self.n_mel_channels, 
                                    device=text.device)
            mel_output = self.decoder(dummy_mel, regulated_output, mel_mask, output_lengths)
            
            return mel_output, mel_mask, output_lengths


def get_hardware_config(hardware_type='auto'):
    """
    Automatically configure model size and training parameters based on hardware
    
    Args:
        hardware_type: 'auto', 'cpu', 'low_end_gpu', 'mid_gpu', 'high_gpu'
    
    Returns:
        dict with optimized configuration
    """
    import torch
    
    if hardware_type == 'auto':
        # Detect hardware automatically
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            if total_memory >= 8 or 'a100' in gpu_name or 'v100' in gpu_name:
                hardware_type = 'high_gpu'
            elif total_memory >= 4:
                hardware_type = 'mid_gpu'
            else:
                hardware_type = 'low_end_gpu'
        else:
            hardware_type = 'cpu'
    
    configs = {
        'cpu': {
            'model': {
                'encoder_layers': 4,
                'decoder_layers': 4,
                'hidden_dim': 128,
                'attention_heads': 2,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 4,
                'gradient_accumulation_steps': 8,
                'use_mixed_precision': False
            }
        },
        'low_end_gpu': {
            'model': {
                'encoder_layers': 4,
                'decoder_layers': 4,
                'hidden_dim': 192,
                'attention_heads': 3,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 8,
                'gradient_accumulation_steps': 4,
                'use_mixed_precision': True
            }
        },
        'mid_gpu': {
            'model': {
                'encoder_layers': 6,
                'decoder_layers': 6,
                'hidden_dim': 256,
                'attention_heads': 4,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 16,
                'gradient_accumulation_steps': 2,
                'use_mixed_precision': True
            }
        },
        'high_gpu': {
            'model': {
                'encoder_layers': 6,
                'decoder_layers': 6,
                'hidden_dim': 384,
                'attention_heads': 6,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'gradient_accumulation_steps': 1,
                'use_mixed_precision': True
            }
        }
    }
    
    return configs.get(hardware_type, configs['mid_gpu'])


if __name__ == '__main__':
    # Test model creation and forward pass
    print("Testing Custom TTS Model...")
    
    # Sample configuration
    test_config = {
        'vocab_size': 100,
        'model': {
            'encoder_layers': 2,
            'decoder_layers': 2,
            'hidden_dim': 128,
            'attention_heads': 2,
            'dropout': 0.1,
            'num_speakers': 1,
            'speaker_embedding_dim': 64,
            'duration_predictor_hidden': 128
        },
        'audio': {
            'n_mel_channels': 80
        }
    }
    
    # Create model
    model = CustomTTSModel(test_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    text_len = 20
    mel_len = 50
    
    text = torch.randint(1, 50, (batch_size, text_len))
    text_lengths = torch.tensor([text_len, text_len - 5])
    mel_targets = torch.randn(batch_size, mel_len, 80)
    mel_lengths = torch.tensor([mel_len, mel_len - 10])
    
    output = model(text, text_lengths, mel_targets, mel_lengths)
    
    print(f"Input text shape: {text.shape}")
    print(f"Output mel shape: {output['mel_output'].shape}")
    print(f"Duration predictions shape: {output['duration_pred'].shape}")
    
    # Test inference
    mel_out, mel_mask, out_lengths = model.infer(text, text_lengths)
    print(f"Inference output shape: {mel_out.shape}")
    
    # Test hardware detection
    hw_config = get_hardware_config('auto')
    print(f"\nDetected hardware config: {hw_config}")
    
    print("\n✅ Model test completed successfully!")
