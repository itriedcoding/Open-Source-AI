"""
Custom TTS Model Architecture.
Transformer-based encoder-decoder with duration, pitch, and energy predictors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_seq_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through encoder layer."""
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through decoder layer."""
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class DurationPredictor(nn.Module):
    """Predicts duration of each phoneme."""
    
    def __init__(self, in_features: int, out_features: int = 1, 
                 num_layers: int = 2, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            conv = nn.Conv1d(
                in_features if i == 0 else in_features,
                in_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            layers.extend([conv, nn.ReLU(), nn.Dropout(dropout)])
        
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict durations."""
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.linear(x)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        
        return x.squeeze(-1)  # (B, T)


class PitchPredictor(nn.Module):
    """Predicts pitch (F0) for each frame."""
    
    def __init__(self, in_features: int, out_features: int = 1,
                 num_layers: int = 2, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            conv = nn.Conv1d(
                in_features if i == 0 else in_features,
                in_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            layers.extend([conv, nn.ReLU(), nn.Dropout(dropout)])
        
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict pitch."""
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.linear(x)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        
        return x.squeeze(-1)  # (B, T)


class EnergyPredictor(nn.Module):
    """Predicts energy for each frame."""
    
    def __init__(self, in_features: int, out_features: int = 1,
                 num_layers: int = 2, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            conv = nn.Conv1d(
                in_features if i == 0 else in_features,
                in_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            layers.extend([conv, nn.ReLU(), nn.Dropout(dropout)])
        
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict energy."""
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.linear(x)
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        
        return x.squeeze(-1)  # (B, T)


class CustomTTSModel(nn.Module):
    """Main TTS model with transformer encoder-decoder architecture."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Extract configuration
        text_enc_cfg = config['model']['text_encoder']
        acoustic_dec_cfg = config['model']['acoustic_decoder']
        
        # Text embedding
        self.text_embedding = nn.Embedding(
            text_enc_cfg['vocab_size'],
            text_enc_cfg['embedding_dim'],
            padding_idx=0  # <pad> token
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            text_enc_cfg['embedding_dim'],
            text_enc_cfg['max_seq_len'],
            text_enc_cfg['dropout']
        )
        
        # Transformer encoder layers
        encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                text_enc_cfg['embedding_dim'],
                text_enc_cfg['num_heads'],
                text_enc_cfg['ff_dim'],
                text_enc_cfg['dropout']
            )
            for _ in range(text_enc_cfg['num_layers'])
        ])
        self.text_encoder = nn.ModuleList(encoder_layers)
        
        # Duration predictor
        self.duration_predictor = DurationPredictor(
            text_enc_cfg['embedding_dim'],
            **{k: v for k, v in config['model']['duration_predictor'].items() 
               if k != 'in_features'}
        )
        
        # Pitch predictor
        self.pitch_predictor = PitchPredictor(
            text_enc_cfg['embedding_dim'],
            **{k: v for k, v in config['model']['pitch_predictor'].items()
               if k != 'in_features'}
        )
        
        # Energy predictor
        self.energy_predictor = EnergyPredictor(
            text_enc_cfg['embedding_dim'],
            **{k: v for k, v in config['model']['energy_predictor'].items()
               if k != 'in_features'}
        )
        
        # Length regulator (expands encoder output based on predicted durations)
        self.length_regulator = LengthRegulator()
        
        # Transformer decoder layers
        decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                acoustic_dec_cfg['in_features'],
                acoustic_dec_cfg['num_heads'],
                acoustic_dec_cfg['ff_dim'],
                acoustic_dec_cfg['dropout']
            )
            for _ in range(acoustic_dec_cfg['num_layers'])
        ])
        self.acoustic_decoder = nn.ModuleList(decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(
            acoustic_dec_cfg['in_features'],
            acoustic_dec_cfg['out_features']
        )
        
        # Learnable speaker embedding (for multi-speaker support)
        self.speaker_embedding = nn.Embedding(100, text_enc_cfg['embedding_dim'])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_mask(self, lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        """Generate attention mask from sequence lengths."""
        batch_size = lengths.size(0)
        if max_len is None:
            max_len = lengths.max().item()
        
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        mask = mask >= lengths.unsqueeze(1)
        return mask
    
    def encode_text(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """Encode input text."""
        # Get text embedding
        x = self.text_embedding(text) * torch.sqrt(torch.tensor(self.config['model']['text_encoder']['embedding_dim']))
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate mask
        mask = self.generate_mask(text_lengths, text.size(1))
        
        # Pass through encoder layers
        for encoder_layer in self.text_encoder:
            x = encoder_layer(x, src_key_padding_mask=mask)
        
        return x, mask
    
    def decode_acoustics(self, encoder_output: torch.Tensor, encoder_mask: torch.Tensor,
                         durations: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode acoustic features from encoder output."""
        batch_size = encoder_output.size(0)
        
        # If durations not provided, predict them
        if durations is None:
            pred_durations = self.duration_predictor(encoder_output, encoder_mask)
            durations = torch.clamp(torch.round(pred_durations), min=0).long()
        
        # Expand encoder output based on durations
        expanded_output, target_lengths = self.length_regulator(encoder_output, durations)
        
        # Generate decoder mask for expanded output
        decoder_mask = self.generate_mask(target_lengths, expanded_output.size(1))
        
        # Create causal mask for decoder self-attention
        tgt_len = expanded_output.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        causal_mask = causal_mask.to(expanded_output.device)
        
        # Pass through decoder layers
        x = expanded_output
        for decoder_layer in self.acoustic_decoder:
            x = decoder_layer(
                x, 
                memory=expanded_output,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=decoder_mask,
                memory_key_padding_mask=None  # Use None since memory and tgt have same lengths now
            )
        
        # Project to output dimension
        output = self.output_projection(x)
        
        return output, target_lengths
    
    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor,
                mel_targets: Optional[torch.Tensor] = None,
                pitch_targets: Optional[torch.Tensor] = None,
                energy_targets: Optional[torch.Tensor] = None,
                durations: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            text: Input text tokens (B, T_text)
            text_lengths: Lengths of text sequences (B,)
            mel_targets: Target mel spectrograms (B, T_mel, n_mels) - optional for inference
            pitch_targets: Target pitch values (B, T_mel) - optional
            energy_targets: Target energy values (B, T_mel) - optional
            durations: Ground truth durations (B, T_text) - optional
        
        Returns:
            Dictionary containing model outputs
        """
        # Encode text
        encoder_output, encoder_mask = self.encode_text(text, text_lengths)
        
        # Predict duration, pitch, and energy
        pred_durations = self.duration_predictor(encoder_output, encoder_mask)
        pred_pitch = self.pitch_predictor(encoder_output, encoder_mask)
        pred_energy = self.energy_predictor(encoder_output, encoder_mask)
        
        # Use ground truth durations during training if available
        if durations is not None:
            durations_long = torch.clamp(torch.round(durations), min=0).long()
            mel_output, mel_lengths = self.decode_acoustics(encoder_output, encoder_mask, durations_long)
        else:
            # Use predicted durations
            pred_durations_rounded = torch.clamp(torch.round(pred_durations), min=0).long()
            mel_output, mel_lengths = self.decode_acoustics(encoder_output, encoder_mask, pred_durations_rounded)
        
        # Prepare outputs
        outputs = {
            'mel_output': mel_output,
            'pred_durations': pred_durations,
            'pred_pitch': pred_pitch,
            'pred_energy': pred_energy,
            'mel_lengths': mel_lengths
        }
        
        # Add ground truth targets if available
        if mel_targets is not None:
            outputs['mel_targets'] = mel_targets
        if pitch_targets is not None:
            outputs['pitch_targets'] = pitch_targets
        if energy_targets is not None:
            outputs['energy_targets'] = energy_targets
        if durations is not None:
            outputs['durations'] = durations
        
        return outputs
    
    @torch.no_grad()
    def inference(self, text: torch.Tensor, text_lengths: torch.Tensor,
                  speed_factor: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Inference mode - generate speech from text.
        
        Args:
            text: Input text tokens (B, T_text)
            text_lengths: Lengths of text sequences (B,)
            speed_factor: Speed adjustment factor (>1 faster, <1 slower)
        
        Returns:
            Dictionary containing generated mel spectrogram and other outputs
        """
        self.eval()
        
        # Encode text
        encoder_output, encoder_mask = self.encode_text(text, text_lengths)
        
        # Predict and adjust durations
        pred_durations = self.duration_predictor(encoder_output, encoder_mask)
        pred_durations_adjusted = torch.clamp(
            torch.round(pred_durations / speed_factor), min=0
        ).long()
        
        # Expand encoder output
        expanded_output, mel_lengths = self.length_regulator(encoder_output, pred_durations_adjusted)
        
        # Predict pitch and energy
        pred_pitch = self.pitch_predictor(encoder_output, encoder_mask)
        pred_energy = self.energy_predictor(encoder_output, encoder_mask)
        
        # Decode acoustics
        mel_output = self.decode_acoustics(encoder_output, encoder_mask, mel_lengths)
        
        return {
            'mel_output': mel_output,
            'pred_durations': pred_durations,
            'pred_pitch': pred_pitch,
            'pred_energy': pred_energy,
            'mel_lengths': mel_lengths
        }


class LengthRegulator(nn.Module):
    """Expands encoder output based on predicted durations."""
    
    def forward(self, encoder_output: torch.Tensor, 
                durations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand encoder output based on durations.
        
        Args:
            encoder_output: Encoder output (B, T_text, D)
            durations: Duration for each token (B, T_text)
        
        Returns:
            Tuple of (expanded_output, output_lengths)
        """
        batch_size, max_text_len, _ = encoder_output.shape
        
        # Ensure durations are integers
        durations = durations.long()
        
        # Calculate output lengths
        output_lengths = durations.sum(dim=1)
        max_output_len = output_lengths.max().item()
        
        # Expand encoder output
        expanded_output = []
        for b in range(batch_size):
            expanded_frames = []
            for t in range(max_text_len):
                # Repeat each encoder output according to its duration
                repeat_count = int(durations[b, t].item())
                if repeat_count > 0:
                    repeated = encoder_output[b, t:t+1, :].repeat(repeat_count, 1)
                    expanded_frames.append(repeated)
            
            # Concatenate all frames
            if expanded_frames:
                expanded = torch.cat(expanded_frames, dim=0)
                # Pad to max length if needed
                if expanded.size(0) < max_output_len:
                    padding = torch.zeros(
                        max_output_len - expanded.size(0),
                        expanded.size(1),
                        device=expanded.device
                    )
                    expanded = torch.cat([expanded, padding], dim=0)
            else:
                expanded = torch.zeros(max_output_len, encoder_output.size(2), 
                                      device=encoder_output.device)
            
            expanded_output.append(expanded)
        
        expanded_output = torch.stack(expanded_output)
        
        return expanded_output, output_lengths


def create_model(config: Dict) -> CustomTTSModel:
    """Create model from configuration."""
    return CustomTTSModel(config)


def get_hardware_optimized_config(hardware: str = 'auto') -> Dict:
    """
    Returns model configuration optimized for specific hardware.
    
    Args:
        hardware: Hardware type - 'cpu', 'low_end_gpu', 'mid_gpu', 'high_gpu', or 'auto'
        
    Returns:
        Dictionary with optimized model parameters
    """
    import torch
    
    configs = {
        'cpu': {
            'model': {
                'text_encoder': {
                    'vocab_size': 512,
                    'embedding_dim': 128,
                    'max_seq_len': 500,
                    'num_layers': 2,
                    'num_heads': 2,
                    'ff_dim': 512,
                    'dropout': 0.1
                },
                'acoustic_decoder': {
                    'in_features': 128,
                    'num_layers': 2,
                    'num_heads': 2,
                    'ff_dim': 512,
                    'out_features': 80,
                    'dropout': 0.1
                },
                'duration_predictor': {'in_features': 128, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5},
                'pitch_predictor': {'in_features': 128, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5},
                'energy_predictor': {'in_features': 128, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5}
            },
            'training': {'batch_size': 4, 'gradient_accumulation_steps': 8, 'mixed_precision': False}
        },
        'low_end_gpu': {
            'model': {
                'text_encoder': {
                    'vocab_size': 512,
                    'embedding_dim': 256,
                    'max_seq_len': 500,
                    'num_layers': 3,
                    'num_heads': 4,
                    'ff_dim': 1024,
                    'dropout': 0.1
                },
                'acoustic_decoder': {
                    'in_features': 256,
                    'num_layers': 3,
                    'num_heads': 4,
                    'ff_dim': 1024,
                    'out_features': 80,
                    'dropout': 0.1
                },
                'duration_predictor': {'in_features': 256, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5},
                'pitch_predictor': {'in_features': 256, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5},
                'energy_predictor': {'in_features': 256, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5}
            },
            'training': {'batch_size': 8, 'gradient_accumulation_steps': 4, 'mixed_precision': True}
        },
        'mid_gpu': {
            'model': {
                'text_encoder': {
                    'vocab_size': 512,
                    'embedding_dim': 384,
                    'max_seq_len': 500,
                    'num_layers': 4,
                    'num_heads': 6,
                    'ff_dim': 1536,
                    'dropout': 0.1
                },
                'acoustic_decoder': {
                    'in_features': 384,
                    'num_layers': 4,
                    'num_heads': 6,
                    'ff_dim': 1536,
                    'out_features': 80,
                    'dropout': 0.1
                },
                'duration_predictor': {'in_features': 384, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5},
                'pitch_predictor': {'in_features': 384, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5},
                'energy_predictor': {'in_features': 384, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5}
            },
            'training': {'batch_size': 16, 'gradient_accumulation_steps': 2, 'mixed_precision': True}
        },
        'high_gpu': {
            'model': {
                'text_encoder': {
                    'vocab_size': 512,
                    'embedding_dim': 512,
                    'max_seq_len': 500,
                    'num_layers': 6,
                    'num_heads': 8,
                    'ff_dim': 2048,
                    'dropout': 0.1
                },
                'acoustic_decoder': {
                    'in_features': 512,
                    'num_layers': 6,
                    'num_heads': 8,
                    'ff_dim': 2048,
                    'out_features': 80,
                    'dropout': 0.1
                },
                'duration_predictor': {'in_features': 512, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5},
                'pitch_predictor': {'in_features': 512, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5},
                'energy_predictor': {'in_features': 512, 'out_features': 1, 'num_layers': 2, 'kernel_size': 3, 'dropout': 0.5}
            },
            'training': {'batch_size': 32, 'gradient_accumulation_steps': 1, 'mixed_precision': True}
        }
    }
    
    if hardware == 'auto':
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if vram_gb < 4 or any(x in gpu_name for x in ['1050', '1060', 'mx', 'gt']):
                print(f"Auto-detected low-end GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
                return configs['low_end_gpu']
            elif vram_gb < 8:
                print(f"Auto-detected mid-range GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
                return configs['mid_gpu']
            else:
                print(f"Auto-detected high-end GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
                return configs['high_gpu']
        else:
            print("No GPU detected, using CPU-optimized configuration")
            return configs['cpu']
    
    if hardware not in configs:
        raise ValueError(f"Unknown hardware type: {hardware}. Choose from: {list(configs.keys())}")
    
    return configs[hardware]


if __name__ == '__main__':
    # Test model creation
    import yaml
    
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    text_len = 20
    mel_len = 50
    
    text = torch.randint(1, 50, (batch_size, text_len))
    text_lengths = torch.tensor([text_len, text_len - 5])
    mel_targets = torch.randn(batch_size, mel_len, 80)
    pitch_targets = torch.randn(batch_size, mel_len)
    energy_targets = torch.randn(batch_size, mel_len)
    durations = torch.randint(1, 5, (batch_size, text_len)).float()
    
    outputs = model(
        text, text_lengths,
        mel_targets, pitch_targets, energy_targets, durations
    )
    
    print("Model test passed!")
    print(f"Mel output shape: {outputs['mel_output'].shape}")
    print(f"Duration predictions shape: {outputs['pred_durations'].shape}")
    
    # Test hardware auto-detection
    print("\nTesting hardware auto-detection:")
    hw_config = get_hardware_optimized_config('auto')
    print(f"Selected config batch size: {hw_config['training']['batch_size']}")
    print(f"Gradient accumulation steps: {hw_config['training']['gradient_accumulation_steps']}")
