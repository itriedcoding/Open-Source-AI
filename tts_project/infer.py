"""
Inference script for Custom TTS Model.
Generate speech from text using trained model.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
import yaml
import soundfile as sf
from tqdm import tqdm

from models.tts_model import create_model
from data.prepare_dataset import TextProcessor


class Vocoder:
    """Simple Griffin-Lim vocoder for mel-to-audio conversion."""
    
    def __init__(self, config: Dict):
        self.sample_rate = config['audio']['sample_rate']
        self.hop_length = config['audio']['hop_length']
        self.win_length = config['audio']['win_length']
        self.n_fft = config['audio']['n_fft']
        self.n_mels = config['audio']['n_mels']
        self.fmin = config['audio']['fmin']
        self.fmax = config['audio']['fmax']
        
        # Pre-compute mel filterbank
        import librosa
        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
    
    def mel_to_audio(self, mel_spec: np.ndarray, n_iter: int = 50) -> np.ndarray:
        """Convert mel spectrogram to audio using Griffin-Lim algorithm."""
        import librosa
        
        # Invert log mel scale
        mel_spec = np.exp(mel_spec)
        
        # Convert mel to linear spectrogram
        # Use pseudo-inverse of mel filterbank
        mel_basis_inv = np.linalg.pinv(self.mel_basis)
        spec = np.dot(mel_basis_inv, mel_spec.T).T
        spec = np.maximum(spec, 1e-10)  # Ensure non-negative
        
        # Griffin-Lim algorithm
        audio = librosa.griffinlim(
            spec,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            n_iter=n_iter
        )
        
        return audio


class TTSInference:
    """Text-to-Speech inference engine."""
    
    def __init__(self, checkpoint_path: str, config_path: str = 'configs/model_config.yaml',
                 device: Optional[str] = None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load vocabulary
        vocab_path = Path(checkpoint_path).parent / 'vocab.json'
        if vocab_path.exists():
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
                self.char_to_idx = vocab_data['char_to_idx']
                self.idx_to_char = vocab_data['idx_to_char']
        else:
            # Try to load from dataset directory
            vocab_path = Path('data/processed/vocab.json')
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    vocab_data = json.load(f)
                    self.char_to_idx = vocab_data['char_to_idx']
                    self.idx_to_char = vocab_data['idx_to_char']
            else:
                raise FileNotFoundError("Vocabulary file not found")
        
        # Create text processor
        self.text_processor = TextProcessor(self.config)
        self.text_processor.char_to_idx = self.char_to_idx
        self.text_processor.idx_to_char = self.idx_to_char
        
        # Create and load model
        self.model = create_model(self.config).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        
        # Create vocoder
        self.vocoder = Vocoder(self.config)
    
    def text_to_sequence(self, text: str) -> torch.Tensor:
        """Convert text to tensor of indices."""
        sequence = self.text_processor.text_to_sequence(text)
        return torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def synthesize(self, text: str, speed_factor: float = 1.0,
                   save_path: Optional[str] = None) -> Dict:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            speed_factor: Speech speed adjustment (>1 faster, <1 slower)
            save_path: Optional path to save generated audio
        
        Returns:
            Dictionary containing audio and metadata
        """
        # Convert text to sequence
        text_tensor = self.text_to_sequence(text)
        text_lengths = torch.tensor([text_tensor.size(1)], dtype=torch.long).to(self.device)
        
        # Generate mel spectrogram
        outputs = self.model.inference(text_tensor, text_lengths, speed_factor)
        
        # Extract mel spectrogram
        mel_output = outputs['mel_output'][0].cpu().numpy()
        
        # Convert to audio
        audio = self.vocoder.mel_to_audio(mel_output)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save audio if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            sf.write(save_path, audio, self.config['audio']['sample_rate'])
            print(f"Saved audio to {save_path}")
        
        return {
            'text': text,
            'audio': audio,
            'mel_spec': mel_output,
            'sample_rate': self.config['audio']['sample_rate'],
            'duration': len(audio) / self.config['audio']['sample_rate']
        }
    
    def synthesize_batch(self, texts: list, output_dir: str, 
                        speed_factor: float = 1.0) -> list:
        """
        Synthesize speech for multiple texts.
        
        Args:
            texts: List of texts to synthesize
            output_dir: Directory to save generated audio files
            speed_factor: Speech speed adjustment
        
        Returns:
            List of paths to generated audio files
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_paths = []
        
        for i, text in enumerate(tqdm(texts, desc="Synthesizing")):
            save_path = os.path.join(output_dir, f'output_{i:04d}.wav')
            result = self.synthesize(text, speed_factor, save_path)
            audio_paths.append(save_path)
        
        return audio_paths


def main():
    parser = argparse.ArgumentParser(description='Custom TTS Inference')
    parser.add_argument('--text', type=str, required=True,
                       help='Text to synthesize')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Path to save generated audio')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech speed factor (default: 1.0)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create inference engine
    tts = TTSInference(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        device=args.device
    )
    
    # Synthesize speech
    print(f"Synthesizing: '{args.text}'")
    result = tts.synthesize(
        args.text,
        speed_factor=args.speed,
        save_path=args.output
    )
    
    print(f"Generated audio duration: {result['duration']:.2f}s")
    print(f"Audio saved to: {args.output}")


if __name__ == '__main__':
    main()
