"""
Dataset Preparation Script for Custom TTS
Handles audio preprocessing, feature extraction, and train/val/test splitting
Optimized for low-end hardware with memory-efficient processing
"""

import os
import numpy as np
import torch
import torchaudio
import librosa
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class AudioPreprocessor:
    """Audio preprocessing with mel spectrogram, pitch, and energy extraction"""
    
    def __init__(self, config):
        self.config = config
        audio_cfg = config.get('audio', {})
        
        self.sample_rate = audio_cfg.get('sample_rate', 22050)
        self.n_mel_channels = audio_cfg.get('n_mel_channels', 80)
        self.n_fft = audio_cfg.get('n_fft', 1024)
        self.hop_length = audio_cfg.get('hop_length', 256)
        self.win_length = audio_cfg.get('win_length', 1024)
        self.fmin = audio_cfg.get('fmin', 0)
        self.fmax = audio_cfg.get('fmax', 8000)
        self.max_wav_value = audio_cfg.get('max_wav_value', 32768.0)
        
        # Mel filterbank
        self.mel_basis = torch.from_numpy(
            librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, 
                               n_mels=self.n_mel_channels, fmin=self.fmin, fmax=self.fmax)
        ).float()
    
    def load_audio(self, filepath):
        """Load and normalize audio file"""
        waveform, sr = torchaudio.load(filepath)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Normalize to [-1, 1]
        waveform = waveform / self.max_wav_value
        
        return waveform.squeeze(0)
    
    def extract_mel_spectrogram(self, waveform):
        """Extract mel spectrogram from waveform"""
        # Apply STFT
        spec = torch.stft(
            waveform.unsqueeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        ).abs().squeeze(0)
        
        # Convert to mel scale
        mel_spec = torch.matmul(self.mel_basis, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return mel_spec.transpose(0, 1)  # (time, mel_bins)
    
    def extract_pitch(self, waveform):
        """Extract pitch (F0) using librosa"""
        waveform_np = waveform.numpy()
        
        # Extract F0
        f0, voiced_flag, _ = librosa.pyin(
            waveform_np,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )
        
        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        
        return torch.from_numpy(f0).float()
    
    def extract_energy(self, mel_spec):
        """Extract energy from mel spectrogram"""
        energy = torch.norm(mel_spec.exp(), p=2, dim=1)
        return energy
    
    def process_file(self, filepath):
        """Process a single audio file"""
        try:
            waveform = self.load_audio(filepath)
            
            # Extract features
            mel_spec = self.extract_mel_spectrogram(waveform)
            pitch = self.extract_pitch(waveform)
            energy = self.extract_energy(mel_spec)
            
            # Adjust lengths to match
            min_len = min(mel_spec.shape[0], len(pitch), len(energy))
            mel_spec = mel_spec[:min_len]
            pitch = pitch[:min_len]
            energy = energy[:min_len]
            
            return {
                'mel': mel_spec,
                'pitch': pitch,
                'energy': energy,
                'duration': float(waveform.shape[0]) / self.sample_rate
            }
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None


class TextTokenizer:
    """Simple character-level tokenizer with custom vocabulary"""
    
    def __init__(self):
        # Basic English characters and symbols
        self.chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.digits = list('0123456789')
        self.punctuation = list('.,!?;:\'"()- ')
        self.symbols = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
        self.vocab = self.symbols + self.chars + self.digits + self.punctuation
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
    
    def encode(self, text):
        """Convert text to token indices"""
        tokens = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in text]
        return [self.char_to_idx['<SOS>']] + tokens + [self.char_to_idx['<EOS>']]
    
    def decode(self, tokens):
        """Convert token indices back to text"""
        text = ''.join([self.idx_to_char.get(t, '<UNK>') for t in tokens])
        return text
    
    @property
    def vocab_size(self):
        return len(self.vocab)


def load_transcripts(transcript_file):
    """Load transcript file (format: audio_id|text)"""
    transcripts = {}
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            parts = line.split('|', 1)
            if len(parts) == 2:
                audio_id, text = parts
                transcripts[audio_id.strip()] = text.strip()
    
    return transcripts


def prepare_dataset(config_path, input_dir, output_dir):
    """Main dataset preparation function"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    preprocessor = AudioPreprocessor(config)
    tokenizer = TextTokenizer()
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(input_dir).rglob(f'*{ext}'))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load transcripts
    transcript_file = Path(input_dir) / 'transcripts.txt'
    transcripts = {}
    
    if transcript_file.exists():
        transcripts = load_transcripts(transcript_file)
        print(f"Loaded {len(transcripts)} transcripts")
    else:
        print("No transcripts.txt found. Using filenames as text.")
    
    # Process each audio file
    processed_data = []
    
    for audio_path in tqdm(audio_files, desc="Processing audio"):
        audio_id = audio_path.stem
        
        # Get transcript
        text = transcripts.get(audio_id, audio_id.replace('_', ' ').replace('-', ' '))
        
        # Process audio
        features = preprocessor.process_file(str(audio_path))
        
        if features is None:
            continue
        
        # Tokenize text
        tokens = tokenizer.encode(text)
        
        # Store data
        processed_data.append({
            'audio_id': audio_id,
            'audio_path': str(audio_path),
            'text': text,
            'tokens': tokens,
            'mel': features['mel'],
            'pitch': features['pitch'],
            'energy': features['energy'],
            'duration': features['duration']
        })
    
    print(f"Successfully processed {len(processed_data)} files")
    
    # Split into train/val/test
    data_cfg = config.get('data', {})
    train_ratio = data_cfg.get('train_split', 0.8)
    val_ratio = data_cfg.get('val_split', 0.1)
    
    # First split: train vs (val+test)
    train_data, temp_data = train_test_split(
        processed_data, 
        train_size=train_ratio,
        random_state=42
    )
    
    # Second split: val vs test
    val_test_ratio = val_ratio / (1 - train_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_test_ratio,
        random_state=42
    )
    
    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Save processed data
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, data in splits.items():
        split_path = output_path / split_name
        split_path.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = []
        for item in data:
            # Save features
            feature_path = split_path / f"{item['audio_id']}.pt"
            torch.save({
                'mel': item['mel'],
                'pitch': item['pitch'],
                'energy': item['energy'],
                'tokens': torch.tensor(item['tokens']),
                'text': item['text'],
                'duration': item['duration']
            }, feature_path)
            
            metadata.append({
                'audio_id': item['audio_id'],
                'feature_file': f"{split_name}/{item['audio_id']}.pt",
                'text': item['text'],
                'duration': item['duration']
            })
        
        # Save metadata JSON-like file
        import json
        with open(output_path / f'{split_name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Save vocabulary
    vocab_info = {
        'vocab_size': tokenizer.vocab_size,
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char
    }
    
    with open(output_path / 'vocab.json', 'w') as f:
        import json
        json.dump(vocab_info, f, indent=2)
    
    # Update config with vocab size
    config['vocab_size'] = tokenizer.vocab_size
    with open(output_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"Output directory: {output_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare TTS dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input_dir', type=str, default='data/raw',
                       help='Input directory with audio files')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Create sample data if input directory doesn't exist
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Creating sample data in {args.input_dir}...")
        input_path.mkdir(parents=True, exist_ok=True)
        
        # Create a simple transcript file
        sample_transcripts = [
            "hello_world|Hello world this is a test",
            "sample_audio|This is another sample for training",
            "demo_file|Artificial intelligence is amazing"
        ]
        
        with open(input_path / 'transcripts.txt', 'w') as f:
            f.write('\n'.join(sample_transcripts))
        
        print("Sample transcripts created. Please add your audio files.")
        print("Audio files should be named: hello_world.wav, sample_audio.wav, etc.")
    
    # Run preparation
    prepare_dataset(args.config, args.input_dir, args.output_dir)
