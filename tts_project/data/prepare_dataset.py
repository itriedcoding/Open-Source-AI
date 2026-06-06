"""
Dataset preparation pipeline for Custom TTS Model.
Handles audio preprocessing, feature extraction, and dataset creation.
"""

import os
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import yaml


class AudioPreprocessor:
    """Preprocesses audio files for TTS training."""
    
    def __init__(self, config: Dict):
        self.sample_rate = config['audio']['sample_rate']
        self.hop_length = config['audio']['hop_length']
        self.win_length = config['audio']['win_length']
        self.n_fft = config['audio']['n_fft']
        self.n_mels = config['audio']['n_mels']
        self.fmin = config['audio']['fmin']
        self.fmax = config['audio']['fmax']
        self.preemphasis = config['audio']['preemphasis']
        
        # Create mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and resample audio file."""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio
    
    def preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter."""
        return librosa.effects.preemphasis(audio, coef=self.preemphasis)
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio."""
        # Apply pre-emphasis
        audio = self.preemphasis(audio)
        
        # Compute spectrogram
        spec = np.abs(librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        ))
        
        # Convert to mel scale
        mel_spec = np.dot(self.mel_basis, spec)
        
        # Convert to log scale
        log_mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
        
        return log_mel_spec.T  # Transpose to (time, mel_bins)
    
    def extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """Extract pitch (F0) from audio."""
        pitch, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        # Fill NaN values with 0
        pitch = np.nan_to_num(pitch, nan=0.0)
        return pitch
    
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract energy from audio."""
        # Compute short-time energy
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )[0]
        return np.log(np.clip(energy, a_min=1e-5, a_max=None))
    
    def process_audio(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Process a single audio file and extract all features."""
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Extract features
        mel_spec = self.extract_mel_spectrogram(audio)
        pitch = self.extract_pitch(audio)
        energy = self.extract_energy(audio)
        
        # Ensure all features have the same time dimension
        min_len = min(len(mel_spec), len(pitch), len(energy))
        mel_spec = mel_spec[:min_len]
        pitch = pitch[:min_len]
        energy = energy[:min_len]
        
        return {
            'mel_spec': mel_spec,
            'pitch': pitch,
            'energy': energy,
            'duration': len(audio) / self.sample_rate
        }


class TextProcessor:
    """Processes text transcripts for TTS training."""
    
    def __init__(self, config: Dict):
        self.vocab_size = config['model']['text_encoder']['vocab_size']
        self.max_seq_len = config['model']['text_encoder']['max_seq_len']
        
        # Build character vocabulary
        self.char_to_idx, self.idx_to_char = self._build_vocab()
        
    def _build_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build character-to-index mapping."""
        # Basic characters
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        chars += list("0123456789")
        chars += list(".,!?;:'\"()- ")
        chars += list("àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ")
        chars += list("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ")
        
        # Special tokens
        chars = ['<pad>', '<bos>', '<eos>', '<unk>'] + chars
        
        # Limit vocabulary size
        chars = chars[:self.vocab_size - 4]  # Reserve space for special tokens
        
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        return char_to_idx, idx_to_char
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices."""
        sequence = []
        for char in text:
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char])
            else:
                sequence.append(self.char_to_idx['<unk>'])
        
        # Add BOS and EOS tokens
        sequence = [self.char_to_idx['<bos>']] + sequence + [self.char_to_idx['<eos>']]
        
        # Truncate if too long
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
            
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of indices back to text."""
        text = ''.join([self.idx_to_char.get(idx, '<unk>') for idx in sequence])
        # Remove special tokens
        text = text.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')
        return text


class DatasetBuilder:
    """Builds the complete TTS dataset."""
    
    def __init__(self, config: Dict):
        self.audio_processor = AudioPreprocessor(config)
        self.text_processor = TextProcessor(config)
        self.config = config
        
        self.min_audio_length = config['data']['min_audio_length']
        self.max_audio_length = config['data']['max_audio_length']
        
    def parse_transcripts(self, transcript_file: str) -> List[Dict]:
        """Parse transcript file into list of samples."""
        samples = []
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                    
                parts = line.split('|', 1)
                if len(parts) != 2:
                    continue
                    
                audio_id, text = parts
                samples.append({
                    'audio_id': audio_id.strip(),
                    'text': text.strip()
                })
                
        return samples
    
    def build_dataset(self, input_dir: str, output_dir: str) -> None:
        """Build the complete dataset."""
        print("Building dataset...")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
        
        # Find transcript file
        transcript_file = None
        for filename in ['transcripts.txt', 'metadata.csv', 'transcripts.csv']:
            potential_path = os.path.join(input_dir, filename)
            if os.path.exists(potential_path):
                transcript_file = potential_path
                break
                
        if transcript_file is None:
            raise FileNotFoundError("No transcript file found in input directory")
        
        # Parse transcripts
        samples = self.parse_transcripts(transcript_file)
        print(f"Found {len(samples)} samples in transcripts")
        
        # Process each sample
        processed_samples = []
        for sample in tqdm(samples, desc="Processing samples"):
            audio_id = sample['audio_id']
            text = sample['text']
            
            # Find audio file
            audio_path = None
            for ext in ['.wav', '.flac', '.mp3', '.ogg']:
                potential_path = os.path.join(input_dir, 'audio', f"{audio_id}{ext}")
                if os.path.exists(potential_path):
                    audio_path = potential_path
                    break
            
            if audio_path is None:
                print(f"Warning: Audio file not found for {audio_id}, skipping...")
                continue
            
            try:
                # Process audio
                features = self.audio_processor.process_audio(audio_path)
                
                # Check duration constraints
                duration = features['duration']
                if duration < self.min_audio_length or duration > self.max_audio_length:
                    print(f"Warning: Audio {audio_id} duration ({duration}s) out of range, skipping...")
                    continue
                
                # Process text
                text_sequence = self.text_processor.text_to_sequence(text)
                
                # Save features
                feature_path = os.path.join(output_dir, 'features', f"{audio_id}.npy")
                np.save(feature_path, {
                    'mel_spec': features['mel_spec'],
                    'pitch': features['pitch'],
                    'energy': features['energy'],
                    'text_sequence': text_sequence,
                    'text': text,
                    'duration': duration
                })
                
                # Add to processed samples
                processed_samples.append({
                    'audio_id': audio_id,
                    'feature_path': feature_path,
                    'text': text,
                    'duration': duration
                })
                
            except Exception as e:
                print(f"Error processing {audio_id}: {str(e)}")
                continue
        
        # Split dataset
        random.shuffle(processed_samples)
        total = len(processed_samples)
        train_split = int(total * self.config['data']['train_split'])
        val_split = int(total * self.config['data']['val_split'])
        
        train_samples = processed_samples[:train_split]
        val_samples = processed_samples[train_split:train_split + val_split]
        test_samples = processed_samples[train_split + val_split:]
        
        # Save dataset metadata
        for split_name, split_data in [
            ('train', train_samples),
            ('val', val_samples),
            ('test', test_samples)
        ]:
            metadata_path = os.path.join(output_dir, f'{split_name}_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(split_data, f, indent=2)
        
        # Save vocabulary
        vocab_path = os.path.join(output_dir, 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump({
                'char_to_idx': self.text_processor.char_to_idx,
                'idx_to_char': self.text_processor.idx_to_char
            }, f, indent=2)
        
        # Save statistics
        stats = {
            'total_samples': len(processed_samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'avg_duration': np.mean([s['duration'] for s in processed_samples]),
            'min_duration': np.min([s['duration'] for s in processed_samples]),
            'max_duration': np.max([s['duration'] for s in processed_samples]),
            'vocab_size': len(self.text_processor.char_to_idx)
        }
        
        stats_path = os.path.join(output_dir, 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDataset built successfully!")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Train: {stats['train_samples']}, Val: {stats['val_samples']}, Test: {stats['test_samples']}")
        print(f"Average duration: {stats['avg_duration']:.2f}s")
        print(f"Vocabulary size: {stats['vocab_size']}")


def main():
    parser = argparse.ArgumentParser(description='Prepare TTS dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing audio and transcripts')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed dataset')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build dataset
    builder = DatasetBuilder(config)
    builder.build_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
