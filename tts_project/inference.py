"""
Inference Script for Custom TTS Model
Generate speech from text with support for long-form synthesis (5+ minutes)
Includes Griffin-Lim vocoder for mel-to-audio conversion
"""

import torch
import numpy as np
import yaml
import argparse
import json
from pathlib import Path
import soundfile as sf
from models import CustomTTSModel


class GriffinLimVocoder:
    """Griffin-Lim algorithm for mel spectrogram to audio conversion"""
    
    def __init__(self, config):
        audio_cfg = config.get('audio', {})
        
        self.sample_rate = audio_cfg.get('sample_rate', 22050)
        self.n_mel_channels = audio_cfg.get('n_mel_channels', 80)
        self.n_fft = audio_cfg.get('n_fft', 1024)
        self.hop_length = audio_cfg.get('hop_length', 256)
        self.win_length = audio_cfg.get('win_length', 1024)
        self.fmin = audio_cfg.get('fmin', 0)
        self.fmax = audio_cfg.get('fmax', 8000)
        
        # Pre-compute mel filterbank
        import librosa
        self.mel_basis = torch.from_numpy(
            librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft,
                               n_mels=self.n_mel_channels, fmin=self.fmin, fmax=self.fmax)
        ).float()
        
        # Inverse mel basis (pseudo-inverse)
        self.inv_mel_basis = torch.pinverse(self.mel_basis)
    
    def mel_to_spectrogram(self, mel_spec):
        """Convert mel spectrogram to linear spectrogram"""
        # Convert to exponential scale
        mel_spec_exp = mel_spec.exp()
        
        # Convert to linear spectrogram
        spec = torch.matmul(self.inv_mel_basis.to(mel_spec_exp.device), mel_spec_exp)
        spec = torch.clamp(spec, min=1e-8)
        
        return spec
    
    def griffin_lim(self, mel_spec, n_iters=30):
        """
        Convert mel spectrogram to waveform using Griffin-Lim algorithm
        
        Args:
            mel_spec: (time, n_mels) or (batch, time, n_mels)
            n_iters: number of iterations
        
        Returns:
            waveform: (time,) or (batch, time)
        """
        was_batch = mel_spec.dim() == 3
        
        if not was_batch:
            mel_spec = mel_spec.unsqueeze(0)
        
        batch_size, num_frames, _ = mel_spec.shape
        
        waveforms = []
        
        for i in range(batch_size):
            single_mel = mel_spec[i]  # (time, n_mels)
            
            # Convert to spectrogram magnitude
            spec = self.mel_to_spectrogram(single_mel.T)  # (n_freq, time)
            spec = spec.cpu().numpy()
            
            # Initialize phase randomly
            angles = np.exp(2j * np.pi * np.random.rand(*spec.shape))
            
            # Griffin-Lim iterations
            for _ in range(n_iters):
                # Reconstruct complex spectrogram
                complex_spec = spec * angles
                
                # Inverse STFT
                waveform = self._istft(complex_spec)
                
                # Forward STFT
                _, angles = self._stft(waveform)
            
            waveforms.append(waveform)
        
        waveforms = np.stack(waveforms)
        
        if not was_batch:
            waveforms = waveforms[0]
        
        return waveforms
    
    def _stft(self, waveform):
        """Short-time Fourier transform"""
        import librosa
        D = librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True
        )
        return np.abs(D), np.exp(1j * np.angle(D))
    
    def _istft(self, complex_spec):
        """Inverse short-time Fourier transform"""
        import librosa
        waveform = librosa.istft(
            complex_spec,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True
        )
        return waveform


class TextToSpeech:
    """Main TTS inference class"""
    
    def __init__(self, checkpoint_path, config_path='config.yaml'):
        # Load configuration
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Try loading config from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.config = checkpoint.get('config', {})
        
        # Load vocabulary
        vocab_file = Path('data/processed/vocab.json')
        if vocab_file.exists():
            with open(vocab_file, 'r') as f:
                vocab_info = json.load(f)
            self.char_to_idx = vocab_info['char_to_idx']
            self.idx_to_char = vocab_info['idx_to_char']
        else:
            # Default vocabulary
            chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            digits = list('0123456789')
            punctuation = list('.,!?;:\'"()- ')
            symbols = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
            vocab = symbols + chars + digits + punctuation
            
            self.char_to_idx = {c: i for i, c in enumerate(vocab)}
            self.idx_to_char = {i: c for i, c in enumerate(vocab)}
        
        # Initialize model
        self.model = CustomTTSModel(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize vocoder
        self.vocoder = GriffinLimVocoder(self.config)
        
        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("💻 Using CPU")
    
    def encode_text(self, text):
        """Convert text to token indices"""
        tokens = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in text]
        tokens = [self.char_to_idx['<SOS>']] + tokens + [self.char_to_idx['<EOS>']]
        return torch.tensor(tokens, dtype=torch.long)
    
    def synthesize(self, text, speed_factor=1.0, pitch_factor=1.0, energy_factor=1.0):
        """
        Synthesize speech from text
        
        Args:
            text: input text string
            speed_factor: speech speed multiplier (1.0 = normal)
            pitch_factor: pitch multiplier (1.0 = normal)
            energy_factor: energy multiplier (1.0 = normal)
        
        Returns:
            waveform: numpy array of audio samples
            mel_spec: generated mel spectrogram
        """
        # Encode text
        tokens = self.encode_text(text)
        tokens = tokens.unsqueeze(0).to(self.device)
        token_lengths = torch.tensor([tokens.shape[1]], device=self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_output, mel_mask, output_lengths = self.model.infer(
                tokens, token_lengths, speed_factor=speed_factor
            )
        
        # Remove padding
        mel_output = mel_output[0][:output_lengths[0].item()]
        mel_output = mel_output.cpu()
        
        # Convert to waveform
        waveform = self.vocoder.griffin_lim(mel_output)
        
        return waveform, mel_output
    
    def synthesize_long_form(self, text, max_chunk_length=30.0, overlap=0.1):
        """
        Synthesize long-form text (5+ minutes) by chunking
        
        Args:
            text: long input text
            max_chunk_length: maximum chunk length in seconds
            overlap: overlap between chunks in seconds for crossfade
        
        Returns:
            waveform: concatenated audio with crossfades
        """
        # Split text into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Estimate characters per second (rough approximation)
        chars_per_second = 15
        
        for sentence in sentences:
            sentence_length = len(sentence) / chars_per_second
            
            if current_length + sentence_length > max_chunk_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"Split text into {len(chunks)} chunks")
        
        # Synthesize each chunk
        all_waveforms = []
        sample_rate = self.vocoder.sample_rate
        overlap_samples = int(overlap * sample_rate)
        
        for i, chunk in enumerate(chunks):
            print(f"Synthesizing chunk {i+1}/{len(chunks)}...")
            waveform, _ = self.synthesize(chunk)
            
            if i > 0 and len(all_waveforms[-1]) > overlap_samples:
                # Crossfade with previous chunk
                prev_waveform = all_waveforms[-1]
                
                # Create crossfade envelope
                fade_out = np.linspace(1, 0, overlap_samples)
                fade_in = np.linspace(0, 1, overlap_samples)
                
                # Apply crossfade
                crossfaded = (prev_waveform[-overlap_samples:] * fade_out + 
                             waveform[:overlap_samples] * fade_in)
                
                # Combine
                combined = np.concatenate([
                    prev_waveform[:-overlap_samples],
                    crossfaded,
                    waveform[overlap_samples:]
                ])
                
                all_waveforms[-1] = combined
            else:
                all_waveforms.append(waveform)
        
        # Concatenate all chunks
        final_waveform = np.concatenate(all_waveforms)
        
        return final_waveform
    
    def save_audio(self, waveform, output_path, sample_rate=None):
        """Save waveform to WAV file"""
        if sample_rate is None:
            sample_rate = self.vocoder.sample_rate
        
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        sf.write(output_path, waveform, sample_rate)
        print(f"✅ Audio saved to: {output_path}")
        
        # Print duration
        duration = len(waveform) / sample_rate
        print(f"   Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")


def main():
    parser = argparse.ArgumentParser(description='Generate speech from text')
    parser.add_argument('--text', type=str, required=True,
                       help='Text to synthesize')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file path')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech speed factor')
    parser.add_argument('--long_form', action='store_true',
                       help='Enable long-form synthesis for 5+ minute audio')
    parser.add_argument('--chunk_length', type=float, default=30.0,
                       help='Maximum chunk length in seconds for long-form')
    
    args = parser.parse_args()
    
    # Initialize TTS
    print("\n🎙️  Loading TTS model...")
    tts = TextToSpeech(args.checkpoint, args.config)
    
    # Generate speech
    print(f"\n📝 Synthesizing: \"{args.text[:100]}{'...' if len(args.text) > 100 else ''}\"")
    
    if args.long_form:
        waveform = tts.synthesize_long_form(
            args.text,
            max_chunk_length=args.chunk_length
        )
    else:
        waveform, _ = tts.synthesize(args.text, speed_factor=args.speed)
    
    # Save audio
    tts.save_audio(waveform, args.output)
    
    print("\n✨ Done!")


if __name__ == '__main__':
    main()
