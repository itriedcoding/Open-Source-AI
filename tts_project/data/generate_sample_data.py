"""
Sample dataset generator for testing the TTS pipeline.
Creates synthetic audio files and transcripts for demonstration.
"""

import os
import argparse
import numpy as np
import soundfile as sf
import yaml


def generate_sample_audio(duration: float, sample_rate: int, filename: str):
    """Generate a simple synthetic audio file."""
    # Create a simple tone with harmonics
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Fundamental frequency (varies to simulate speech-like patterns)
    f0 = 200 + 50 * np.sin(2 * np.pi * 0.5 * t)
    
    # Generate waveform with multiple harmonics
    audio = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.2 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    
    # Add some noise
    audio += 0.1 * np.random.randn(len(audio))
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Save
    sf.write(filename, audio, sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Generate sample dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory for sample data')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of sample audio files to generate')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    sample_rate = config['audio']['sample_rate']
    
    # Create output directories
    audio_dir = os.path.join(args.output_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    
    # Sample texts for demonstration
    sample_texts = [
        "Hello, this is a test of my custom text to speech model.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Welcome to the future of voice synthesis.",
        "This is a sample sentence for training purposes.",
        "Machine learning enables computers to learn from data.",
        "Speech synthesis has many practical applications.",
        "Thank you for using our text to speech system.",
        "Custom voices can be created with enough training data.",
        "The technology behind modern TTS is truly remarkable."
    ]
    
    # Generate sample audio files and transcripts
    transcript_lines = []
    
    print(f"Generating {args.num_samples} sample audio files...")
    
    for i in range(args.num_samples):
        audio_id = f"speaker1_{i+1:03d}"
        text = sample_texts[i % len(sample_texts)]
        
        # Generate audio file
        audio_path = os.path.join(audio_dir, f"{audio_id}.wav")
        duration = 2.0 + (i % 3) * 0.5  # Vary duration between 2-3 seconds
        generate_sample_audio(duration, sample_rate, audio_path)
        
        # Add to transcripts
        transcript_lines.append(f"{audio_id}|{text}")
        
        print(f"  Generated {audio_id}.wav ({duration:.1f}s)")
    
    # Write transcript file
    transcript_path = os.path.join(args.output_dir, 'transcripts.txt')
    with open(transcript_path, 'w') as f:
        f.write('\n'.join(transcript_lines))
    
    print(f"\nSample dataset created successfully!")
    print(f"Audio files: {audio_dir}")
    print(f"Transcripts: {transcript_path}")
    print(f"\nNext steps:")
    print(f"1. Run: python data/prepare_dataset.py --input_dir {args.output_dir} --output_dir data/processed")
    print(f"2. Run: python train.py --config configs/model_config.yaml")


if __name__ == '__main__':
    main()
