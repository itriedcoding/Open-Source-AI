# Custom Text-to-Audio AI Model

A fully custom text-to-speech (TTS) model implementation with dataset preparation, training, and inference pipelines.

## Project Structure

```
tts_project/
├── configs/              # Configuration files
├── data/                 # Dataset preparation scripts
├── models/               # Model architecture definitions
├── utils/                # Utility functions
├── train.py              # Main training script
├── infer.py              # Inference script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Features

- Custom transformer-based TTS architecture
- Comprehensive dataset preparation pipeline
- Multi-speaker support
- Mixed precision training support
- Distributed training ready
- Real-time inference capabilities

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Place your audio files and transcripts in the `data/raw/` directory, then run:

```bash
python data/prepare_dataset.py --input_dir data/raw --output_dir data/processed
```

### 3. Train the Model

```bash
python train.py --config configs/model_config.yaml
```

### 4. Generate Speech

```bash
python infer.py --text "Hello, this is my custom TTS model!" --checkpoint_path checkpoints/best_model.pt
```

## Training on Cloud Platforms

Recommended platforms for training:
- **RunPod**: Cost-effective GPU instances
- **Lambda Labs**: High-performance GPUs
- **AWS SageMaker**: Scalable infrastructure
- **Google Colab Pro**: Good for experimentation

### Example RunPod Setup

1. Create a pod with RTX 4090 or A100 GPU
2. Clone this repository
3. Upload your dataset
4. Run the training script

## Model Architecture

The model uses a transformer-based encoder-decoder architecture with:
- Text encoder with positional embeddings
- Acoustic feature predictor
- Neural vocoder integration
- Attention mechanisms for alignment

## Dataset Format

Expected format for training data:
```
data/raw/
├── audio/
│   ├── speaker1_001.wav
│   ├── speaker1_002.wav
│   └── ...
└── transcripts.txt
```

Where `transcripts.txt` contains:
```
speaker1_001|This is the transcript for the first audio file.
speaker1_002|This is the transcript for the second audio file.
```

## License

MIT License
