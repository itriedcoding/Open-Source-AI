#!/bin/bash

# Cloud Training Setup Script for Custom TTS Model
# This script helps you set up and train the model on cloud platforms

set -e

echo "========================================="
echo "Custom TTS Model - Cloud Training Setup"
echo "========================================="

# Check if running on a cloud platform with GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠ No NVIDIA GPU detected. Training will use CPU (slow)."
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/raw/audio
mkdir -p data/processed/features
mkdir -p checkpoints
mkdir -p logs/tensorboard
mkdir -p output

# Generate sample data if no dataset exists
if [ ! -f "data/raw/transcripts.txt" ]; then
    echo ""
    echo "No dataset found. Generating sample data for testing..."
    python data/generate_sample_data.py --output_dir data/raw --num_samples 20
fi

# Prepare dataset
echo ""
echo "Preparing dataset..."
python data/prepare_dataset.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --config configs/model_config.yaml

# Show dataset statistics
if [ -f "data/processed/dataset_stats.json" ]; then
    echo ""
    echo "Dataset Statistics:"
    cat data/processed/dataset_stats.json | python -m json.tool
fi

# Start training
echo ""
echo "========================================="
echo "Starting Training"
echo "========================================="
echo ""
echo "To monitor training progress, run:"
echo "  tensorboard --logdir logs/tensorboard --host 0.0.0.0"
echo ""

python train.py \
    --config configs/model_config.yaml \
    --data_dir data/processed \
    --device cuda

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "Model checkpoints saved to: checkpoints/"
echo "TensorBoard logs saved to: logs/tensorboard/"
echo ""
echo "To generate speech with your trained model:"
echo "  python infer.py --text \"Your text here\" --checkpoint_path checkpoints/best_model.pt --output output.wav"
