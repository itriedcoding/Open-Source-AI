# Cloud Training Guide for Custom TTS Model

This guide explains how to train your custom TTS model on various cloud platforms.

## Prerequisites

Before starting, ensure you have:
1. A cloud platform account (RunPod, Lambda Labs, AWS, etc.)
2. Your audio dataset prepared in the correct format
3. Basic understanding of command-line operations

## Recommended Cloud Platforms

### 1. RunPod (Recommended for Cost-Effectiveness)

**Pricing**: Starting at $0.45/hour for RTX 3090

**Setup Steps**:
1. Go to [runpod.io](https://runpod.io) and create an account
2. Click "Deploy" → "New Pod"
3. Select template: "PyTorch 2.0 + CUDA 11.8"
4. Choose GPU: RTX 4090 (24GB) or A100 (40/80GB) recommended
5. Set storage: Minimum 50GB
6. Deploy and connect via SSH or Jupyter

**Training Commands**:
```bash
# Clone your repository or upload files
git clone <your-repo-url>
cd tts_project

# Install dependencies
pip install -r requirements.txt

# Upload your dataset to data/raw/
# Then run the setup script
bash setup_cloud_training.sh
```

### 2. Lambda Labs

**Pricing**: Starting at $0.50/hour for RTX A6000

**Setup Steps**:
1. Go to [lambdalabs.com](https://lambdalabs.com)
2. Create account and navigate to "Cloud Instances"
3. Launch instance with "PyTorch 2.0" template
4. Select GPU: RTX A6000 or A100
5. Connect via SSH

**Training Commands**:
```bash
# Same as RunPod
bash setup_cloud_training.sh
```

### 3. Google Colab Pro/Pro+

**Pricing**: $10-50/month

**Setup Steps**:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload this project to Google Drive
3. Open notebook in Colab
4. Runtime → Change runtime type → GPU (V100/A100 with Pro+)

**Note**: Limited session duration (up to 24 hours), good for experimentation

### 4. AWS SageMaker

**Pricing**: Pay per use, varies by instance type

**Setup Steps**:
1. Go to AWS Console → SageMaker
2. Create Notebook Instance
3. Choose instance type: ml.p3.2xlarge (V100) or ml.p4d.24xlarge (A100)
4. Upload code and data to S3
5. Run training job

## Dataset Preparation

### Format Requirements

Your dataset should be organized as follows:

```
data/raw/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── transcripts.txt
```

### Transcript File Format

Each line in `transcripts.txt`:
```
audio_id|Transcript text here
```

Example:
```
sample_001|Hello, this is a test.
sample_002|Welcome to my custom TTS system.
```

### Audio Requirements

- **Format**: WAV, FLAC, or MP3
- **Sample Rate**: Will be converted to 22050 Hz automatically
- **Duration**: Between 0.5 and 15 seconds per clip
- **Quality**: Clean recordings with minimal background noise

### Recommended Dataset Sizes

- **Minimum**: 100 samples (basic functionality)
- **Good**: 1,000-5,000 samples (decent quality)
- **Excellent**: 10,000+ samples (high quality)

## Training Process

### Step 1: Upload Dataset

Upload your dataset to the cloud instance:

```bash
# Using scp
scp -r data/raw/ user@instance_ip:/path/to/tts_project/data/

# Or using rsync
rsync -avz data/raw/ user@instance_ip:/path/to/tts_project/data/
```

### Step 2: Prepare Dataset

```bash
python data/prepare_dataset.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --config configs/model_config.yaml
```

### Step 3: Start Training

```bash
python train.py \
    --config configs/model_config.yaml \
    --data_dir data/processed \
    --device cuda
```

### Step 4: Monitor Training

Open another terminal and run TensorBoard:

```bash
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006
```

Then access http://your-instance-ip:6006 in your browser.

### Step 5: Generate Speech

After training completes:

```bash
python infer.py \
    --text "Your custom text here" \
    --checkpoint_path checkpoints/best_model.pt \
    --output output.wav
```

## Configuration Tuning

### For Limited GPU Memory (< 16GB)

Edit `configs/model_config.yaml`:

```yaml
training:
  batch_size: 8  # Reduce from 32
  gradient_accumulation_steps: 4  # Increase to compensate
  
model:
  text_encoder:
    num_layers: 4  # Reduce from 6
    embedding_dim: 256  # Reduce from 512
```

### For Better Quality (More VRAM)

```yaml
training:
  batch_size: 64  # Increase if you have 80GB VRAM
  
model:
  text_encoder:
    num_layers: 8  # More layers
    num_heads: 16  # More attention heads
```

## Cost Optimization Tips

1. **Use Spot Instances**: Save up to 70% on cloud costs
2. **Pre-process Locally**: Prepare dataset before uploading
3. **Monitor Training**: Stop early if loss plateaus
4. **Choose Right GPU**: RTX 4090 offers best price/performance
5. **Save Checkpoints Frequently**: Avoid losing progress

## Estimated Training Times

| Dataset Size | GPU | Estimated Time | Cost (RunPod) |
|-------------|-----|----------------|---------------|
| 100 samples | RTX 4090 | 1-2 hours | $0.45-0.90 |
| 1,000 samples | RTX 4090 | 10-15 hours | $4.50-6.75 |
| 5,000 samples | A100 40GB | 20-30 hours | $10-15 |
| 10,000 samples | A100 80GB | 40-60 hours | $20-30 |

## Troubleshooting

### Out of Memory Error

Reduce batch size in config:
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4
```

### Slow Training

- Ensure you're using GPU: `nvidia-smi`
- Enable mixed precision: Already enabled in config
- Use faster GPU (A100 > RTX 4090 > RTX 3090)

### Poor Audio Quality

- Train for more epochs
- Increase dataset size
- Ensure clean audio recordings
- Adjust learning rate

## Next Steps After Training

1. **Test Thoroughly**: Generate various texts
2. **Fine-tune**: Continue training on specific voice characteristics
3. **Deploy**: Use the inference script in production
4. **Share**: Export model for others to use

## Support

For issues or questions:
1. Check the README.md
2. Review error messages carefully
3. Monitor TensorBoard for training issues
4. Adjust hyperparameters based on results
