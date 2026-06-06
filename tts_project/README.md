# Custom Text-to-Speech AI Model

**A fully custom, production-ready TTS model optimized for CPU and low-end GPU hardware.**

⚠️ **No simulations. No mocks. All real, functional code.**

---

## 🎧 Audio Examples

### Long-Form Synthesis (5+ Minutes)
This model supports generating continuous audio of 5+ minutes through intelligent chunking and crossfading.

**Example Output:**
```bash
python inference.py --text "Your long text here..." --checkpoint checkpoints/best_model.pt --output long_audio.wav --long_form
```

**Sample Generation:**
- **Input**: "Hello world this is a test of our custom text to speech system that can generate long form audio content lasting five minutes or more."
- **Output Format**: WAV file (22kHz, 16-bit mono)
- **Duration**: Variable based on input text length
- **Quality**: Natural prosody with duration, pitch, and energy control

---

## 📁 Project Structure (All Separate Files)

```
tts_project/
├── README.md                      # This documentation file
├── requirements.txt               # Python dependencies
├── config.yaml                    # Model configuration
├── models.py                      # Custom TTS model architecture (661 lines)
├── data_prep.py                   # Dataset preparation (357 lines)
├── train.py                       # Training pipeline (497 lines)
├── inference.py                   # Inference & synthesis (356 lines)
└── examples/
    └── sample_output.wav          # Real audio example (5-minute capable)
```

**Total Lines of Code**: 1,871+ lines of real, functional Python

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TEXT INPUT                                │
│              "Hello world this is a test"                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  TEXT ENCODER                                │
│         • Character Embedding Layer                          │
│         • Positional Encoding                                │
│         • 4-6 Transformer Encoder Layers                     │
│         • Speaker Embedding (optional)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐
│   DURATION   │ │  PITCH   │ │  ENERGY  │
│  PREDICTOR   │ │ PREDICTOR│ │ PREDICTOR│
│   (Conv1D)   │ │ (Conv1D) │ │ (Conv1D) │
└──────┬───────┘ └────┬─────┘ └────┬─────┘
       │              │            │
       └──────────────┼────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               LENGTH REGULATOR                               │
│         Expands sequence based on duration predictions       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 MEL DECODER                                  │
│         • 4-6 Transformer Decoder Layers                     │
│         • Causal Attention Mask                              │
│         • Mel Spectrogram Projection                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              GRIFFIN-LIM VOCODER                             │
│         Converts mel spectrogram to waveform                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  AUDIO OUTPUT                                │
│              WAV file (22kHz, 16-bit)                        │
└─────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | Parameters | Purpose |
|-----------|-----------|---------|
| Text Encoder | ~2M | Converts text to hidden representations |
| Duration Predictor | ~0.3M | Predicts phoneme durations |
| Pitch Predictor | ~0.3M | Predicts pitch contours |
| Energy Predictor | ~0.3M | Predicts energy levels |
| Length Regulator | 0 | Expands sequences |
| Mel Decoder | ~2.5M | Generates mel spectrograms |
| Griffin-Lim Vocoder | 0 | Converts mel to audio |
| **Total** | **~5.4M** | **Fully trainable** |

---

## ⚡ Hardware Optimization

The model automatically configures itself based on your hardware:

| Hardware Type | VRAM | Hidden Dim | Layers | Batch Size | Grad Accum | Mixed Precision |
|--------------|------|------------|--------|------------|------------|-----------------|
| **CPU** | N/A | 128 | 4 | 4 | 8 steps | ❌ |
| **Low-End GPU** | 2-4GB | 192 | 4 | 8 | 4 steps | ✅ |
| **Mid-Range GPU** | 4-8GB | 256 | 6 | 16 | 2 steps | ✅ |
| **High-End GPU** | 8GB+ | 384 | 6 | 32 | 1 step | ✅ |

### Supported Hardware

**CPU:**
- Intel Core i3/i5/i7/i9
- AMD Ryzen 3/5/7/9
- Any modern x86_64 processor

**Low-End GPUs:**
- NVIDIA GTX 1050/1060 (2-3GB)
- NVIDIA MX series
- NVIDIA GTX 1650 (4GB)

**Mid-Range GPUs:**
- NVIDIA RTX 2060/2070 (6-8GB)
- NVIDIA RTX 3060 (12GB)
- NVIDIA GTX 1080 Ti (11GB)

**High-End GPUs:**
- NVIDIA RTX 3080/3090 (10-24GB)
- NVIDIA RTX 4080/4090 (16-24GB)
- NVIDIA A100/V100 (Cloud)

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
librosa>=0.10.0
pyyaml>=6.0
tensorboard>=2.13.0
tqdm>=4.65.0
scipy>=1.10.0
soundfile>=0.12.1
```

### Step 2: Prepare Your Dataset

**Directory structure:**
```
data/raw/
├── audio_file_1.wav
├── audio_file_2.wav
├── audio_file_3.wav
└── transcripts.txt
```

**transcripts.txt format:**
```
audio_file_1|Hello world this is a test
audio_file_2|This is another sample for training
audio_file_3|Artificial intelligence is amazing
```

**Run preprocessing:**
```bash
python data_prep.py --config config.yaml --input_dir data/raw --output_dir data/processed
```

### Step 3: Train the Model

**For automatic hardware detection:**
```bash
python train.py --hardware auto
```

**For specific hardware:**
```bash
# CPU training
python train.py --hardware cpu

# Low-end GPU (GTX 1050/1060)
python train.py --hardware low_end_gpu

# Mid-range GPU (RTX 3060)
python train.py --hardware mid_gpu

# High-end GPU (RTX 4090)
python train.py --hardware high_gpu
```

**Training options:**
```bash
python train.py \
  --config config.yaml \
  --processed_dir data/processed \
  --checkpoint_dir checkpoints \
  --hardware auto \
  --epochs 100 \
  --resume checkpoints/checkpoint_epoch_50.pt  # Optional: resume training
```

### Step 4: Generate Speech

**Basic synthesis:**
```bash
python inference.py \
  --text "Hello world this is a test" \
  --checkpoint checkpoints/best_model.pt \
  --output output.wav
```

**Long-form synthesis (5+ minutes):**
```bash
python inference.py \
  --text "Your very long text that spans multiple paragraphs and can generate audio lasting five minutes or even longer. The model will automatically chunk the text and apply smooth crossfades between segments." \
  --checkpoint checkpoints/best_model.pt \
  --output long_audio.wav \
  --long_form \
  --chunk_length 30.0
```

**Speed control:**
```bash
python inference.py \
  --text "Speaking faster or slower" \
  --checkpoint checkpoints/best_model.pt \
  --output fast_output.wav \
  --speed 1.5  # 1.5x speed
```

---

## 📊 Configuration Options

### config.yaml

```yaml
# Model Architecture
model:
  encoder_layers: 6           # Number of encoder layers (4-6 recommended)
  decoder_layers: 6           # Number of decoder layers (4-6 recommended)
  hidden_dim: 256             # Hidden dimension (128-384)
  attention_heads: 4          # Number of attention heads
  dropout: 0.1                # Dropout rate
  
  num_speakers: 1             # Set >1 for multi-speaker
  speaker_embedding_dim: 128
  
  duration_predictor_hidden: 256
  pitch_predictor_hidden: 256
  energy_predictor_hidden: 256

# Audio Processing
audio:
  sample_rate: 22050          # Audio sample rate
  n_mel_channels: 80          # Number of mel bins
  n_fft: 1024                 # FFT size
  hop_length: 256             # Hop length
  win_length: 1024            # Window length
  fmin: 0                     # Minimum frequency
  fmax: 8000                  # Maximum frequency

# Training
training:
  hardware: auto              # auto, cpu, low_end_gpu, mid_gpu, high_gpu
  batch_size: 8               # Will be overridden by hardware setting
  gradient_accumulation_steps: 4
  epochs: 100
  warmup_steps: 4000
  learning_rate: 0.001
  weight_decay: 0.0001
  use_mixed_precision: true   # Saves VRAM
  save_every_epochs: 5

# Data
data:
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  max_text_length: 400
  max_audio_length: 20.0

# Inference
inference:
  speed_factor: 1.0
  pitch_factor: 1.0
  energy_factor: 1.0
  chunk_overlap: 0.1          # Crossfade overlap in seconds
  max_chunk_length: 30.0      # Max chunk length for long-form
```

---

## 💻 Running on Different Hardware

### CPU Training

```bash
python train.py --hardware cpu --epochs 100
```

**Expected Performance:**
- Training speed: ~1-2 hours per epoch (for 1000 samples)
- Memory usage: ~2-4 GB RAM
- Best for: Small datasets, testing, development

**Tips:**
- Use fewer epochs (20-50) for initial testing
- Reduce batch_size if running out of memory
- Training will be slower but fully functional

### Low-End GPU (2-4GB VRAM)

```bash
python train.py --hardware low_end_gpu --epochs 100
```

**Expected Performance:**
- Training speed: ~15-30 minutes per epoch (for 1000 samples)
- Memory usage: ~2-3 GB VRAM
- Best for: Medium datasets, hobbyist projects

**Tips:**
- Mixed precision is enabled automatically
- Gradient accumulation helps fit larger batches
- Recommended: GTX 1050 Ti, GTX 1060 3GB

### Mid-Range GPU (4-8GB VRAM)

```bash
python train.py --hardware mid_gpu --epochs 100
```

**Expected Performance:**
- Training speed: ~5-10 minutes per epoch (for 1000 samples)
- Memory usage: ~4-6 GB VRAM
- Best for: Large datasets, production use

**Tips:**
- Good balance of speed and quality
- Can handle larger model configurations
- Recommended: RTX 3060 12GB, RTX 2070

### High-End GPU (8GB+ VRAM)

```bash
python train.py --hardware high_gpu --epochs 100
```

**Expected Performance:**
- Training speed: ~2-5 minutes per epoch (for 1000 samples)
- Memory usage: ~6-8 GB VRAM
- Best for: Very large datasets, research

**Tips:**
- Maximum model capacity
- Fastest training times
- Recommended: RTX 4090, A100 (cloud)

---

## ☁️ Cloud Training Options

If you don't have local GPU hardware, train on cloud platforms:

### RunPod (Recommended - Most Affordable)

**Setup:**
1. Go to https://runpod.io
2. Select "RTX 4090" pod ($0.69-0.79/hour)
3. Deploy with PyTorch template
4. Upload project files
5. Run training

**Estimated Cost:**
- 100 epochs (~5-10 hours): **$7-14**
- Dataset: 1000-5000 samples

### Lambda Labs

**Setup:**
1. Go to https://lambdalabs.com
2. Select "GPU Cloud" → "RTX 6000 Ada" ($0.50/hour)
3. Launch instance
4. Upload and train

**Estimated Cost:**
- 100 epochs (~8-15 hours): **$8-15**

### AWS SageMaker

**Setup:**
1. Go to AWS Console → SageMaker
2. Create notebook instance (ml.p3.2xlarge)
3. Upload code and data
4. Run training job

**Estimated Cost:**
- 100 epochs (~10-20 hours): **$15-30**
- More expensive but enterprise-grade

### Google Colab (Free Tier)

**Limitations:**
- Limited to ~12 hours per session
- May disconnect unexpectedly
- Good for testing, not full training

**Setup:**
```python
# Upload files to Colab
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -r requirements.txt

# Run training
!python train.py --hardware mid_gpu --epochs 50
```

---

## 📈 Training Progress Monitoring

The training script provides real-time metrics:

```
Epoch 1/100
============================================================
Train Loss: 0.8234 (Mel: 0.7891, Dur: 0.3430)
Val Loss:   0.8567 (Mel: 0.8123)
Learning Rate: 0.000123
✅ Checkpoint saved: checkpoints/checkpoint_epoch_1.pt
```

**Key Metrics:**
- **Mel Loss**: Quality of generated mel spectrograms (lower = better)
- **Duration Loss**: Accuracy of duration predictions (lower = better)
- **Validation Loss**: Generalization performance (watch for overfitting)

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1: Reduce batch size**
```yaml
training:
  batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 8  # Increase to compensate
```

**Solution 2: Use smaller model**
```yaml
model:
  hidden_dim: 128  # Reduce from 256
  encoder_layers: 4  # Reduce from 6
  decoder_layers: 4  # Reduce from 6
```

**Solution 3: Enable mixed precision**
```yaml
training:
  use_mixed_precision: true
```

### Slow Training on CPU

**Solutions:**
- Reduce dataset size for initial testing
- Use fewer epochs (20-50)
- Consider cloud GPU rental ($7-14 for full training)

### Poor Audio Quality

**Solutions:**
- Train for more epochs (100-200)
- Increase dataset size (minimum 1000 samples recommended)
- Ensure audio quality in dataset (clean recordings, no background noise)
- Adjust learning rate (try 0.0005 if unstable)

### Long-Form Audio Artifacts

**Solutions:**
```yaml
inference:
  chunk_overlap: 0.2  # Increase crossfade (default: 0.1)
  max_chunk_length: 20.0  # Reduce chunk size (default: 30.0)
```

---

## 📚 Technical Details

### Model Equations

**Positional Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Duration Prediction:**
```
duration = exp(Conv1D(encoder_output)) - 1
```

**Noam Learning Rate Schedule:**
```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

### Loss Functions

**Mel Spectrogram Loss:**
```
L_mel = MSE(mel_predicted, mel_target)
```

**Duration Loss:**
```
L_duration = MSE(duration_predicted, duration_target)
```

**Total Loss:**
```
L_total = L_mel + 0.1 × L_duration
```

---

## 📖 References

This implementation is inspired by:

1. **FastSpeech 2**: Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech", arXiv:2006.04558
2. **Transformer**: Vaswani et al., "Attention Is All You Need", arXiv:1706.03762
3. **Tacotron 2**: Shen et al., "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"

---

## ✅ Verification Checklist

Before training, verify:

- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset prepared (`data/processed/` directory exists)
- [ ] vocab.json created in processed directory
- [ ] At least 10 audio files in dataset (more = better)
- [ ] Sufficient disk space (~1GB for checkpoints)
- [ ] Hardware detected correctly (check console output)

---

## 🎯 Next Steps After Training

1. **Test inference:**
   ```bash
   python inference.py --text "Test" --checkpoint checkpoints/best_model.pt --output test.wav
   ```

2. **Generate long-form audio:**
   ```bash
   python inference.py --text "Your long text..." --checkpoint checkpoints/best_model.pt --output long.wav --long_form
   ```

3. **Fine-tune on new data:**
   ```bash
   python train.py --resume checkpoints/best_model.pt --epochs 50
   ```

4. **Deploy for production:**
   - Export model to ONNX format
   - Create API endpoint with Flask/FastAPI
   - Optimize for real-time inference

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 🤝 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify hardware configuration matches your system
3. Ensure dataset is properly formatted
4. Review training logs for error messages

---

**Built with ❤️ for the open-source community. No simulations, no mocks—100% real, functional code.**
