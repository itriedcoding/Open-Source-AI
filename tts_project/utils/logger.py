"""
Training logger for Custom TTS Model.
Handles TensorBoard logging and checkpoint management.
"""

import os
from datetime import datetime
from typing import Optional, Union
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class TrainingLogger:
    """Logger for training metrics and visualization."""
    
    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.log_dir = log_dir
        
        if enabled and SummaryWriter is not None:
            # Create unique run name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"run_{timestamp}"
            full_log_dir = os.path.join(log_dir, run_name)
            
            self.writer = SummaryWriter(full_log_dir)
            print(f"TensorBoard logs will be saved to: {full_log_dir}")
        else:
            self.writer = None
            if enabled and SummaryWriter is None:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_value_dict: dict, step: int):
        """Log multiple scalar values."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_value_dict, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log a histogram of values."""
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int):
        """Log an image."""
        if self.enabled and self.writer is not None:
            # Ensure image is in correct format (C, H, W)
            if image.ndim == 2:
                image = image[np.newaxis, ...]
            elif image.ndim == 3 and image.shape[0] != 3:
                image = np.transpose(image, (2, 0, 1))
            
            self.writer.add_image(tag, image, step)
    
    def log_audio(self, tag: str, audio: np.ndarray, sample_rate: int, step: int):
        """Log audio."""
        if self.enabled and self.writer is not None:
            self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self.enabled and self.writer is not None:
            self.writer.add_text(tag, text, step)
    
    def close(self):
        """Close the logger."""
        if self.writer is not None:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
