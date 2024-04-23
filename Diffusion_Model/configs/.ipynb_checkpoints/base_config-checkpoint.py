from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    image_size: List = field(default_factory=lambda: [200, 64])  # the generated image resolution
    train_batch_size: int = 16
    train_steps_by_epoch: int = 10  # Steps by epoch
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_train_timesteps: int = 1000  #for noise scheduler
    num_inference_steps: int = 1000  #for noise scheduler
    num_epochs: int = 1 #50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = 'no'  # "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    seed: int = 0