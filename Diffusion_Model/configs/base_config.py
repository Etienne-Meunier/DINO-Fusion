from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    data_file : str = '../../normalized_data.npy'
    image_size: List = field(default_factory=lambda: [200, 64])  # the generated image resolution
    train_batch_size: int = 16
    train_steps_by_epoch: int = 10  # Steps by epoch
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_train_timesteps: int = 1000  #for noise scheduler
    num_inference_steps: int = 1000  #for noise scheduler
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 0 #500
    mixed_precision: str = 'no'  # "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "wandb"  #  wandb means the directory will be named with the id of the run 
    seed: int = 0


@dataclass
class MiniConfig:
    data_file : str = '../../mini_data.npy'
    image_size: List = field(default_factory=lambda: [200, 64])  # the generated image resolution
    train_batch_size: int = 2# 16
    train_steps_by_epoch: int = 10  # Steps by epoch
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_train_timesteps: int = 3#1000  #for noise scheduler
    num_inference_steps: int = 3#1000  #for noise scheduler
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 0 #500
    mixed_precision: str = 'no'  # "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "wandb"  #  wandb means the directory will be named with the id of the run 
    seed: int = 0