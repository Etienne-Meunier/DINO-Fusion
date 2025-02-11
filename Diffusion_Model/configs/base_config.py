from dataclasses import dataclass, field
from typing import List, Dict
import os


@dataclass
class BaseConfig:
    data_file : str = os.environ['OCEANDATA']+'/Dino-Fusion/dino_1_4_degree_coarse_240125.tar'
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 0 #500
    mixed_precision: str = "no" # "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed: int = 0
    base_output_dir : str = "../../diffModel_experiences" # Base dir for output model
    lr_schedule: str = 'cosine_schedule'
    st_path: str | None = None
    generation_frequency : int = 50
    pretrained_model_path : str | None = None

@dataclass
class TrainingConfig(BaseConfig):
    train_batch_size: int = 8
    train_steps_by_epoch: int = 200  # Steps by epoch
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_train_timesteps: int = 1000  #for noise scheduler
    num_inference_steps: int = 1000  #for noise scheduler
    num_epochs: int = 5000
    output_dir: str = "wandb"  #  wandb means the directory will be named with the id of the run
    logger = 'wandb'
    fields : Dict = field(default_factory=lambda: ({'toce' : slice(0, -1, 2), 'soce' : slice(0, -1, 2), 'ssh' : slice(0, 1)}))
    normalisation : str = 'min-max'

@dataclass
class SSHTrainingConfig(TrainingConfig):
    fields : Dict = field(default_factory=lambda: ({'ssh' : slice(0, 1)}))

@dataclass
class TOCETrainingConfig(TrainingConfig):
    fields : Dict = field(default_factory=lambda: ({'toce' : slice(0, -1, 2)}))

@dataclass
class SOCETrainingConfig(TrainingConfig):
    fields : Dict = field(default_factory=lambda: ({'soce' : slice(0, -1, 2)}))

@dataclass
class FineTuningConfig(TrainingConfig) :
    st_path : str= "../../diffModel_experiences/vh9dn5be/"
