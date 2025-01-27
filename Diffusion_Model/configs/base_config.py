from dataclasses import dataclass, field
from typing import List, Dict
import os


@dataclass
class BaseConfig:
    #check https://huggingface.co/docs/accelerate/concept_guides/performance
    data_file : str = os.environ['OCEANDATA']+'/Dino-Fusion/dino_1_4_degree_coarse_240125.tar' #'/home/meunier/Data/Dino-Fusion/dino_1_4_degree.tar',
    # /lustre/fswork/projects/rech/omr/ufk69pe/
    #image_size: List = field(default_factory=lambda: [800, 248])  # the generated image resolution
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 0 #500
    mixed_precision: str = "no" # "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed: int = 0
    use_LDM: bool = False # Latent diffusion Model, needs pretrained weights for VAEs, also needs to set the number of input/output channels of the UNET
    use_ema: bool = False
    base_output_dir : str = "../../diffModel_experiences" # Base dir for output model
    lr_schedule = 'cosine_schedule'
    st_path = None
    generation_frequency : int = 50

@dataclass
class TrainingConfig(BaseConfig):
    train_batch_size: int = 8 #!!!!!!! this is batch size per GPU actually, so if 4 GPU, this is equivalent to using 8*4 = 32 as batch size
    train_steps_by_epoch: int = 200  # Steps by epoch
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_train_timesteps: int = 1000  #for noise scheduler
    num_inference_steps: int = 1000  #for noise scheduler
    num_epochs: int = 1000
    output_dir: str = "wandb"  #  wandb means the directory will be named with the id of the run
    logger = 'wandb'
    fields : Dict = field(default_factory=lambda: ({'toce' : slice(0, -1, 2), 'soce' : slice(0, -1, 2), 'ssh' : slice(0, 1)}))
    normalisation : str = 'global-min-max'

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

@dataclass
class MiniConfig(TrainingConfig):
    train_steps_by_epoch: int = 20  # Steps by epoch
    num_train_timesteps: int = 100  #for noise scheduler
    num_inference_steps: int = 100  #for noise scheduler
    num_epochs: int = 1000
    output_dir: str = "wandb"  #  wandb means the directory will be named with the id of the run
    logger = 'wandb'
    mixed_precision: str = "no" # "fp16"  # `no` for float32, `fp16` for automatic mixed precision

@dataclass
class DevConfig(TrainingConfig):
    train_batch_size: int = 3 #!!!!!!! this is batch size per GPU actually, so if 4 GPU, this is equivalent to using 32 as batch size
    train_steps_by_epoch: int = 20  # Steps by epoch
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_train_timesteps: int = 100  #for noise scheduler
    num_inference_steps: int = 100  #for noise scheduler
    num_epochs: int = 100
    mixed_precision: str = "no" # "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    logger=None
    output_dir: str = "dev"  #  wandb means the directory will be named with the id of the run
    generation_frequency : int = 1
