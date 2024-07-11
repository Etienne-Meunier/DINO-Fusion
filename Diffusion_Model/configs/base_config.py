from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BaseConfig:
    #check https://huggingface.co/docs/accelerate/concept_guides/performance
    data_file : str = '/home/meunier/Data/Dino-Fusion/dino_1_4_degree.tar'#'/home/tissot/data/dataset2.tar'
    #image_size: List = field(default_factory=lambda: [800, 248])  # the generated image resolution
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 0 #500
    mixed_precision: str = "fp16" # "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed: int = 0
    use_ema: bool = False
    base_output_dir : str = "../../diffModel_experiences/" # Base dir for output model
    lr_schedule = 'linear'
    st_path = None
    



@dataclass
class TrainingConfig(BaseConfig):
    train_batch_size: int = 8 #!!!!!!! this is batch size per GPU actually, so if 4 GPU, this is equivalent to using 32 as batch size
    train_steps_by_epoch: int = 200  # Steps by epoch
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_train_timesteps: int = 1000  #for noise scheduler
    num_inference_steps: int = 1000  #for noise scheduler
    num_epochs: int = 1000
    output_dir: str = "wandb"  #  wandb means the directory will be named with the id of the run 
    logger = 'wandb'
    fields : Dict = field(default_factory=lambda: ({'soce' : slice(0, -1, 5), 'toce' : slice(0, -1, 5), 'ssh' : slice(0, 1)}))

@dataclass
class SSHTrainingConfig(TrainingConfig):
    fields : Dict = field(default_factory=lambda: ({'ssh' : slice(0, 1)}))



@dataclass 
class FineTuningConfig(TrainingConfig) : 
    st_path : str= "../../diffModel_experiences/vh9dn5be/"

@dataclass
class MiniConfig(BaseConfig):
    data_file : str = '/Users/emeunier/Documents/scai/mini_dataset2.tar'
    train_batch_size: int = 16 #!!!!!!! this is batch size per GPU actually, so if 4 GPU, this is equivalent to using 32 as batch size
    train_steps_by_epoch: int = 20  # Steps by epoch
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_train_timesteps: int = 100  #for noise scheduler
    num_inference_steps: int = 100  #for noise scheduler
    num_epochs: int = 1000
    output_dir: str = "wandb"  #  wandb means the directory will be named with the id of the run 
    logger = 'wandb'
    mixed_precision: str = "no" # "fp16"  # `no` for float32, `fp16` for automatic mixed precision

@dataclass
class DevConfig(BaseConfig):
    train_batch_size: int = 3 #!!!!!!! this is batch size per GPU actually, so if 4 GPU, this is equivalent to using 32 as batch size
    train_steps_by_epoch: int = 20  # Steps by epoch
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_train_timesteps: int = 1000  #for noise scheduler
    num_inference_steps: int = 1000  #for noise scheduler
    num_epochs: int = 1000
    mixed_precision: str = "no" # "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    logger=None
    output_dir: str = "dev"  #  wandb means the directory will be named with the id of the run 
