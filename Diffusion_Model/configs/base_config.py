from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    train_steps_by_epoch = 10 # Steps by epoch
    eval_batch_size = 16  # how many images to sample during evaluation
    num_train_timesteps=5#1000 # for noise scheduler
    num_inference_steps = 2#1000
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    seed = 0

config = TrainingConfig()