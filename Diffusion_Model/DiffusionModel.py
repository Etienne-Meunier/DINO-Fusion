from diffusers import DDPMScheduler, DDPMPipeline
import torch.nn as nn
import torch
import torch.nn.functional as F
from backbones.backbones import get_simple_unet
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import os

class DiffusionModel(nn.Module) : 
    
    def __init__(self, config) : 
        super().__init__()
        self.config = config
        self.denoiser = get_simple_unet(self.config.image_size)
        self.noise_scheduler = DDPMScheduler(self.config.num_train_timesteps, beta_schedule="squaredcos_cap_v2")
        self.optimizer, self.lr_scheduler = self.config_optimizer()
        self.accelerator = self.config_accelerate()
        
    def config_accelerate(self) :
        os.makedirs(self.config.output_dir, exist_ok=True) 
        accelerator = Accelerator(mixed_precision=self.config.mixed_precision,
                                  gradient_accumulation_steps=self.config.gradient_accumulation_steps, 
                                  log_with='wandb',
                                  project_dir=os.path.join(self.config.output_dir, 'logs'))
        if accelerator.is_main_process:
            accelerator.init_trackers('dino-fusion', config=vars(self.config))
        return accelerator

    def config_optimizer(self) : 
        """
        Setup optimizer and scheduler
        Return 
            optimizer
            lr_scheduler
        """

        optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
                            optimizer=optimizer,
                            num_warmup_steps=self.config.lr_warmup_steps,
                            num_training_steps=(self.config.train_steps_by_epoch * self.config.num_epochs),
                        )
        return optimizer, lr_scheduler

    def denoise(self, noisy_images, timesteps) :
        """
            Takes as input a tensor of noisy images and a set of timesteps. 
            return the denoised image. 
            noisy_images (b, c, i, j) : noisy image
            timesteps (b,) : timestep for each batch
        """
        assert noisy_images.shape[0] == timesteps.shape[0], f'ShapeError : {noisy_images.shape[0]} != {timesteps.shape[0]}'
        return self.denoiser(noisy_images, timesteps, return_dict=False)[0]
    
    def get_noisy_images(self, clean_images) : 
        """
        Add noise to images using the scheduler 
        Params : 
            clean_images (b, c, i , j)
            noises (b, c, i , j)
        Returns : 
            noisy_images (b, c, i , j)
            noises (b, c, i, j)
            timesteps (b, ) 
        """
        noises = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]
        timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bs,), device=clean_images.device,
                dtype=torch.int64
            )
        noisy_images = self.noise_scheduler.add_noise(clean_images, noises, timesteps)
        return noisy_images, noises, timesteps
    
    def training_step(self, batch, mask) :
        """
        Takes a batch of images and do a training step.
        """
        clean_images = batch#["images"]
        mask_tensor = torch.from_numpy(mask).expand(clean_images.shape).type(torch.float).to(self.accelerator.device)
        ### pad one line at the bottom to make size number of lines 200 a multiple of 4
        clean_images = F.pad(clean_images, (1, 1, 0, 1), "constant", 0)
        mask_tensor = F.pad(mask_tensor, (1, 1, 0, 1), "constant", 0)
        noisy_images, noises, timesteps = self.get_noisy_images(clean_images)

        with self.accelerator.accumulate(self.denoiser) :
            noises_pred = self.denoiser(noisy_images, timesteps, return_dict=False)[0]

            loss = F.mse_loss(mask_tensor*noises_pred, mask_tensor*noises)
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.denoiser.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss.detach().item()

    def test_step(self) : 
        """
        Generate a list of images `List[PIL.Image]` from noise
        """
        pipeline = DDPMPipeline(unet=self.accelerator.unwrap_model(self.denoiser), scheduler=self.noise_scheduler)
        images = pipeline(
                        batch_size=self.config.eval_batch_size,
                        generator=torch.manual_seed(self.config.seed),
                        num_inference_steps=self.config.num_inference_steps,
                        output_type='numpy'
                        ).images
        print(images.shape)
        return images
    


    
