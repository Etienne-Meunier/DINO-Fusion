from diffusers import DDPMScheduler, DDPMPipeline, AutoencoderKL

import torch.nn as nn
import torch
import torch.nn.functional as F
from backbones.backbones import get_simple_unet
from diffusers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils.operations import gather_object
import os
from ipdb import set_trace
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from diffusers.training_utils import EMAModel

class DiffusionModel(nn.Module) : 
    
    def __init__(self, config) : 
        super().__init__()
        self.config = config
        self.accelerator = self.config_accelerate()
        self.config.data_shape = torch.Size([37, 208, 64]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!! while training works, test breaks so I had to put this manually

        if self.config.use_LDM:
            self.AE_model = AutoencoderKL.from_pretrained('./backbones/AutoEncoder/SSH/').to('cuda')
            self.AE_model.requires_grad_(False)
            self.config.data_shape = torch.Size([4, 200, 64]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            
        self.denoiser = get_simple_unet(self.config.data_shape)
        self.noise_scheduler = DDPMScheduler(self.config.num_train_timesteps,
                                             beta_schedule="squaredcos_cap_v2", 
                                             clip_sample=True)
        if self.config.st_path is not None : 
            self.load_model(self.config.st_path)
        self.optimizer, self.lr_scheduler = self.config_optimizer()


        if self.config.use_ema:
            self.ema_model = EMAModel(
                self.denoiser.parameters(),
                #inv_gamma=config.ema_inv_gamma,
                #power=config.ema_power,
                #max_value=config.ema_max_decay
            )
        
    def config_accelerate(self) :        
        accelerator = Accelerator(mixed_precision=self.config.mixed_precision,
                                  gradient_accumulation_steps=self.config.gradient_accumulation_steps, 
                                  log_with=self.config.logger)
        
        if (self.config.logger == 'wandb') and accelerator.is_main_process:
            accelerator.init_trackers('dino-fusion', config=vars(self.config))
            accelerator.trackers[0].run.log_code("../")
        if self.config.output_dir == 'wandb' : 
            wandb_id = accelerator.trackers[0].run.id
            self.config.output_dir = f"{self.config.base_output_dir}/{wandb_id}"
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f'Config output dir : {self.config.output_dir}')
        return accelerator

    def config_optimizer(self) : 
        """
        Setup optimizer and scheduler
        Return ls
            optimizer
            lr_scheduler
        """
        #https://github.com/huggingface/accelerate/issues/963
        optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.config.learning_rate)
        num_update_steps_per_epoch = self.config.train_steps_by_epoch / self.config.gradient_accumulation_steps
        if self.config.lr_schedule == 'cosine_schedule' :
            lr_scheduler = get_cosine_schedule_with_warmup( #constant
                                optimizer=optimizer,
                                num_warmup_steps=self.config.lr_warmup_steps * self.accelerator.num_processes,
                                num_training_steps=(num_update_steps_per_epoch * self.config.num_epochs * self.accelerator.num_processes),
                            )
        elif self.config.lr_schedule == 'linear' :
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=self.config.lr_warmup_steps * self.accelerator.num_processes)
            
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
    
    def training_step(self, batch) :
        """
        Takes a batch of images and do a training step.
        """
        clean_images = batch#["images"]
        #print(clean_images.shape)
        
        if self.config.use_LDM:           
            clean_images = self.AE_model.encode(clean_images).latent_dist.sample()
            clean_images = F.pad(clean_images, (1,1,0,0) , "constant", 0) ##### padding from 62 to 64 to make it work with the Unet
            
        noisy_images, noises, timesteps = self.get_noisy_images(clean_images)
        
        with self.accelerator.accumulate(self.denoiser) :
            noises_pred = self.denoiser(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noises_pred, noises) ################# changed from mse_loss to smooth_l1
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.denoiser.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        if self.accelerator.sync_gradients:
            if self.config.use_ema:
                self.ema_model.step(self.denoiser.parameters())
                
        return loss.detach().item()
    
    def get_pipeline(self) : 
        pipeline = DDPMPipeline_Tensor(unet=self.accelerator.unwrap_model(self.ema_model.averaged_model if self.config.use_ema else self.denoiser),
                                scheduler=self.noise_scheduler)
        return pipeline

    def test_step(self) : 
        """
        Generate a list of images `List[PIL.Image]` from noise
        """
        pipeline = self.get_pipeline()
        images = pipeline(batch_size=self.config.eval_batch_size,
                        generator=torch.manual_seed(self.config.seed),
                        num_inference_steps=self.config.num_inference_steps,
                        return_dict=False)[0]
        
        if self.config.use_LDM:
            print("decoding")
            with torch.no_grad():
                images = self.AE_model.decode(images[:,:,:,1:-1]).sample

        return images
    
    def save_model(self) : 
        pipeline = self.get_pipeline()
        pipeline.save_pretrained(self.config.output_dir)

    def load_model(self, model_path) : 
        print(f'Loading model : {model_path}')
        pipeline = DDPMPipeline_Tensor.from_pretrained(model_path).to('cuda')
        self.denoiser.load_state_dict(pipeline.unet.state_dict())

    



    
