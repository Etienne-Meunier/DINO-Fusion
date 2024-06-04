from DiffusionModel import DiffusionModel
from configs.base_config import TrainingConfig, MiniConfig
from torchvision import transforms
from utils import get_dataloader
from tqdm import tqdm
import torch 
from ipdb import set_trace
from utils import save_images
import numpy as np

def main() :
    print("\n----------INITIALISATION----------\n")
    # Load config
    config = TrainingConfig()
    print("Config loaded")

    mask = None#TODO:load mask here
    train_dataloader = get_dataloader(config.data_file, batch_size=config.train_batch_size,step=2)
    print("Data loaded")

    # Load Model 
    diffusion = DiffusionModel(config)
    print("Model loaded")

    global_step = 0
    diffusion.denoiser, diffusion.optimizer, diffusion.lr_scheduler, train_dataloader = diffusion.accelerator.prepare(diffusion.denoiser,
                                                                                        diffusion.optimizer,
                                                                                        diffusion.lr_scheduler,
                                                                                        train_dataloader)
    print("Accelerate setted")

    print("\n----------TRAINING----------\n")
    for epoch in range(config.num_epochs) : 
        progress_bar = tqdm(total=config.train_steps_by_epoch)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            if step == config.train_steps_by_epoch : break
            loss = diffusion.training_step(batch)
            progress_bar.update(1)
            logs = {"loss": loss,
                    "lr": diffusion.lr_scheduler.get_last_lr()[0],
                    "step": global_step}
            progress_bar.set_postfix(**logs)
            diffusion.accelerator.log(logs, step=global_step)
            global_step += 1
        
        if epoch % 20 == 0:
            print('Generate images ...')
            generated_images = diffusion.test_step()
            save_images(generated_images, './' + config.output_dir + f'/epoch_{epoch}.png')
            diffusion.save_model()

    diffusion.accelerator.end_training()

if __name__ == '__main__' : 
    main()

    
