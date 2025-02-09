"""
Training script for the Diffusion Model.
This script handles the training loop, model initialization, and image generation.
"""

from torch.nn.modules import normalization
from DiffusionModel import DiffusionModel
from configs.base_config import TrainingConfig
from utils import get_dataloader, save_images
from tqdm import tqdm

def main():
    """
    Main training function that handles:
    1. Configuration and data loading
    2. Model initialization
    3. Training loop
    4. Periodic image generation and model saving
    """
    print("\n----------INITIALISATION----------\n")

    # Initialize configuration
    config = TrainingConfig()
    print("Config loaded")

    # Load and prepare data
    train_dataloader = get_dataloader(
        data_file=config.data_file,
        batch_size=config.train_batch_size,
        fields=config.fields,
        normalisation=config.normalisation
    )
    config.data_shape = train_dataloader.get_data_shape()
    print("Data loaded")

    # Initialize diffusion model
    diffusion = DiffusionModel(config)
    if config.pretrained_model_path is not None:
        diffusion.load_model(config.pretrained_model_path)
    print("Model loaded")

    # Prepare model components with accelerator
    global_step = 0
    (
        diffusion.denoiser,
        diffusion.optimizer,
        diffusion.lr_scheduler,
        train_dataloader
    ) = diffusion.accelerator.prepare(
        diffusion.denoiser,
        diffusion.optimizer,
        diffusion.lr_scheduler,
        train_dataloader
    )
    print("Accelerator setup completed")

    print("\n----------TRAINING----------\n")
    # Main training loop
    for epoch in range(config.num_epochs):
        # Setup progress bar
        progress_bar = tqdm(total=config.train_steps_by_epoch)
        progress_bar.set_description(f"Epoch {epoch}")

        # Training steps
        for step, batch in enumerate(train_dataloader):
            if step == config.train_steps_by_epoch:
                break

            # Perform training step
            loss = diffusion.training_step(batch)

            # Update progress bar and logging
            progress_bar.update(1)
            logs = {
                "loss": loss,
                "lr": diffusion.lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            diffusion.accelerator.log(logs, step=global_step)
            global_step += 1

        # Generate images and save model periodically
        if epoch % config.generation_frequency == 0:
            print('Generating images...')
            generated_images = diffusion.test_step()
            save_images(
                generated_images,
                f'./{config.output_dir}/epoch_{epoch}.png'
            )
            diffusion.save_model()

    # Cleanup
    diffusion.accelerator.end_training()

if __name__ == '__main__':
    main()
