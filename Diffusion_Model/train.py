from DiffusionModel import DiffusionModel
from configs.base_config import TrainingConfig
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import torch 
from utils import save_images

# Load config
config = TrainingConfig()

# Load dataset
config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")


preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# Load Model 
diffusion = DiffusionModel(config)

for epoch in range(config.num_epochs) : 
    progress_bar = tqdm(total=config.train_steps_by_epoch)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(train_dataloader):
        if step == config.train_steps_by_epoch : break
        loss = diffusion.training_step(batch)
        progress_bar.update(1)
        logs = {"loss": loss,
                "lr": diffusion.lr_scheduler.get_last_lr()[0],
                "step": step}
        progress_bar.set_postfix(**logs)
    
    print('Generate images ...')
    generated_images = diffusion.test_step()
    save_images(generated_images, './' + config.output_dir + f'/epoch_{epoch}.png')


    