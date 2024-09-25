from DiffusionModel import DiffusionModel
from configs.base_config import *
from torchvision import transforms
from utils import get_dataloader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
from diffusers.optimization import get_cosine_schedule_with_warmup

def main() :
    print("\n----------INITIALISATION----------\n")
    
    exp="TOCE"
    
    # Load config
    if exp=="SSH":
        config = SSHTrainingConfig()
    if exp=="TOCE":
        config = TOCETrainingConfig()
    if exp=="SOCE":
        config = SOCETrainingConfig()
        
    print("Config loaded")
    config.train_batch_size = 8 ####### 8 
    train_dataloader = get_dataloader(config.data_file, batch_size=config.train_batch_size, fields=config.fields)
    config.data_shape = train_dataloader.get_data_shape()
    print("Data loaded")
    print(config.data_shape)

    print("\n----------TRAINING----------\n")

    if exp=="SSH":
        model = AutoencoderKL(in_channels = 1,
            out_channels = 1,
            down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",),
            up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",),
            block_out_channels = (64, 64, 64,),
            layers_per_block = 1,
            act_fn = "silu",
            latent_channels= 1,
            scaling_factor = 1.).to('cuda') #### attention au scaling factor, check D1 in https://arxiv.org/pdf/2112.10752
    if exp=="TOCE":
        model = AutoencoderKL(in_channels = 18,
            out_channels = 18,
            down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",),
            up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",),
            block_out_channels = (64, 128, 128,),
            layers_per_block = 1,
            act_fn = "silu",
            latent_channels= 4,
            scaling_factor = 1.).to('cuda') #### attention au scaling factor, check D1 in https://arxiv.org/pdf/2112.10752
    if exp=="SOCE":
        model = AutoencoderKL(in_channels = 18,
            out_channels = 18,
            down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",),
            up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",),
            block_out_channels = (64, 128, 128,),
            layers_per_block = 1,
            act_fn = "silu",
            latent_channels= 4,
            scaling_factor = 1.).to('cuda') #### attention au scaling factor, check D1 in https://arxiv.org/pdf/2112.10752
        
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    epochs = 200 #100 for SSH
    loss_l1 = torch.nn.L1Loss().to('cuda')
    loss_mse = torch.nn.MSELoss().to('cuda')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4) 
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                  num_warmup_steps=0,
                                                  num_training_steps=(1800/config.train_batch_size)*epochs) #### le chiffre 1800 est à vérifier
    
    losses = []
    kl_weight = 1e-8

    model.train()
    
    for epoch in range(epochs):
        running_loss=0.
        print(epoch)
        for image in train_dataloader:
            image = image.to('cuda')
            posterior = model.encode(image).latent_dist
            reconstructed = model.decode(posterior.sample()).sample
            #############
            loss_rec = loss_l1(reconstructed,image)
            loss_kl = posterior.kl().mean()
            loss = loss_rec + kl_weight * loss_kl
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(loss_rec.item(), loss_kl.item(), lr_scheduler.get_last_lr()[0])
            #print(loss.item())
            running_loss += loss.item()
        losses.append(running_loss)

    
    # Defining the Plot Style
    fig = plt.figure()
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.semilogy(losses)
    if exp=="SSH":
        fig.savefig('./backbones/AutoEncoder/SSH/losses_VAE.png', dpi=fig.dpi)
    if exp=="TOCE":
        fig.savefig('./backbones/AutoEncoder/TOCE/losses_VAE.png', dpi=fig.dpi)
    if exp=="SOCE":
        fig.savefig('./backbones/AutoEncoder/SOCE/losses_VAE.png', dpi=fig.dpi) 
        
    print("\n----------SAVE----------\n")
    if exp=="SSH":
        #model.from_pretrained('./backbones/AutoEncoder/SSH/')
        model.save_pretrained('./backbones/AutoEncoder/SSH/')
    if exp=="TOCE":
        #model.from_pretrained('./backbones/AutoEncoder/TOCE/')
        model.save_pretrained('./backbones/AutoEncoder/TOCE/')
    if exp=="SOCE":
        #model.from_pretrained('./backbones/AutoEncoder/TOCE/')
        model.save_pretrained('./backbones/AutoEncoder/SOCE/')
        
    print("\n----------TEST----------\n")

    model.eval()

    with torch.no_grad():
        test_image = next(iter(train_dataloader))
        test_image.shape
        reconstructed = model(test_image.to("cuda"))

    ####################
    fig, axs = plt.subplots(2, 2, figsize=(5, 15))
    for i in range(4):
        ax = axs[i // 2, i % 2]
        cax = ax.imshow(test_image[i, 0].detach().cpu(), cmap='coolwarm', vmin=-1, vmax=1)
        ax.axis('off')  # Hide the axis
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position and size of colorbar
    fig.colorbar(cax, cax=cbar_ax)
    if exp=="SSH":
        fig.savefig('./backbones/AutoEncoder/SSH/original_VAE.png', dpi=fig.dpi)
    if exp=="TOCE":
        fig.savefig('./backbones/AutoEncoder/TOCE/original_VAE.png', dpi=fig.dpi)
    if exp=="SOCE":
        fig.savefig('./backbones/AutoEncoder/SOCE/original_VAE.png', dpi=fig.dpi)   
    
    #####################
    fig, axs = plt.subplots(2, 2, figsize=(5, 15))
    for i in range(4):
        ax = axs[i // 2, i % 2]
        cax = ax.imshow(reconstructed.sample[i, 0].detach().cpu(), cmap='coolwarm', vmin=-1, vmax=1)
        ax.axis('off')  # Hide the axis
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position and size of colorbar
    fig.colorbar(cax, cax=cbar_ax)
    if exp=="SSH":
        fig.savefig('./backbones/AutoEncoder/SSH/reconstructed_VAE.png', dpi=fig.dpi)
    if exp=="TOCE":
        fig.savefig('./backbones/AutoEncoder/TOCE/reconstructed_VAE.png', dpi=fig.dpi)
    if exp=="SOCE":
        fig.savefig('./backbones/AutoEncoder/SOCE/reconstructed_VAE.png', dpi=fig.dpi)
    #####################
    def compute_laplacian(image):
        # Compute gradients along x and y axis
        grad_x = np.gradient(image, axis=0)
        grad_y = np.gradient(image, axis=1)
        
        # Compute second derivatives
        laplacian = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1)
        return laplacian

    fig, axs = plt.subplots(2, 2, figsize=(10, 30))
    laplacians = []
    for i in range(4):
        image = reconstructed.sample[i, 0].detach().cpu()
        laplacian = compute_laplacian(image)
        laplacians.append(laplacian)
        ax = axs[i // 2, i % 2]
        cax = ax.imshow(laplacian, cmap='seismic', vmax=0.01, vmin=-0.01)
        ax.axis('off')  # Hide the axis
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position and size of colorbar
    fig.colorbar(cax, cax=cbar_ax)
    if exp=="SSH":
        fig.savefig('./backbones/AutoEncoder/SSH/reconstructed_laplac.png', dpi=fig.dpi)
    if exp=="TOCE":
        fig.savefig('./backbones/AutoEncoder/TOCE/reconstructed_laplac.png', dpi=fig.dpi)
    if exp=="SOCE":
        fig.savefig('./backbones/AutoEncoder/SOCE/reconstructed_laplac.png', dpi=fig.dpi)
    #####################
    fig, axs = plt.subplots(2, 2, figsize=(10, 30))
    laplacians = []
    for i in range(4):
        image = test_image[i, 0].detach().cpu()
        laplacian = compute_laplacian(image)
        laplacians.append(laplacian)
        ax = axs[i // 2, i % 2]
        cax = ax.imshow(laplacian, cmap='seismic', vmax=0.01, vmin=-0.01)
        ax.axis('off')  # Hide the axis
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position and size of colorbar
    fig.colorbar(cax, cax=cbar_ax)
    if exp=="SSH":
        fig.savefig('./backbones/AutoEncoder/SSH/original_laplac.png', dpi=fig.dpi)
    if exp=="TOCE":
        fig.savefig('./backbones/AutoEncoder/TOCE/original_laplac.png', dpi=fig.dpi)
    if exp=="SOCE":
        fig.savefig('./backbones/AutoEncoder/SOCE/original_laplac.png', dpi=fig.dpi)

if __name__ == '__main__' : 
    main()

    