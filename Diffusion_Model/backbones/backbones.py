from diffusers import UNet2DModel
from diffusers.training_utils import EMAModel
from math import ceil

sl_len = lambda sl : ceil((sl.stop - sl.start) / sl.step)

def get_simple_unet(data_shape , use_ema) : 
    """
    Return the base unet introduced in HG tutorial
    https://huggingface.co/docs/diffusers/tutorials/basic_training
    image_size (int) : size of the image
    """
    model = UNet2DModel(
        sample_size=data_shape[1:],  # the target image resolution
        in_channels= data_shape[0],   # the number of input channels, 3 for RGB images
        out_channels= data_shape[0],  # the number of output channels
        layers_per_block= 2,  # how many ResNet layers to use per UNet block
        block_out_channels= (32, 64, 64, 128),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            #"DownBlock2D",
            "DownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "UpBlock2D",  # a ResNet upsampling block with spatial self-attention
            #"UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    
    if use_ema:
        ema_model = EMAModel(
            model.parameters(),
            #inv_gamma=config.ema_inv_gamma,
            #power=config.ema_power,
            #max_value=config.ema_max_decay
        )
    print(model)
    return model #ema_model if use_ema else model 
