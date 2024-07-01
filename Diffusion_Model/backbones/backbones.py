from diffusers import UNet2DModel
from diffusers.training_utils import EMAModel

def get_simple_unet(image_size, use_ema) : 
    """
    Return the base unet introduced in HG tutorial
    https://huggingface.co/docs/diffusers/tutorials/basic_training
    image_size (int) : size of the image
    """
    
    
    if use_ema:
        ema_model = EMAModel(
            model.parameters(),
            #inv_gamma=config.ema_inv_gamma,
            #power=config.ema_power,
            #max_value=config.ema_max_decay
        )

    return model #ema_model if use_ema else model 
