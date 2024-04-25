from diffusers import UNet2DModel


def get_simple_unet(image_size) : 
    """
    Return the base unet introduced in HG tutorial
    https://huggingface.co/docs/diffusers/tutorials/basic_training
    image_size (int) : size of the squared image
    """
    model = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=73,  # the number of input channels, 3 for RGB images
        out_channels=73,  # the number of output channels
        layers_per_block=4,  # how many ResNet layers to use per UNet block
        block_out_channels= (256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            #"DownBlock2D",
            #"DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            #"UpBlock2D",
            #"UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model
