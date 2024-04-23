import os 
from pathlib import Path
from diffusers.utils import make_image_grid

def save_images(images, output_path) : 
    """
    Save List[PIL.Image] as a grid
    """
    image_grid = make_image_grid(images[:,0,:,:], rows=4, cols=4)
    Path(output_path).parent.mkdir(exist_ok=True)
    print(output_path)
    image_grid.save(str(output_path))
    