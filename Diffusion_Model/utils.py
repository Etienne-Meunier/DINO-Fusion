import os 
from pathlib import Path
#from diffusers.utils import make_image_grid
from torchvision.utils import make_grid, save_image
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

def save_images(images, output_path) : 
    """
    XXX
    """
    #image_grid = make_image_grid(images[:,:,:,0], nrows=4)
    Path(output_path).parent.mkdir(exist_ok=True)
    print(output_path)
    #save_image(from_numpy(images[:,:,:,0].squeeze()), str(output_path), nrow=4)
    

    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
    image_data = [images[i,:,:,0].squeeze() for i in range(images.shape[0])]

    for ax, im in zip(grid, image_data):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='coolwarm')

    fig.savefig(str(output_path), bbox_inches='tight')
