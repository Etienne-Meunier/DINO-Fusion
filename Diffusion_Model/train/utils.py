from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_images(images, output_path) :
    """
    Save `images` (b, c, i, j) as `<output_path>.npy` and a figure of surface T, S and SSH of the first 8 states in `output_path`.
    """
    images = images.cpu()
    Path(output_path).parent.mkdir(exist_ok=True)

    np.save(str(output_path).replace(".png",".npy"),images)
    try :

        fig, axs = plt.subplots(3, min(len(images), 8), figsize=(15,15))
        for ci, c in enumerate([(0, 'surface_toce'), (8, 'surface_soce'), (-1, 'ssh')]) :
            for b in range(min(len(images), 8)) :
                axs[ci, b].imshow(images[b, c[0]], origin='lower')

                if b == 0 :
                    axs[ci, b].set_title(c[1])
                else :
                    axs[ci, b].axis('off')

        fig.savefig(str(output_path), bbox_inches='tight')
    except Exception as e :
        print(f'Error drawing figure {e}')
