from pathlib import Path
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import webdataset as wds
import pickle

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

class TransformFields :

        def __init__(self, mu=None, std=None,mask=None) : 
            if mu is None:
                self.mu,self.std,self.mask = {},{},{}
                self.get_infos()
            else:
                self.mu   = mu
                self.std  = std
                self.mask = mask
        
        def __call__(self, sample) :
            dico = {}
            for feature in ["soce","toce","ssh"]:
                #1. standardize
                data = self.standardize_4D(sample,feature)
                #1. replace padding and edges by 0 
                data = self.replaceEdges(data,feature)
                dico[feature] = data
            return dico


        def get_infos(self):
            for feature in ["soce","toce","ssh"]:
                file_path = f'/data/emeunier/dataDiffModel/{feature}_infos.pickle'
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    self.mu[feature]   = data["mean"]
                    self.std[feature]  = data["std"]
                    self.mask[feature] = data["mask"]


        def standardize_4D(self,sample,feature):
            """
                Standardize the data given a mean and a std
            """
            return (sample[f'{feature}.npy'] - self.mu[feature]) / self.std[feature]

        def replaceEdges(data,feature,values=None):
            """
                Replace edges by a values. Default is 0
                data : batch, depth, x, y 
            """
            batch_size,depth = np.shape(data)[0:2]
            if values is None:
                mask = np.tile(self.mask[feature], (batch_size, depth, 1, 1))
                data[mask] = 0
            else:
                mask = np.tile(self.mask[feature], (batch_size, 1, 1))
                data = data.transpose(1, 0, 2, 3)
                values = np.squeeze(values)
                for i in range(len(values)):
                    data[i][mask] = values[i]
                data = data.transpose(1, 0, 2, 3)
            return data
                                        
                

def get_dataloader(tar_file, batch_size=5) : 
    composed = transforms.Compose([TransformFields()])
    dataset = wds.WebDataset(tar_file).shuffle(100).decode().map(composed)
    dl = DataLoader(dataset=dataset, batch_size=batch_size)
    return dl