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

        def __init__(self):#, mu, std) : 
            self.mu   = {}
            self.std  = {} 
            self.mask = {}
            #print(f'Transform init with mu {self.mu} and std {self.std}')
        
        def __call__(self, sample) :
            #toce, soce, ssh = sample['toce.npy'], sample['soce.npy'], sample['ssh.npy']
            dico = {}
            for feature in ["soce","toce","ssh"]:
                mu,std,mask = get_infos(feature)  
                #1. standardize
                data = standardize_4D(sample[f"{feature}.npy"],mu,std)
                #1. replace padding and edges by 0 
                data = replaceEdges(data,mask)#,mean_stand)
                dico[feature] = data
            #return {'toce' : toce, 'soce' : soce, 'ssh' : ssh }

        def get_infos(feature):
            file_path = f'/data/emeunier/dataDiffModel/{feature}_infos.pickle'
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                mu[feature] = data["mean"]
                std[feature] = data["std"]
                mask[feature] = data["mask"]
                return data["mean"],  data["std"], data["mask"]


        def standardize_4D(data,x_mean=None,x_std=None):
            """
                Standardize the data given a mean and a std
                OR
                Obtain mean and std of a dataset
                    - data :         (batch,depth,x,y) 
                    - mean and std : (1,depth,1,1)
            """
            if x_mean is None:
                # Get min, max value aming all elements for each column
                x_mean = np.nanmean(data, axis=(0,2,3), keepdims=True)
                x_std  = np.nanstd(data, axis=(0,2,3), keepdims=True)
                return x_mean, x_std
            else : 
                data = (data - x_mean)/ x_std
                return data

        def replaceEdges(data,mask,values=None):
            """
                Replace edges by a values. Default is 0
                data : batch, depth, x, y 
            """
            batch_size,depth = np.shape(data)[0:2]
            if values is None:
                mask = np.tile(mask, (batch_size, depth, 1, 1))
                data[mask] = 0
            else:
                mask = np.tile(mask, (batch_size, 1, 1))
                data = data.transpose(1, 0, 2, 3)
                values = np.squeeze(values)
                for i in range(len(values)):
                    data[i][mask] = values[i]
                data = data.transpose(1, 0, 2, 3)
            return data
                                        
                

def get_dataloader(tar_file, batch_size=5) : 

    composed = transforms.Compose([TransformFields(0,0)])

    dataset = wds.WebDataset(tar_file).shuffle(100).decode().map(composed)
    dl = DataLoader(dataset=dataset, batch_size=batch_size)
    return dl