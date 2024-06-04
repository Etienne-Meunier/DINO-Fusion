from pathlib import Path
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import webdataset as wds
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from ipdb import set_trace

def save_images(images, output_path) : 
    """
    XXX
    """
    images = images.cpu()
    Path(output_path).parent.mkdir(exist_ok=True)
    
    np.save(str(output_path).replace(".png",".npy"),images)

    fig, axs = plt.subplots(3, min(len(images), 8), figsize=(15,15))
    for ci, c in enumerate([(0, 'surface_toce'), (18, 'surface_soce'), (-1, 'ssh')]) : 
        for b in range(min(len(images), 8)) : 
            axs[ci, b].imshow(images[b, c[0]], origin='lower')
            
            if b == 0 : 
                axs[ci, b].set_title(c[1])
            else :
                axs[ci, b].axis('off')

    fig.savefig(str(output_path), bbox_inches='tight')


class TransformFields :

        def __init__(self, mu=None, std=None,mask=None,step=1) : 
            self.step=step
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
                #2. replace padding and edges by 0 
                data = self.replaceEdges(data,feature,val=0)
                #3. pad data
                data = self.padData(data,xup=1,xdown=1,yup=4,ydown=5,val=0)
                dico[feature] = data
            #set_trace()
            data = np.concatenate((dico['soce'][:,::self.step], dico['toce'][:,::self.step], dico['ssh']), axis=1).squeeze(axis=0)
            return data


        def get_infos(self):
            for feature in ["soce","toce","ssh"]:
                file_path = f'/home/tissot/data/infos/{feature}_info.pkl'
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    self.mu[feature]   = data["mean"]
                    self.std[feature]  = data["std"]
                    self.mask[feature] = data["mask"]


        def standardize_4D(self,sample,feature):
            """
                Standardize the data given a mean and a std
            """
            return (sample[f'{feature}.npy'] - self.mu[feature]) / (2*self.std[feature])

        def replaceEdges(self,data,feature,val=None,values=None):
            """
                Replace edges by a values. Default is 0
                data : batch, depth, x, y 
            """
            batch_size,depth = np.shape(data)[0:2]
            if val is not None:
                mask = np.tile(self.mask[feature], (batch_size, depth, 1, 1))
                data[mask] = val
            #find less greedy than transpose method
            #if values is not None:
            #    mask = np.tile(self.mask[feature], (batch_size, 1, 1))
            #    data = data.transpose(1, 0, 2, 3)
            #    values = np.squeeze(values)
            #    for i in range(len(values)):
            #        data[i][mask] = values[i]
            #    data = data.transpose(1, 0, 2, 3)
            return data
        
        def padData(self,dataset,xup,xdown,yup,ydown,val):
            """
                pad data on axis x and y
            """
            return np.pad(dataset, ((0, 0), (0, 0), (yup, ydown), (xup, xdown)), mode='constant',constant_values=val)

                                        
                
def get_dataloader(tar_file, batch_size=5,step=1) :
    composed = transforms.Compose([TransformFields(step=step)])
    dataset = wds.WebDataset(tar_file).shuffle(100).decode().map(composed) 
    dl = DataLoader(dataset=dataset, batch_size=batch_size)
    return dl
    #if distributed:
    #    torch.distributed.init_process_group(backend='gloo') 
    #    sampler = DistributedSampler(dataset)
    #else:
    #    sampler = None
