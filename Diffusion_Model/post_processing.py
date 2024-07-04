import torch
import matplotlib.pyplot as plt
import seaborn as sns
import einops
import numpy as np
import pickle
from utils import TransformFields



#path_generated='/Users/emeunier/Documents/scai/qeeyvkep/interence/infesteps_1000/20240604-135240.npy'
path_generated='/Users/emeunier/Documents/scai/clean_images_2.npy'
generated_images = np.load(path_generated)
generated_images.shape


tr = TransformFields(step=2)
processed_generated = tr.uncall(generated_images)
with open(path_generated.replace('.npy', '_postprocesses.pkl'), 'wb') as f : 
    pickle.dump(processed_generated, f)

