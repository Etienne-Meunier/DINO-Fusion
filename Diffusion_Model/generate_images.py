from pipelines.pipeline_tensor import DDPMPipeline_Tensor

import argparse
import numpy as np
import torch
from utils import save_images
from pathlib import Path
import time
from ipdb import set_trace


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images")
    parser.add_argument("--model_path", type=str, help= "path of the trained diffuser model")
    parser.add_argument("--batch", type=int, help= "Number of generated states", default=8)
    parser.add_argument("--inf_steps", type=int, help= "Number of inference steps")
    parser.add_argument("--save_file", type=str, help= "path of the generated images")
    args = parser.parse_args()
    
    print("Import pipeline")
    pipeline = DDPMPipeline_Tensor.from_pretrained(args.model_path).to('cuda')
    generator = torch.Generator("cuda").manual_seed(0)
    print("Image generation...")
    images = pipeline( batch_size=args.batch, num_inference_steps=args.inf_steps, return_dict=False)[0]
    # Save
    timestr = time.strftime("%Y%m%d-%H%M%S")
    outpath = args.model_path + f'/inference/infesteps_{args.inf_steps}/{timestr}.png'
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    save_images(images, outpath)
    print(f"Image saved at {outpath}")

    #path = "/home/meunier/diffModel_experiences/cb7xsahm"
    #path = "/home/tissot/DINO-Fusion/Diffusion_Model/wandb/logs"

#python generate_images.py --model_path "/home/meunier/diffModel_experiences/cb7xsahm" --batch 1 --inf_steps 1000 --save_file "/home/meunier/Data/Dino-Fusion"