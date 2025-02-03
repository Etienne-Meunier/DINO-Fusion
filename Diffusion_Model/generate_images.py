from pipelines.pipeline_tensor import DDPMPipeline_Tensor
import argparse
import numpy as np
import torch
from utils import save_images
from pathlib import Path
import time
from ipdb import set_trace
from accelerate import Accelerator
from pipelines.constraints import *

# Dictionary mapping constraint names to their classes
AVAILABLE_CONSTRAINTS = {
    'zero_mean': ZeroMeanConstraint,
    'gradient_zero_mean': GradientZeroMeanConstraint,
    'border_zero' : BorderZeroConstraint
}

def get_constraints(constraint_names):
    """Create constraint objects from their names"""
    constraints = []
    for name in constraint_names:
        if name not in AVAILABLE_CONSTRAINTS:
            raise ValueError(f"Unknown constraint: {name}. Available constraints: {list(AVAILABLE_CONSTRAINTS.keys())}")
        constraints.append(AVAILABLE_CONSTRAINTS[name]())
    return constraints

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images")
    parser.add_argument("--model_path", type=str, help="path of the trained diffuser model")
    parser.add_argument("--batch", type=int, help="Number of generated states", default=8)
    parser.add_argument("--inf_steps", type=int, help="Number of inference steps", default=1000)
    parser.add_argument("--seed", type=int, help="seed to use", default=0)
    parser.add_argument("--constraints", nargs="*", choices=AVAILABLE_CONSTRAINTS.keys(),
                          default=[], help="List of constraints to apply")
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device

    print("Import pipeline")
    pipeline = DDPMPipeline_Tensor.from_pretrained(args.model_path).to(device)

    pipeline.constraints =  get_constraints(args.constraints)


    generator = torch.Generator(device)
    if args.seed != -1 :
        print(f'initialise with seed : {args.seed}')
        generator.manual_seed(args.seed)

    print(f"Image generation on {device}...")
    images = pipeline(
        batch_size=args.batch,
        num_inference_steps=args.inf_steps,
        generator=generator,
        return_dict=False
    )[0]

    # If using distributed training, make sure to gather results
    if accelerator.num_processes > 1:
        images = accelerator.gather(images)

    # Create constraint string for path
    constraint_str = '_'.join(args.constraints) if args.constraints else 'no_constraints'

    # Save
    timestr = time.strftime("%Y%m%d-%H%M%S")
    outpath = args.model_path + f'/inference/infesteps_{args.inf_steps}/constraints_{constraint_str}/{timestr}.png'
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)

    # Make sure images are on CPU before saving
    images = images.cpu()
    save_images(images, outpath)
    print(f"Image saved at {outpath}")
