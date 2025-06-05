from pipelines.pipeline_tensor import DDPMPipeline_Tensor
import argparse
import numpy as np
import torch
from utils import *
from pathlib import Path
import time
from ipdb import set_trace
from accelerate import Accelerator
from pipelines.constraints import *

# Dictionary mapping constraint names to their classes
AVAILABLE_CONSTRAINTS = {
    'zero_mean': ZeroMeanConstraint,
    'gradient_zero_mean': GradientZeroMeanConstraint,
    'gradient_zero_mean_density': GradientZeroMeanDensityConstraint,
    'gradient_density': GradientDensityConstraint,
    'border_zero': BorderZeroConstraint,
    'conditional_generation_sshlow': ConditionalGeneration_SSHLow,
    'conditional_generation_sshhigh': ConditionalGeneration_SSHHigh,
    'conditional_generation_stemplow': ConditionalGeneration_STempLow,
    'conditional_generation_stemphigh': ConditionalGeneration_STempHigh
}

def get_constraints(constraint_names, **kwargs):
    """Create constraint objects from their names"""
    constraints = []
    for name in constraint_names:
        if name not in AVAILABLE_CONSTRAINTS:
            raise ValueError(f"Unknown constraint: {name}. Available constraints: {list(AVAILABLE_CONSTRAINTS.keys())}")
        if 'gradient' in name: #name == 'gradient_zero_mean' or name == 'gradient_zero_mean_density':
            constraints.append(AVAILABLE_CONSTRAINTS[name](**kwargs))
        else: 
            constraints.append(AVAILABLE_CONSTRAINTS[name]())
    return constraints

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images")
    parser.add_argument("--model_path", type=str, help="path of the trained diffuser model")
    parser.add_argument("--batch", type=int, help="Number of generated states", default=8)
    parser.add_argument("--inf_steps", type=int, help="Number of inference steps", default=1000)
    parser.add_argument("--seed", type=int, help="seed to use", default=0)
    parser.add_argument('--beta', type=float, default=0.003)
    parser.add_argument('--beta_type', type=str, default='constant')
    parser.add_argument("--constraints", nargs="*", choices=AVAILABLE_CONSTRAINTS.keys(),
                          default=[], help="List of constraints to apply")
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device

    print("Import pipeline")
    pipeline = DDPMPipeline_Tensor.from_pretrained(args.model_path).to(device)

    pipeline.constraints = get_constraints(args.constraints, beta=args.beta, beta_type=args.beta_type)

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
    beta_str = f"{args.beta:.4g}"

    # Save
    timestr = time.strftime("%Y%m%d-%H%M%S")
    outpath = args.model_path + f'/inference/infesteps_{args.inf_steps}/constraints_{constraint_str}/beta_{beta_str}_{timestr}.png'
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)

    # Make sure images are on CPU before saving
    images = images.cpu()
    save_images(images, outpath)
    print(f"Image saved at {outpath}")
