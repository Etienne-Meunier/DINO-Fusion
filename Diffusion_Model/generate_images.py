from diffusers import DDPMScheduler, DDPMPipeline
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images")
    parser.add_argument("--model_path", type=str, help= "path of the trained diffuser model")
    parser.add_argument("--batch", type=int, help= "Number of generated states")
    parser.add_argument("--inf_steps", type=int, help= "Number of inference steps")
    parser.add_argument("--save_file", type=str, help= "path of the generated images")
    args = parser.parse_args()
    
    print("Import pipeline")
    pipeline = DDPMPipeline.from_pretrained(args.model_path)
    print("Image generation...")
    images = pipeline( batch_size=args.batch, num_inference_steps=args.inf_steps, output_type='numpy').images
    print(f"Image saved at {args.save_file}/generated_images.npy")
    np.save(f"{args.save_file}/generated_images.npy",images)
    #path = "/home/tissot/DINO-Fusion/model/cb7xsahm"
    #path = "/home/tissot/DINO-Fusion/Diffusion_Model/wandb/logs"

#python generate_images.py --model_path "/home/tissot/DINO-Fusion/Diffusion_Model/wandb/logs" --batch 1 --inf_steps 1000 --save_file "/home/tissot/DINO-Fusion"