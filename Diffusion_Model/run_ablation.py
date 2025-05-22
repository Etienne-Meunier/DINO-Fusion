import os
import numpy as np

beta_values = np.linspace(0.00008, 0.00009, 2)

for beta in beta_values:
    print('Testing with current beta: ' + str(beta))
    os.system(f"uv run Diffusion_Model/generate_images.py --model_path ../test-generate-img --constraints border_zero gradient_zero_mean --beta {beta} ")
