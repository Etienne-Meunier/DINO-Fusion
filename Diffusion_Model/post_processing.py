"""
Turn normalised generated states (b, 37, 208, 64) into physical fields saved in `<save_dir>/{toce,soce,ssh}.npy` :
    toce, soce : (b, 36, 199, 62)
    ssh        : (b, 1, 199, 62)
The values of the row next to the southern wall are copied from the nearest interior row.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from configs.base_config import TrainingConfig
from DataLoader import get_infos
from TransformFields import TransformationPipeline


def unnormalise(generated, transform) :
    """
        generated (b, c, i, j) : normalised states
        transform (TransformationPipeline) : pipeline used for training
    Returns :
        {'toce', 'soce' : (b, 36, 199, 62), 'ssh' : (b, 1, 199, 62)} numpy arrays
    """
    samples = [transform.__uncall__(g) for g in generated]
    fields = {k : torch.stack([s[k] for s in samples]).cpu().numpy() for k in ['toce', 'soce', 'ssh']}

    # Deal with bottom boundary
    fields['toce'][:,:,1, :] = fields['toce'][:,:,2, :]
    fields['soce'][:,:,1, :] = fields['soce'][:,:,2, :]
    fields['ssh'][:,1, :] = fields['ssh'][:,2, :]
    fields['ssh'] = fields['ssh'][:, None]
    return fields


def save_clean(generated, save_dir, transform) :
    """
        Un-normalise `generated` (b, c, i, j) and save `{toce,soce,ssh}.npy` in `save_dir`.
    """
    fields = unnormalise(generated, transform)
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    for k, v in fields.items() :
        np.save(f'{save_dir}/{k}.npy', v)
    print(f'Saved clean to : {save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Un-normalise generated states")
    parser.add_argument("--generated", type=str, help="path of the .npy of generated normalised states (b, c, i, j)")
    args = parser.parse_args()

    config = TrainingConfig()
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    save_clean(torch.tensor(np.load(args.generated)), args.generated.replace('.npy', '_clean/'), transform)
