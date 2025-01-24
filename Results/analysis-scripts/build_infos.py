import tarfile, collections, numpy as np, io

info_file='/Volumes/LoCe/oceandata/Dino-Fusion/dino_1_4_degree_coarse_130924.tar'

tar = tarfile.open(info_file)

target_path='infos/'
max_return = 9

infos = collections.defaultdict(dict)
while max_return > 0 :
    member = tar.next()
    if member.path.startswith(target_path):
        feature, metric, _ = member.name.replace('infos/', '').split('.')
        infos[feature][metric] = np.load(io.BytesIO(tar.extractfile(member).read()))
        max_return -= 1

# Add to existing tar
with tarfile.open(info_file, 'a') as tar:
    tar.add('add_infos/soce/min.npy', 'infos/min.soce.npy')

# Clean up temp file
import os
os.remove('temp.npy')
