import sys
sys.path.append('..')
import json
import os
import glob
import shutil
import numpy as np
from utils.cow import get_cow_regno


with open('./config.json', 'r') as fp:
    config = json.load(fp)
rump_path = config['rump_path']
npz_path = config['npz_path']
full_vew_path = f"{npz_path}/full"
if os.path.exists(full_vew_path):
    shutil.rmtree(full_vew_path)
os.mkdir(full_vew_path)
for view in ['head', 'tail', 'left', 'right', 'center']:
    pass

list_npz = glob.glob(f'{npz_path}/center/*.npz')
print(len(list_npz))
for fname in list_npz:
    cow_regno = get_cow_regno(fname)
    print("Processing: ", cow_regno)
    distance_to_manifold = []
    distance_gaussian = []
    for view in ['head', 'tail', 'left', 'right', 'center']:
        npzfile = np.load(f'{npz_path}/{view}/{cow_regno}.npz')
        if view == 'center':
            verts = npzfile['vertices']
            normals = npzfile['normals']
        distance_to_manifold.append(npzfile['distance_on_the_manifold'])
        distance_gaussian.append(npzfile['gaussian_distance'])
    distance_to_manifold = np.hstack(distance_to_manifold)
    distance_gaussian = np.hstack(distance_gaussian)
    np.savez(f"{full_vew_path}/{cow_regno}",
        vertices=verts,
        normals=normals,
        distance_on_the_manifold=distance_to_manifold,
        gaussian_distance=distance_gaussian,
    )