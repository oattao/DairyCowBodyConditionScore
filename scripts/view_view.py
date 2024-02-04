import sys
sys.path.append('..')
import json
import glob
import numpy as np
from Kpnet.visualization import visualize_view
from utils.cow import get_cow_regno


with open('./config.json', 'r') as fp:
    config = json.load(fp)
npz_path = config['npz_path']
list_npz = glob.glob(f'{npz_path}/center/*.npz')
for fname in list_npz:
    cow_regno = get_cow_regno(fname)
    for view in ['head', 'tail', 'left', 'right', 'center']:
        npzfile = np.load(f'{npz_path}/{view}/{cow_regno}.npz')
        verts = npzfile['vertices']
        # labels_distance = npzfile['distance_on_the_manifold']
        labels_distance = npzfile['gaussian_distance']
        visualize_view(verts, labels_distance, view)
    stop = input("Stop?")
    if stop in ['Y', 'y']:
        break