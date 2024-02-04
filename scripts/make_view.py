import sys
sys.path.append('..')
import json
import os
import glob
import shutil
from Kpnet.labelling import process_view


with open('./config.json', 'r') as fp:
    config = json.load(fp)
rump_path = config['rump_path']
npz_path = config['npz_path']
if os.path.exists(npz_path):
    shutil.rmtree(npz_path)
os.mkdir(npz_path)
for view in ['head', 'tail', 'left', 'right', 'center']:
    os.mkdir(f"{npz_path}/{view}")

list_pcd = glob.glob(f'{rump_path}/*.ply')
print(len(list_pcd))
for fname in list_pcd:
    print("Processing: ", fname)
    process_view(fname, npz_path, num_points=20000, epsilon=2.0)