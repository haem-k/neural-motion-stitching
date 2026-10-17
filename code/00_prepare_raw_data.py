import argparse
import yaml

import nms.utils.utils as utils
from nms.training.process import convert_bvh_to_npy

# Load all motion data, split into train / val / test
import_motion_dir = "../data/raw/bvh"
npy_motion_dir = "../data/raw/npy"
parser = argparse.ArgumentParser()
parser.add_argument("--convert-bvh", action="store_true")
args = parser.parse_args()
if args.convert_bvh:
    convert_bvh_to_npy(import_motion_dir, npy_motion_dir)
files = utils.list_files_in_dir(npy_motion_dir)

val_files = []
train_files = []
for f in files:
    if "obstacles" in f.lower():
        continue
    filename = f[:-4]
    npy_filename = f[:-4] + ".npy"
    
    if filename.endswith('subject5'):
        val_files.append(npy_filename)
    else:
        train_files.append(npy_filename)

splits = {
    "train": train_files,
    "val": val_files
}

export_dir = "../data"
with open(f"{export_dir}/file_splits.yaml", "w") as yaml_file:
    yaml.dump(splits, yaml_file)