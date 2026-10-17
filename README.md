# Length-varying Neural Motion Stitching via Cluster Transition Graph

Official implementation of *Length-varying Neural Motion Stitching via Cluster Transition Graph*, presented at SIGGRAPH Asia 2026.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.37167)
[![Project Page](https://img.shields.io/badge/Project-Page-1F6FEB?logo=googlechrome&logoColor=white)](https://haem-k.github.io/nms/)
[![YouTube](https://img.shields.io/badge/YouTube-Video-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=gVvdhzmmegU)

## ⚙️ Installation

Create and activate the Conda environment from the repository root:

```bash
conda env create -f environment.yaml
conda activate nms
```

The commands below are shown from the repository root and enter `code/` when needed.

## 🚀 Pretrained Models and Data

Download the pretrained VQ-VAE, cluster transition graph, and motion generator from [this link](https://drive.google.com/drive/folders/1duZXV9y_8ejB8zxm5NgwzHde7F_HCZK6?usp=sharing).

Place the downloaded files under `result/` as follows:

```text
result/
├── vqvae/
│   └── cluster_construction/
│       ├── checkpoint/epoch_99.pt
│       ├── codebook/
│       │   ├── codebook.npy
│       │   └── codebook_decoded.npy
│       ├── graph/
│       │   ├── valid_trans_bool.npz
│       │   ├── valid_node_filter.npy
│       │   ├── dist.pkl
│       │   └── all_pair_cost.npy
│       ├── vqvae_datarep.yaml
│       └── vqvae_train.yaml
└── mg/
    └── motion_generator/
        ├── checkpoint/epoch_99.pt
        ├── mg_train.yaml
        ├── gt_datarep.yaml
        ├── guide_datarep.yaml
        ├── out_datarep.yaml
        └── target_datarep.yaml
```

### Motion data

Download the [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) motion dataset and the [Mixamo YBot character](https://www.mixamo.com/).

We used a 22-joint YBot skeleton, retargeted the LAFAN1 motions to this skeleton at 60 FPS, and created mirrored sequences with the `mirrored_` filename prefix using MotionBuilder. The list of 22-joints is available in `code/nms/datarep/vqvae_datarep.yaml`. After retargeting, save the motions as BVH files under `data/raw/bvh`.

The prepared data should follow this layout:

```text
data/
├── character/
│   └── ybot.bvh
├── raw/
│   ├── bvh/                 # source BVH files
│   └── npy/                 # converted local transformation matrices
└── test_input/
    └── test_input_total.yaml
```

To convert prepared BVH files under `data/raw/bvh/` and generate `data/file_splits.yaml`:

```bash
cd code
python 00_prepare_raw_data.py --convert-bvh
```

## 🤖 Inference

After preparing the motion data and downloading the pretrained files, run neural motion stitching on the samples configured in `data/test_input/test_input_total.yaml`:

```bash
cd code
python 40_nms.py
```

Each test input specifies a NumPy motion filename and frame range under `data/raw/npy/`. Results are written to `result/nms/<exp_name>/`; the default `exp_name` in `code/nms/config/nms.yaml` is `nms`. Set `save_bvh: true` to additionally export generated motions as BVH files.


## 🏃‍♂️ Training

To train from scratch, prepare LAFAN1 and YBot first, then run the stages below. Each script loads its corresponding YAML configuration from `code/nms/config/`.

| Script | Default configuration |
| --- | --- |
| `10_generate_vqvae_data.py` | `vqvae_datagen.yaml` |
| `11_train_vqvae.py` | `vqvae_train.yaml` |
| `20_construct_graph.py` | `construct_graph.yaml` |
| `30_generate_mg_data.py` | `mg_datagen.yaml` |
| `31_train_mg.py` | `mg_train.yaml` |
| `40_nms.py` | `nms.yaml` |

Training and inference outputs are created under `<result_dir>/<exp_name>/`. 

### 1. Motion Clustering

Generate the VQ-VAE dataset and train the motion clustering model:

```bash
cd code
python 10_generate_vqvae_data.py
python 11_train_vqvae.py
```

Training saves the checkpoint, codebook, and decoded codebook under `result/vqvae/<exp_name>/`.

### 2. Cluster Transition Graph

After training a new VQ-VAE, set `vqvae_train_config_dir` in `code/nms/config/construct_graph.yaml` to its experiment directory, then construct the graph:

```bash
cd code
python 20_construct_graph.py
```

This script creates the graph directory inside the VQ-VAE experiment directory and generates the transition-connectivity, node-filter, distance-cache, and all-pairs cost files.

### 3. Motion Generator

Set `vqvae_train_config_dir` in `code/nms/config/mg_datagen.yaml` to the VQ-VAE experiment directory. Then generate the motion-generator dataset and train the Transformer motion generator:

```bash
cd code
python 30_generate_mg_data.py
python 31_train_mg.py
```

Outputs are written to `result/mg/<exp_name>/`.

### 4. Neural Motion Stitching

Update `vqvae_train_config_dir` and `mg_train_config_dir` in `code/nms/config/nms.yaml` to point to the new experiment directories, then follow the commands in the Inference section.

## 🙏 Acknowledgements

This project uses the [LAFAN1 dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) introduced with *Robust Motion In-betweening* and the [Mixamo YBot character](https://www.mixamo.com/) from Adobe Mixamo. Please follow the respective dataset and asset licenses when preparing and using these resources.

We thank all researchers who shared their code. This project would not have been possible without these projects:
* [LAFAN1-resolved](https://github.com/orangeduck/lafan1-resolved)
* [fairmotion](https://github.com/facebookresearch/fairmotion)
* [MoMask](https://github.com/EricGuo5513/momask-codes)
* [CVQ-VAE](https://github.com/lyndonzheng/CVQ-VAE)
* [aPyOpenGL](https://github.com/seokhyeonhong/aPyOpenGL)

## 📜 Citation

```bibtex
@article{kim2026length,
  title={Length-varying Neural Motion Stitching via Cluster Transition Graph},
  author={Kim, Haemin and Nam, Junghyun and Hong, Seokhyeon and Tan, Vanessa and Noh, Junyong},
  journal={arXiv preprint arXiv:2609.37167},
  year={2026}
}
```
