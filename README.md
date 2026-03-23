# UniDex

<p align="center">
  <img src="assets/new_teaser.png" alt="UniDex teaser" width="100%">
</p>

<p align="center">
  <strong>Official implementation of the CVPR 2026 paper</strong><br>
  <strong>UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos</strong>
</p>

<p align="center">
  <a href="assets/UniDex_Arxiv.pdf">
    <img src="assets/badges/arxiv.svg" alt="arXiv badge">
  </a>
  <a href="assets/UniDex_Arxiv.pdf">
    <img src="assets/badges/paper.svg" alt="Paper badge">
  </a>
  <a href="https://unidex-ai.github.io/">
    <img src="assets/badges/project-page.svg" alt="Project Page badge">
  </a>
  <a href="https://huggingface.co/UniDex-ai/UniDex">
    <img src="assets/badges/model.svg" alt="Model badge">
  </a>
</p>

<p align="center">
  <a href="assets/new_teaser.pdf">Teaser PDF</a>
</p>

UniDex provides the codebase for dataset preparation, hand retargeting, pre-training, and real-world post-training for universal dexterous hand control from egocentric human videos.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Retargeting](#retargeting)
- [Adding a New Hand](#adding-a-new-hand)
- [Pre-training](#pre-training)
- [Finetuning](#finetuning)
- [Checkpoints and Model Assets](#checkpoints-and-model-assets)
- [Citation](#citation)

## Overview

This repository includes:

- environment setup and dependency instructions
- dataset preparation for H2O, HOI4D, Hot3D, and Taco
- retargeting from human hand motion to multiple robot hands
- pre-training and real-world post-training pipelines

## Setup

Detailed environment instructions are available in [doc/SETUP.md](doc/SETUP.md). A minimal setup looks like:

```bash
conda create -n unidex python=3.10 -y
conda activate unidex
pip install -r requirements.txt
pip install -e .
```

You also need to install `pytorch3d` and `manopth` separately:

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..

git clone https://github.com/hassony2/manopth.git
cd manopth
pip install -e .
cd ..
```

Then download the required pretrained assets:

- Uni3D point-cloud encoder
- PaliGemma tokenizer and weights
- MANO hand model
- optionally `SAM2` and `WiLoR` for full Taco preprocessing

Please refer to [doc/SETUP.md](doc/SETUP.md) for the exact commands and paths.

## Dataset Preparation


Before processing any dataset, download the released dataset annotations:

```bash
hf download UniDex-ai/UniDex --include dataset_annotations/* --local-dir .
```

### H2O

Download all `subjectX_ego_v1_1.tar.gz` files from the [H2O website](https://h2odataset.ethz.ch/) and unpack them under `data/H2o/all_img`.

Then add the released language annotations:

```bash
cd data/H2o
cp ../../dataset_annotations/H2o_annotations.tar.gz .
tar -xzvf H2o_annotations.tar.gz
rm H2o_annotations.tar.gz
cd ../..
```

### HOI4D

Download `HOI4D_color`, `HOI4D_depth`, `HOI4D_annotation`, `HOI4D_Handpose`, and `HOI4D_cameras` from the [HOI4D website](https://hoi4d.github.io/) and place them under `data/HOI4D/`.

Then unpack RGB and depth images from the raw videos:

```bash
python scripts/process_HOI4D.py
```

### Hot3D

Follow the instructions from the [Hot3D repository](https://github.com/facebookresearch/hot3d) and place the downloaded sequences under `data/hot3d/`.

We also release manually labeled language instructions for Hot3D:

```bash
cd data/hot3d
cp ../../dataset_annotations/hot3d_prompts.tar.gz .
tar -xzvf hot3d_prompts.tar.gz
rm hot3d_prompts.tar.gz
cd ../..
```

### Taco

Download `Egocentric_RGB_Videos`, `Egocentric_Depth_Videos`, `Egocentric_Camera_Parameters`, and `Hand_Poses` from the [Taco download page](https://www.dropbox.com/scl/fo/8w7xir110nbcnq8uo1845/AOaHUxGEcR0sWvfmZRQQk9g?rlkey=xnhajvn71ua5i23w75la1nidx&e=2&st=9t8ofde7&dl=0), place them under `data/Taco/`, and run:

```bash
python scripts/process_Taco.py
```

### Expected Data Layout

After dataset preparation, the directory should look like:

```text
data/
├── H2o/
│   ├── all_img/
│   └── annotation/
├── HOI4D/
│   ├── HOI4D_release/
│   ├── camera/
│   └── Hand_pose/
├── hot3d/
└── Taco/
    ├── Egocentric_RGB_Videos/
    ├── Egocentric_Depth_Videos/
    ├── Egocentric_Camera_Parameters/
    └── Hand_Poses/
```

## Retargeting

To convert the processed human datasets into robot-hand trajectories, run:

```bash
python HandAdapter/hand_processor.py \
  --hand_type {Allegro,Ability,Inspire,Leap,Oymotion,Shadow,Wuji,Xhand} \
  --dataset {H2o,HOI4D,Hot3D,Taco} \
  --cont
```

You can add `--randperm` for random permutation during parallel processing. Retargeted results are saved to:

```text
data/${dataset}/retarget_RGBD/${sequence_relative_path}/${hand_type}.h5
```

## Adding a New Hand

To add a new dexterous hand:

- put the URDF assets under `HandAdapter/urdf/base`
- name the left and right hand files as `left/main.urdf` and `right/main.urdf`
- add a matching `config.json` following the format of the existing hands
- add the new hand name to `HAND_TYPES` in `HandAdapter/visualizer.py`
- use `python HandAdapter/visualizer.py` to tune IK parameters before batch retargeting

## Pre-training

After setting up the datasets and pretrained assets, launch UniDex pre-training with the default config:

```bash
python train.py
```

The default setup in [config/train.yaml](config/train.yaml) uses:

- `8` GPUs
- `batch_size = 4`
- `accumulate_grad_batches = 4`
- `max_epochs = 32`

If you only want to finetune from the released checkpoints, you can skip the full pre-training dataset setup.

## Finetuning

Real-world post-training is launched with:

```bash
python finetune.py
```

Before running it, update [config/finetune.yaml](config/finetune.yaml) to point to:

- your pretrained checkpoint
- your real-world dataset
- your preferred run name and hardware configuration

The default finetuning config uses `2` GPUs and loads a pretrained checkpoint from `train.load_checkpoint`.

## Checkpoints and Model Assets

We provide UniDex checkpoints and released assets on [Hugging Face](https://huggingface.co/UniDex-ai/UniDex).

## Citation

If you find UniDex useful, please cite:

```bibtex
@inproceedings{zhang2026unidex,
  title={UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos},
  author={Zhang, Gu and Xu, Qicheng and Zhang, Haozhe and Ma, Jianhan and He, Long and Bao, Yiming and Ping, Zeyu and Yuan, Zhecheng and Lu, Chenhao and Yuan, Chengbo and Liang, Tianhai and Tian, Xiaoyu and Shao, Maanping and Zhang, Feihong and Ding, Mingyu and Gao, Yang and Zhao, Hao and Zhao, Hang and Xu, Huazhe},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
