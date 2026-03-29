# Dataset

## Pre-train Datasets

All pre-train datasets will be under the `data/` directory by default. After all processing, it will take up to 80TB of disk space. So make full use of soft links to avoid `No space Left on Device`.

Before starting, download datasets annotations from our Hugging Face repository:
```bash
hf download UniDex-ai/UniDex --include dataset_annotations/* --local-dir .
```


### H2o (2 Hands and Objects)

Download all `subjectX_ego_v1_1.tar.gz` (X=1,2,3,4) files from [H2o official website](https://h2odataset.ethz.ch/) and unpack them under `data/H2o/all_img`. After unpacking, the directory structure should look like:
```
H2o/
└── all_img/
    ├── subject1_ego/
    ├── subject2_ego/
    ├── subject3_ego/
    └── subject4_ego/
```

For language instructions, run the following command:
```bash
# Assuming you are in the root directory of the project
cd data/H2o
cp ../../dataset_annotations/H2o_annotations.tar.gz .
tar -xzvf H2o_annotations.tar.gz
rm H2o_annotations.tar.gz
cd ../..
```

### HOI4D (4D Egocentric Dataset for Category-Level Human-Object Interaction)

From [HOI4D official website](https://hoi4d.github.io/), download `HOI4D_color`, `HOI4D_depth`, `HOI4D_annotation` and unpack them under `data/HOI4D/HOI4D_release`. Also download `HOI4D_Handpose` and `HOI4D_cameras` and unpack them under `data/HOI4D/Hand_pose` and `data/HOI4D/camera` respectively. After unpacking, the directory structure should look like:
```
HOI4D/
├── HOI4D_release/
│   ├── ZY20210800001/
│   │   ├── H1/
│   │   │   ├── C1/
│   │   │   │   ├── N01/
│   │   │   │   │   ├── S000/
│   │   │   │   │   │   ├── s01/
│   │   │   │   │   │   │   ├── T1/
│   │   │   │   │   │   │   │   ├── align_rgb/
│   │   │   │   │   │   │   │   ├── align_depth/
│   │   │   │   │   │   │   │   ├── 2Dseg/
│   │   │   │   │   │   │   │   └── ...
│   │   │   │   │   │   │   └── ...
│   │   │   │   │   │   └── ...
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── camera/
└── Hand_pose/
```

Then run the following command to unpack the rgb and depth images from video files:
```bash
# Assuming you are in the root directory of the project
python scripts/process_HOI4D.py
```

### Hot3D (An egocentric dataset for 3D hand and object tracking)

Follow instructions from [Hot3D github repository](https://github.com/facebookresearch/hot3d) to download the dataset and put them under `data/hot3d/`. After unpacking, the directory structure should look like:
```hot3d/
├── P0001_4bf4e21a/
├── ...
└── P0020_ff537251/
```

We manually labeled all language instructions for Hot3D. To add them to the dataset, run the following command:
```bash
# Assuming you are in the root directory of the project
cd data/hot3d
cp ../../dataset_annotations/hot3d_prompts.tar.gz .
tar -xzvf hot3d_prompts.tar.gz
rm hot3d_prompts.tar.gz
cd ../..
```

### Taco (Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding)

Download the Taco dataset from [Taco dataset](https://www.dropbox.com/scl/fo/8w7xir110nbcnq8uo1845/AOaHUxGEcR0sWvfmZRQQk9g?rlkey=xnhajvn71ua5i23w75la1nidx&e=2&st=9t8ofde7&dl=0), including `Egocentric_RGB_Videos`, `Egocentric_Depth_Videos`, `Egocentric_Camera_Parameters` and `Hand_Poses`. After unpacking, the directory structure should look like:
```Taco/
├── Egocentric_RGB_Videos/
├── Egocentric_Depth_Videos/
├── Egocentric_Camera_Parameters/
└── Hand_Poses/
```

Then run the following command to process the Taco dataset:
```bash
# Assuming you are in the root directory of the project
python scripts/process_Taco.py
```

## Staged Results

If you have followed the intructions above, you should have your `data/` directory structured as follows:
```
data/
├── H2o/
│   ├── all_img/
│   │   ├── subject1_ego/
│   │   ├── subject2_ego/
│   │   ├── subject3_ego/
│   │   └── subject4_ego/
│   └── annotation/
├── HOI4D/
│   ├── HOI4D_release/
│   ├── camera/
│   └── Hand_pose/
├── hot3d/
│   ├── P0001_4bf4e21a/
│   ...
│   └── P0020_ff537251/
└── Taco/
    ├── Egocentric_RGB_Videos/
    ├── Egocentric_Depth_Videos/
    ├── Egocentric_Camera_Parameters/
    └── Hand_Poses/
```

## Retarget Robotic Hands
To generate retargeted robotic hand data from the above datasets, run the following command:
```bash
python HandAdapter/hand_processor.py --hand_type {Allegro, Ability, Inspire, Leap, Oymotion, Shadow, Wuji, Xhand} --dataset {H2o, HOI4D, Hot3D, Taco} --cont
```
You can add `--randperm` to randomly permute the data order for parallel processing. The retargeted data will be saved under `data/${dataset}/retarget_RGBD/${sequence_relative_path}/${hand_type}.h5` by default.

## Add New Robotic Hands
First place your new hand urdf files under `HandAdapter/urdf/base`, where left and right hand urdf files should be named as `left/main.urdf` and `right/main.urdf` respectively. Then add a `config.json` file under `HandAdapter/urdf/${YourHandName}/config.json` to specify the parameters for your new hand, following the format of existing config files.

Then ensure the coordinate frame of the new hand URDF is set so that the X-axis points into the palm and the Z-axis points along the fingers. Also add the new hand type to the `HAND_TYPES` list in `HandAdapter/visualizer.py`.

Finally run `python HandAdapter/visualizer.py` and adjust inverse kinematics parameters of the new hand on all datasets in the web interface until satisfactory retargeting results are achieved. Now you can use the new hand type in `hand_processor.py` to generate retargeted data.