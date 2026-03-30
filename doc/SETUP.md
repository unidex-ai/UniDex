# Setup

This document provides instructions for setting up the conda environment and downloading necessary models. For details about pre-train datasets, please refer to [DATASET.md](DATASET.md).

## Conda Environment Setup

To create and activate the conda environment, run the following commands:

```bash
conda create -n unidex python=3.10 -y
conda activate unidex
pip install -r requirements.txt
pip install -e . # Install src as a package
```

`pytorch3d` and `manopth` need to be installed separately. `pytorch3d` depends on the specific CUDA version of your system. First time installation may take some time to compile from source. 

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

(Optional) `sam2` and `WiLoR` also need to be installed if you want to use the `Taco` dataset in pretraining.
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
git clone https://github.com/rolpotamias/WiLoR.git
export PYTHONPATH=$(pwd)/WiLoR:$PYTHONPATH # Add WiLoR to PYTHONPATH since it cannot be installed as a package
# It is recommended to add the above line to your .bashrc or .zshrc for future use.
```
## Downloading Models

### 1. Uni3D Pretrained Point-Cloud Encoder
First download the Uni3D pretrained point-cloud encoder.
```bash
hf download BAAI/Uni3D # Download from Hugging Face
# or
modelscope download --model=BAAI/Uni3D # Download from ModelScope. Recommended for users in China mainland.
```

Then move the downloaded model to the appropriate directory by running:
```bash
bash scripts/move_pretrained_uni3d.sh
```

### 2. Paligemma Tokenizer
Skip if you have access to the gated Hugging Face model `google/paligemma-3b-pt-224`.

If you encounter authentication issues when downloading the Paligemma tokenizer from Hugging Face, you can download it from ModelScope instead:
```bash
modelscope download --model=AI-ModelScope/paligemma-3b-pt-224
mkdir -p ~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224
mv ~/.cache/modelscope/hub/models/AI-ModelScope/paligemma-3b-pt-224/* ~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/
```
Additionally, you need to change huggingface model id to absolute path in `config/model/unidex.yaml` and `config/model/unidex_inference.yaml`:
```yaml
pretrained_model_path: google/paligemma-3b-pt-224
tokenizer_path: google/paligemma-3b-pt-224  
# change to absolute path ~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224
```

### 3.Mano Model
- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format `mano_v*_*.zip`). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the `models` folder into the `manopth/mano` folder
- Your folder structure should look like this:
```
manopth/
  mano/
    models/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
      ...
  manopth/
    __init__.py
    ...
```

### (Optional) 4. SAM2 and WiLoR
Download the SAM2 and WiLoR pretrained models by running:
```bash
bash scripts/download_sam2_and_WiLoR.sh
```