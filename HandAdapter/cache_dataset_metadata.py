"""
Cache dataset metadata for faster loading later
"""

import os
import copy
import json
import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import cv2
from dataclasses import dataclass
import hydra
import argparse
import h5py
import random
from tqdm import tqdm

DATASET_CONFIG = {
    "H2o": {
        "_target_": "dataset.H2o.H2oDataset",
        "data_dir": "data/H2o",
        "load_pcd": False,
    },
    "HOI4D": {
        "_target_": "dataset.HOI4D.HOI4DDataset", 
        "data_dir": "data/HOI4D",
        "load_pcd": False,
    },
    "hot3d": {
        "_target_": "dataset.hot3d.hot3dDataset",
        "data_dir": "data/hot3d",
        "load_pcd": False,
    },
    "Taco": {
        "_target_": "dataset.Taco.TacoDataset",
        "data_dir": "data/Taco",
        "load_pcd": False,
    }
}

def main():
    if not os.path.exists("HandAdapter/dataset_cache"):
        os.makedirs("HandAdapter/dataset_cache")
    for name, cfg in DATASET_CONFIG.items():
        dataset = hydra.utils.instantiate(cfg)
        dataset.save(f"HandAdapter/dataset_cache/{name}.pkl")
        print(f"Cached dataset metadata for {name} at HandAdapter/dataset_cache/{name}.pkl")

if __name__ == "__main__":
    main()