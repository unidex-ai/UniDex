from typing import Dict, List
from omegaconf import DictConfig

import numpy as np
import torch


def dict_apply(func, d):
    """
    Apply a function to all values in a dictionary recursively.
    If the value is a dictionary, it will apply the function to its values.
    """
    for key, value in d.items():
        if isinstance(value, dict) or isinstance(value, DictConfig):
            dict_apply(func, value)
        elif not isinstance(value, str):
            d[key] = func(value)
    return d


class Normalizer:
    def __init__(
        self,
        norm_stats: Dict[str, Dict[str, np.ndarray]] | None = None,
        norm_type: Dict[str, str] | None = None,
        normalizers: List['Normalizer'] | None = None,
    ):
        if normalizers is not None:
            self.norm_stats = {}
            self.norm_type = {}
            self.merge_normalizers(normalizers)
            self.norm_stats = dict_apply(lambda x: torch.tensor(x).to(torch.float32), self.norm_stats)
        else:
            self.norm_stats = dict_apply(lambda x: np.array(x).astype(np.float32), norm_stats)
            self.norm_type = norm_type or {}

    def merge_normalizers(self, normalizers: List['Normalizer']):
        print(f"Merging {len(normalizers)} normalizers")
        for normalizer in normalizers:
            for key, stats in normalizer.norm_stats.items():
                if key not in self.norm_stats:
                    self.norm_stats[key] = stats
                    self.norm_type[key] = normalizer.norm_type.get(key, "identity")
                else:
                    if self.norm_type[key] == "identity":
                        self.norm_stats[key] = stats
                        self.norm_type[key] = normalizer.norm_type.get(key, "identity")
                    elif self.norm_type[key] == normalizer.norm_type[key]:
                        if self.norm_type[key] == "minmax":
                            max_dim = max(self.norm_stats[key]["max"].shape, stats["max"].shape)[0]
                            # pad to max dimension
                            for dim in ["max", "min"]:
                                self.norm_stats[key][dim] = np.pad(
                                    self.norm_stats[key][dim],
                                    (0, max_dim - self.norm_stats[key][dim].shape[0]),
                                    mode='constant',
                                    constant_values=0
                                )
                                stats[dim] = np.pad(
                                    stats[dim],
                                    (0, max_dim - stats[dim].shape[0]),
                                    mode='constant',
                                    constant_values=0
                                )
                                self.norm_stats[key][dim] = np.minimum(
                                    self.norm_stats[key][dim], stats[dim]
                                ) if dim == "min" else np.maximum(
                                    self.norm_stats[key][dim], stats[dim]
                                )
                        else:
                            raise NotImplementedError(
                                f"Normalization type {self.norm_type[key]} not implemented for merging."
                            )
                    else:
                        raise ValueError(
                            f"Cannot merge different normalization types for key '{key}': "
                            f"{self.norm_type[key]} vs {normalizer.norm_type[key]}"
                        )
                    

    def normalize(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        normalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                if norm_type == "meanstd":
                    mean = self.norm_stats[key]["mean"]
                    std = self.norm_stats[key]["std"]
                    normalized_value = (value - mean) / (std + 1e-6)
                elif norm_type == "std":
                    std = self.norm_stats[key]["std"]
                    normalized_value = value / (std + 1e-6)
                elif norm_type == "minmax":
                    min_val = self.norm_stats[key]["min"]
                    max_val = self.norm_stats[key]["max"]
                    mid_val = (max_val + min_val) / 2
                    normalized_value = (value - mid_val) / (
                        max_val - min_val + 1e-6
                    ) * 2
                elif norm_type == "identity":
                    normalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', and 'identity'."
                    )
                normalized_data[key] = normalized_value
            else:
                # If the key is not in norm_stats, we assume no normalization is needed
                normalized_data[key] = value
        return normalized_data

    def unnormalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        unnormalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                if norm_type == "meanstd":
                    mean = self.norm_stats[key]["mean"]
                    std = self.norm_stats[key]["std"]
                    unnormalized_value = value * (std + 1e-6) + mean
                elif norm_type == "std":
                    std = self.norm_stats[key]["std"]
                    unnormalized_value = value * (std + 1e-6)
                elif norm_type == "minmax":
                    min_val = self.norm_stats[key]["min"]
                    max_val = self.norm_stats[key]["max"]
                    mid_val = (max_val + min_val) / 2
                    unnormalized_value = value / 2 * (
                        max_val - min_val + 1e-6
                    ) + mid_val
                elif norm_type == "identity":
                    unnormalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', and 'identity'."
                    )
                unnormalized_data[key] = unnormalized_value
            else:
                # If the key is not in norm_stats, we assume no unnormalization is needed
                unnormalized_data[key] = value
        return unnormalized_data
