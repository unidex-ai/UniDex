# ! In development, not ready for use.

import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
from torch.utils.data._utils.collate import default_collate
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import hydra
import os
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
import gc

from src.dataset.mixture import MixtureDataset

def custom_collate_fn(batch):
    def recursive_collate(data):
        if isinstance(data, np.ndarray):
            return torch.tensor(data)
        elif isinstance(data, dict):
            return {key: recursive_collate(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [recursive_collate(item) for item in data]
        return data

    return default_collate([recursive_collate(item) for item in batch])

def to_device_dtype(d, device, dtype):
    for key, value in d.items():
        if isinstance(value, dict):
            to_device_dtype(value, device, dtype)
        elif isinstance(value, torch.Tensor):
            if key not in ["action_is_pad"]:
                d[key] = value.to(device=device, dtype=dtype)
            else:
                d[key] = value.to(device=device)
        else:
            pass
    return d

class LightningTrainingWrapper(L.LightningModule):
    def __init__(self, policy_config, config: DictConfig):
        super().__init__()
        self.policy_config = policy_config
        self.config = config
        self.policy = None

        self.lr_scheduler = hydra.utils.instantiate(config.scheduler)

    def setup(self, stage: str):
        if stage == 'fit' and self.policy is None:
            self._initialize_and_load()

    def _initialize_and_load(self):
        self.policy = hydra.utils.instantiate(self.policy_config)
        print(f"Process {self.global_rank}: Model instantiated.")

        if self.config.load_checkpoint is None and hasattr(self.policy_config, "pretrained_paligemma"):
            if self.policy_config.pretrained_paligemma is not None:
                print(f"Process {self.global_rank}: Loading pretrained PaliGemma...")
                self.policy.load_pretrained(
                    pretrained_pointcloud_encoder_path=self.policy_config.pointcloud_encoder.pretrained_model_path,
                    pretrained_paligemma=self.policy_config.pretrained_paligemma,
                )

        if self.config.load_checkpoint:
            print(f"Process {self.global_rank}: Loading checkpoint from {self.config.load_checkpoint} to device {self.device}...")
            state_dict = torch.load(self.config.load_checkpoint, weights_only=False, map_location=self.device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            new_state_dict = {}
            for key in list(state_dict.keys()):
                if key.startswith("policy."):
                    new_state_dict[key[len("policy."):]] = state_dict.pop(key)
                else:
                    new_state_dict[key] = state_dict.pop(key)

            self.policy.load_state_dict(new_state_dict)
            print(f"Process {self.global_rank}: Checkpoint loaded.")
            
            del state_dict, new_state_dict
            gc.collect()

    def forward(self, batch):
        return self.policy(batch)[0]

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.policy(batch)
        for key, value in loss_dict.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("learning_rate", self.lr_scheduler(self.global_step), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            target = batch['action']
            pred = self.policy.infer_action(batch)
            val_loss = torch.nn.functional.mse_loss(pred, target)
            self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optimizer_config = dict(self.config.optimizer)
        optimizer_config["params"] = self.policy.parameters()
        optimizer = hydra.utils.instantiate(optimizer_config)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: self.lr_scheduler(step))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(config):
    print(OmegaConf.to_yaml(config))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OmegaConf.set_struct(config, False)
    config.train.timestamp = timestamp
    OmegaConf.set_struct(config, True)
    train_config = config.train
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    wandb_config = train_config.wandb
    wandb_logger = WandbLogger(
        project=wandb_config.project,
        name=wandb_config.run_name,
        config=config_dict,
        notes=timestamp,
    )

    callbacks = []

    checkpoint_config = train_config.checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_config.checkpoint_path,
        filename="checkpoint-{epoch:02d}-{train_loss:.4f}",
        save_top_k=checkpoint_config.save_top_k,
        every_n_epochs=checkpoint_config.save_every_n_epochs,
        save_last=True,
        monitor="val_loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(learning_rate_monitor)

    dataset: MixtureDataset = hydra.utils.instantiate(config.dataset)
    print(len(dataset), "samples in dataset")
    # prompts = []
    # for i in range(4):
    #     for j, seq in enumerate(dataset.datasets[i].sequences):
    #         prompt = seq['prompt']
    #         prompt = ' '.join(prompt.strip().split()[4:])
    #         prompts.append(prompt)
    # for i, seq in enumerate(dataset.datasets[4].sequences):
    #     ep = {
    #         'seq_info': seq,
    #         'zarr_path': seq['zarr_path'],
    #     }
    #     prompt = dataset.datasets[4]._load_prompt(ep)
    #     prompt = ' '.join(prompt.strip().split()[4:])
    #     prompts.append(prompt)
    #     if i % 100 == 0:
    #         print(f"Dataset 4 processed {i} windows")
    # with open('vocab.txt', 'w') as f:
    #     for prompt in prompts:
    #         f.write(prompt + '\n')
    # exit()

    val_ratio = getattr(config.train.dataloader, 'val_ratio', 0.05)
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=train_config.dataloader.batch_size, shuffle=True, num_workers=train_config.dataloader.num_workers, persistent_workers=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_config.dataloader.batch_size, shuffle=False, num_workers=train_config.dataloader.num_workers, persistent_workers=True, collate_fn=custom_collate_fn)

    trainer_config = train_config.trainer
    trainer = L.Trainer(
        devices=trainer_config.devices,
        strategy=trainer_config.strategy,
        max_epochs=trainer_config.max_epochs,
        enable_progress_bar=True,
        gradient_clip_val=trainer_config.gradient_clip_val,
        precision=trainer_config.precision,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        check_val_every_n_epoch= trainer_config.check_val_every_n_epoch,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    os.makedirs(train_config.checkpoint_path, exist_ok=True)
    if dataset.normalizer is not None:
        torch.save(dataset.normalizer.norm_stats, f"{train_config.checkpoint_path}/normalizer-{timestamp}.pt")
    with open(f"{train_config.checkpoint_path}/config-{timestamp}.yaml", "w") as f:
        yaml.dump(config_dict, f)
    training_policy = LightningTrainingWrapper(config.model, train_config)
    
    trainer.fit(training_policy, train_loader, val_loader)

if __name__ == "__main__":
    main()