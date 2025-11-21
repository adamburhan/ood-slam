"""Base configuration dataclass for ood-slam."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    """Base configuration for ood-slam experiments."""
    
    # Experiment settings
    experiment_name: str = "ood_slam_deepvo"
    name: str = "${experiment_name}"
    seed: int = 42
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 10
    
    # Environment variables
    scratch_dir: str = "${get_constant:SCRATCH}"
    slurm_tmpdir: str = "${get_constant:SLURM_TMPDIR}"
    
    # Wandb settings
    use_wandb: bool = True
    wandb: dict[str, Any] = field(default_factory=lambda: {
        "project": "ood-slam",
        "name": "${experiment_name}_${now:%Y%m%d_%H%M%S}"
    })
    
    # Debug mode
    debug: bool = False
    
    # Model configuration
    model: dict[str, Any] = field(default_factory=lambda: {
        "rnn_hidden_size": 1000,
        "conv_dropout": [0.2] * 8 + [0.5],
        "rnn_dropout_out": 0.5,
        "rnn_dropout_between": 0,
        "batch_norm": True,
        "pretrained_flownet": None
    })
    
    # Data configuration  
    data: dict[str, Any] = field(default_factory=lambda: {
        "_target_": "ood_slam.data.image_seq_datamodule.ImageSequenceDataModule",
        "data_dir": "${get_constant:DATA_DIR}/KITTI",
        "train_sequences": ["00", "01", "02", "05", "08", "09"],
        "valid_sequences": ["04", "06", "07", "10"],
        "img_w": 608,
        "img_h": 184,
        "img_means": [0.19007764876619865, 0.15170388157131237, 0.10659445665650864],
        "img_stds": [0.2610784009469139, 0.25729316928935814, 0.25163823815039915],
        "resize_mode": "rescale",
        "minus_point_5": True,
        "seq_len": [5, 7],
        "sample_times": 3,
        "batch_size": "${batch_size}",
        "num_workers": 4,
        "pin_memory": True,
        "use_cache": True,
        "cache_dir": None
    })
    
    # Trainer configuration
    trainer: dict[str, Any] = field(default_factory=lambda: {
        "accelerator": "auto",
        "devices": "auto", 
        "max_epochs": "${max_epochs}"
    })
