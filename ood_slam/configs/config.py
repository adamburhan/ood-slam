"""Base configuration dataclass for ood-slam."""

from dataclasses import dataclass, field
from typing import Any
import random


@dataclass
class Config:
    """Base configuration for ood-slam experiments."""
    # Model configuration
    model: Any
    
    # Data configuration  
    data: Any
    
    preprocessing: Any
    
    experiment: Any
    
    # Experiment settings
    name: str = "default"
    seed: int = field(default_factory=lambda: random.randint(0, int(1e5)))
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 10
    
    # Environment variables
    scratch_dir: str = "${get_constant:SCRATCH}"
    slurm_tmpdir: str = "${get_constant:SLURM_TMPDIR}"
    checkpoint_dir: str = "./checkpoints" 
    
    # Wandb settings
    use_wandb: bool = True
    wandb: dict[str, Any] = field(default_factory=lambda: {
        "project": "ood-slam",
        "name": "${name}_${now:%Y%m%d_%H%M%S}",
        "entity": None,  # Set to your wandb username/team if needed
        "group": None,   # For grouping related experiments
        "job_type": "train",
        "notes": "",
        "tags": []
    })
    
    # Debug mode
    debug: bool = False
    
    
    # Trainer configuration
    trainer: dict[str, Any] = field(default_factory=lambda: {
        "accelerator": "auto",
        "devices": "auto", 
        "max_epochs": "${max_epochs}"
    })
