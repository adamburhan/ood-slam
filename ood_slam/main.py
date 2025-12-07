"""Training script using [Hydra](https://hydra.cc).

This does the following:
1. Parses the config using Hydra;
2. Instantiated the components (trainer / algorithm), optionally datamodule and network;
3. Trains the model;
4. Optionally runs an evaluation loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from hydra_plugins.auto_schema import auto_schema_plugin

import ood_slam
from ood_slam import configs  # Import to trigger config registration
from ood_slam.utils import remote_launcher_plugin  # Import to register patched launcher

from ood_slam.trainer import train


import wandb
    

PROJECT_NAME = ood_slam.__name__
REPO_ROOTDIR = Path(__file__).parent.parent
logger = logging.getLogger(__name__)

# Configure auto schema plugin
auto_schema_plugin.config = auto_schema_plugin.AutoSchemaPluginConfig(
    schemas_dir=REPO_ROOTDIR / ".schemas",
    regen_schemas=False,
    stop_on_error=False,
    quiet=False,
    verbose=False,
    add_headers=False,  # don't fallback to adding headers if we can't use vscode settings file.
)

#register_configs()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger(__name__)

@hydra.main(
    config_path=f"pkg://{PROJECT_NAME}.configs", 
    config_name="config",
    version_base="1.3",  # Updated to newer version
)
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    
    # Instantiate components
    log.info("Setting up data module...")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    
    log.info("Setting up model...")
    model = hydra.utils.instantiate(cfg.model)
    
    # Get data loaders
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    
    log.info(f"Training batches: {len(train_dl)}")
    log.info(f"Validation batches: {len(val_dl) if val_dl else 0}")
    
    # Setup wandb if enabled
    if cfg.use_wandb:
        # Resolve wandb config interpolations
        wandb_config = OmegaConf.to_container(cfg.wandb, resolve=True)
        
        wandb_config.update({
            "model_name": cfg.model._target_.split('.')[-1] if hasattr(cfg.model, '_target_') else "unknown",
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size, 
            "max_epochs": cfg.max_epochs,
            "seed": cfg.seed,
        })
        
        log.info("Initializing Weights & Biases...")
        wandb.init(
            project=wandb_config.get("project", "ood-slam"),
            name=wandb_config.get("name", f"run_{cfg.seed}"),
            config=wandb_config,
            tags=["ood-slam", cfg.model._target_.split('.')[-1] if hasattr(cfg.model, '_target_') else "model"],
        )
        
        # Log the full config as an artifact (skip missing values marked with ???)
        try:
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
        except Exception as e:
            log.warning(f"Could not update wandb config with full cfg: {e}")
        log.info(f"Wandb run: {wandb.run.name}")
    
    # Start training
    log.info("Starting training...")
    results = train(model, train_dl, val_dl, device, cfg)
    
    log.info("Training completed!")
    log.info(f"Final results: {results}")
    
    # Finish wandb run
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
