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
from ood_slam.configs import register_configs

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

register_configs()

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
    log.info("Starting OOD-SLAM")
    log.info(f"Experiment: {cfg.experiment_name}")
    log.info(f"Seed: {cfg.seed}")
    
    # Check CUDA availability
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Print resolved configuration
    log.info("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    


if __name__ == "__main__":
    main()
