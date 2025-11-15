import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

# Import utilities to register resolvers
from utils.env_vars import get_constant

# Register the environment variable resolver
OmegaConf.register_new_resolver("get_constant", get_constant)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger(__name__)

@hydra.main(
    config_path="configs", 
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
