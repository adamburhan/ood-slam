import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
import ood_slam

# Import configurations (registers resolvers and launcher plugin)
import ood_slam.configs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger(__name__)

PROJECT_NAME = ood_slam.__name__

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
