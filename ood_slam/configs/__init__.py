"""Configuration registration for ood-slam project."""

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# Import to trigger registration

from ood_slam.utils.env_vars import get_constant
from ood_slam.configs.config import Config

# Register resolvers
OmegaConf.register_new_resolver("get_constant", get_constant)

# Store base config
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


__all__ = [
    "Config",
]