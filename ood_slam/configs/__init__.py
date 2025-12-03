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

def register_configs():
    from ood_slam.utils.remote_launcher_plugin import RemoteSlurmQueueConf
    """Adds all configs to the Hydra Config store."""
     
    ConfigStore.instance().store(
        group="hydra/launcher",
        name="remote_submitit_slurm",
        node=RemoteSlurmQueueConf,
        provider="Mila",
    )

# # Auto-register when imported
# register_configs()

__all__ = [
    "Config",
]