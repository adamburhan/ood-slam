"""Configuration registration for ood-slam project."""

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# Import to trigger registration
from ood_slam.utils.remote_launcher_plugin import RemoteSlurmQueueConf
from ood_slam.utils.env_vars import get_constant

# Register resolvers
OmegaConf.register_new_resolver("get_constant", get_constant)


def register_configs():
    """Adds all configs to the Hydra Config store."""
     
    ConfigStore.instance().store(
        group="hydra/launcher",
        name="remote_submitit_slurm",
        node=RemoteSlurmQueueConf,
        provider="Mila",
    )

# Auto-register when imported
register_configs()