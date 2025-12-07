# Patched Submitit SLURM launcher configuration
# Adds support for setup commands and additional SLURM parameters

import dataclasses
from typing import Any

import hydra_zen
from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from submitit.slurm.slurm import _make_sbatch_string

# Interesting idea: Create the config based on the signature of that function directly.
_AddedArgumentsConf = hydra_zen.builds(
    _make_sbatch_string,
    populate_full_signature=True,
    hydra_convert="object",
    zen_exclude=["command", "folder", "map_count"],
)


@dataclasses.dataclass
class PatchedSlurmQueueConf(_AddedArgumentsConf, SlurmQueueConf):
    """Adds more SLURM parameters to the config for the SLURM submitit launcher of Hydra."""

    _target_: str = "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"  # type: ignore

    signal_delay_s: int = 120
    """USR1 signal delay before timeout."""

    max_num_timeout: int = 0
    """Maximum number of retries on job timeout.

    Change this only after you confirmed your code can handle re-submission by properly resuming
    from the latest stored checkpoint. check the following for more info on slurm_max_num_timeout
    https://github.com/facebookincubator/submitit/blob/master/docs/checkpointing.md
    """

    additional_parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Useful to add parameters which are not currently available in the plugin.

    Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    """

    array_parallelism: int = 256
    """Maximum number of jobs running in parallel."""

    setup: list[str] | None = None
    """A list of commands to run in sbatch before running srun."""


ConfigStore.instance().store(
    group="hydra/launcher",
    name="patched_submitit_slurm",
    node=PatchedSlurmQueueConf,
    provider="Mila",
)
