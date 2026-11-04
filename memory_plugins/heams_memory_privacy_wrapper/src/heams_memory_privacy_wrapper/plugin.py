"""The privacy wrapper as a memory wrapper plugin.

A memory-system config enables it with a top-level `privacy_config:` block next
to `backend_config`; the wrapper then applies retention and recall policies
around whichever memory system `backend_config` names.
"""

from heams.memory_systems.api import MemoryWrapperPlugin

from .config import PrivacyConfig
from .wrapper import wrap_memory_with_privacy

plugin = MemoryWrapperPlugin(
    config_key="privacy_config",
    config_model=PrivacyConfig,
    wrap=wrap_memory_with_privacy,
)
