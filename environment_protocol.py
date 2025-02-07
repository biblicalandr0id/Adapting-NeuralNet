from typing import Protocol, Tuple, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptive_environment import Resource, EnvironmentalState
    from agent import AdaptiveAgent

class EnvironmentProtocol(Protocol):
    def get_state(self) -> 'EnvironmentalState':
        ...
    def get_resources(self) -> List['Resource']:
        ...
    def get_agents(self) -> List['AdaptiveAgent']:
        ...
    def get_size(self) -> Tuple[int, int]:
        ...