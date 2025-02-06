from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class NetworkArchitecture:
    input_size: int
    output_size: int
    memory_size: int
    hidden_layers: int

class IGeneticCore(ABC):
    @abstractmethod
    def get_network_architecture(self) -> NetworkArchitecture:
        pass

class IAdaptiveNetwork(ABC):
    @abstractmethod
    def adapt_to_genetics(self, genetic_core: IGeneticCore) -> None:
        pass