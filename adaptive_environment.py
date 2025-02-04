# adaptive_environment.py
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import uuid
import numpy as np
import math
import random
import perlin


class ResourceType(Enum):  # From agent-architecture.py
    ENERGY = "energy"
    INFORMATION = "information"
    MATERIALS = "materials"


@dataclass  # From agent-architecture.py
class Resource:
    type: ResourceType
    quantity: float
    position: Tuple[int, int]
    complexity: float  # How difficult it is to extract/process
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass  # From agent-architecture.py
class EnvironmentalState:
    """Represents the current state of the environment"""
    resources: List[Resource]
    threats: List[Tuple[int, int]]  # Positions of hazards/threats
    time_step: int
    complexity_level: float  # Overall difficulty/complexity of current state
    agents: List['AdaptiveAgent'] = field(
        default_factory=list)  # Added agents to state
    size: Tuple[int, int] = (0, 0)  # Default size if none provided


# From agent-architecture.py + adaptive_environment.py (Base Environment Class)
class AdaptiveEnvironment:
    def __init__(self, size: Tuple[int, int], complexity: float):
        self.size = size
        self.complexity = complexity
        self.current_state = EnvironmentalState(
            resources=[],
            threats=[],
            time_step=0,
            complexity_level=complexity
            size=size,
            agents=[]
        )

    def step(self, agents: List['AdaptiveAgent']) -> EnvironmentalState:
        """Advance the environment by one time step"""
        self._update_state()
        return self.current_state

    def _update_state(self):
        """Update environment state - resource dynamics, threats, etc."""
        self.current_state.time_step += 1

        # Example: Resource regeneration (simplified)
        for resource in self.current_state.resources:
            if resource.quantity < 100:
                resource.quantity += random.uniform(0, 0.5)

        # Example: Threat movement (simplified random walk)
        for i in range(len(self.current_state.threats)):
            threat_pos = self.current_state.threats[i]
            movement = self._calculate_threat_movement(threat_pos)
            new_threat_pos = (
                threat_pos[0] + movement[0], threat_pos[1] + movement[1])
            # Keep threats within bounds
            self.current_state.threats[i] = (max(0, min(
                self.size[0]-1, int(new_threat_pos[0]))), max(0, min(self.size[1]-1, int(new_threat_pos[1]))))

        # Example: New resource spawning (probability based on complexity)
        if random.random() < 0.01 * self.current_state.complexity_level:
            self.current_state.resources.append(
                Resource(
                    type=random.choice(list(ResourceType)),
                    quantity=random.uniform(10, 50),
                    position=(random.randint(
                        0, self.size[0]-1), random.randint(0, self.size[1]-1)),
                    complexity=random.uniform(0.1, 0.9)
                )
            )

    def _calculate_threat_movement(self, threat_pos: Tuple[float, float]) -> Tuple[float, float]:
        return (random.uniform(-1, 1), random.uniform(-1, 1))

    def _generate_perlin_noise(self, size: Tuple[int, int], scale: float) -> np.ndarray:
        return np.zeros(size)

    def _initialize_weather(self) -> Dict:
        return {}

    def _update_weather(self, current_weather: Dict) -> Dict:
        return current_weather

    def _get_terrain_factor(self, position: Tuple[int, int]) -> float:
        return 1.0

    def _get_weather_factor(self, position: Tuple[int, int]) -> float:
        return 1.0

    def _calculate_terrain_gradient(self, position: Tuple[int, int]) -> Tuple[float, float]:

        return (0.0, 0.0)

    def _find_nearest_agent(self, pos: Tuple[float, float]) -> Optional['AdaptiveAgent']:
        return None
