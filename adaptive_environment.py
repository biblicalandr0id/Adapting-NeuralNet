# adaptive_environment.py
# Template for other modules
# Standard library imports
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
import json
import uuid
import logging

# Third-party imports
import numpy as np
import torch
import pygame

# Local application imports
from genetics import GeneticCore
from neural_networks import NeuralAdaptiveNetwork
# ... other local imports as needed

# Module logger
logger = logging.getLogger(__name__)

from enum import Enum
from typing import TYPE_CHECKING
from predator import Predator  # Changed from relative import to absolute

if TYPE_CHECKING:
    from agent import AdaptiveAgent

class ResourceType(Enum):  # From agent-architecture.py
    ENERGY = "energy"
    INFORMATION = "information"
    MATERIALS = "materials"


@dataclass  # From agent-architecture.py
class Resource:
    type: str
    quantity: float
    position: Tuple[float, float]
    complexity: float  # How difficult it is to extract/process
    regeneration_rate: float    # How quickly it replenishes
    max_quantity: float         # Maximum capacity
    depletion_threshold: float  # Level at which regeneration speeds up
    quality: float             # Value multiplier
    seasonal_factor: float     # How it varies with environment cycles
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class EnvironmentalState:
    """Represents the current state of the environment"""
    resources: List[Resource] = field(default_factory=list)
    predators: List[Predator] = field(default_factory=list)
    time_step: int = 0
    complexity_level: float = 0.5
    agents: List['AdaptiveAgent'] = field(default_factory=list)
    size: Tuple[int, int] = (0, 0)
    weather_conditions: Dict = field(default_factory=dict)


class PredatorType(Enum):
    HUNTER = "hunter"      # Actively seeks agents
    AMBUSHER = "ambusher" # Waits to ambush passing agents
    SCAVENGER = "scavenger" # Follows from a distance, attacks weak agents

@dataclass
class Predator:
    type: PredatorType
    position: Tuple[float, float]
    strength: float         # Attack power
    speed: float           # Movement speed
    perception: float      # Detection range
    stamina: float         # Energy for pursuing
    hunt_strategy: str     # Current behavioral state
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class AdaptiveEnvironment:
    def __init__(self, size: Tuple[int, int], complexity: float):
        self.size = size
        self.complexity = complexity
        self.current_state = EnvironmentalState(
            resources=[],
            predators=[],  # Now stores AdaptivePredator instances
            time_step=0,
            complexity_level=complexity,
size=size
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
        for i in range(len(self.current_state.predators)):  # Changed from threats
            predator_pos = self.current_state.predators[i]  # Changed from threats
            movement = self._calculate_threat_movement(predator_pos.position)  # Changed from threats
            new_predator_pos = (
                predator_pos.position[0] + movement[0], predator_pos.position[1] + movement[1])  # Changed from threats
            # Keep threats within bounds
            self.current_state.predators[i].position = (max(0, min(
                self.size[0]-1, int(new_predator_pos[0]))), max(0, min(self.size[1]-1, int(new_predator_pos[1]))))  # Changed from threats

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

    def add_resources(self, count: int, resource_type: Optional[ResourceType] = None) -> None:
        """Add new resources with complex properties"""
        for _ in range(count):
            position = (
                random.uniform(0, self.size[0]),
                random.uniform(0, self.size[1])
            )
            
            # Generate resource properties based on type and environment complexity
            if resource_type is None:
                resource_type = random.choice(list(ResourceType))
                
            # Base properties modified by environment complexity
            complexity_factor = self.complexity * random.uniform(0.8, 1.2)
            base_quantity = random.uniform(20.0, 100.0) * complexity_factor
            
            new_resource = Resource(
                type=resource_type,
                quantity=base_quantity,
                position=position,
                complexity=random.uniform(0.2, 0.8) * complexity_factor,
                regeneration_rate=random.uniform(0.1, 0.5) * (1 - complexity_factor),
                max_quantity=base_quantity * 1.5,
                depletion_threshold=base_quantity * 0.3,
                quality=random.uniform(0.5, 1.5) * complexity_factor,
                seasonal_factor=random.uniform(0.7, 1.3)
            )
            
            self.current_state.resources.append(new_resource)
            logger.info(f"Added {resource_type.name} resource: quality={new_resource.quality:.2f}, "
                       f"complexity={new_resource.complexity:.2f}")

    def add_predators(self, count: int, predator_type: Optional[PredatorType] = None) -> None:
        """Add predators to the environment"""
        for _ in range(count):
            position = (
                random.uniform(0, self.size[0]),
                random.uniform(0, self.size[1])
            )
            
            if predator_type is None:
                predator_type = random.choice(list(PredatorType))
                
            # Adjust attributes based on predator type
            base_strength = random.uniform(0.5, 1.0)
            if predator_type == PredatorType.HUNTER:
                speed, perception = 1.2, 1.1
                strategy = "patrol"
            elif predator_type == PredatorType.AMBUSHER:
                speed, perception = 0.8, 1.3
                strategy = "hide"
            else:  # SCAVENGER
                speed, perception = 1.0, 1.4
                strategy = "follow"
                
            new_predator = Predator(
                type=predator_type,
                position=position,
                strength=base_strength * self.complexity,
                speed=speed * random.uniform(0.9, 1.1),
                perception=perception * random.uniform(0.9, 1.1),
                stamina=random.uniform(0.6, 1.0),
                hunt_strategy=strategy
            )
            
            self.current_state.predators.append(new_predator)
            logger.info(f"Added {predator_type.name} predator: strength={new_predator.strength:.2f}, "
                       f"speed={new_predator.speed:.2f}")

    def add_predator(self, count: int) -> None:
        """Add predators to the environment"""
        for i in range(count):
            try:
                genetic_core = GeneticCore()
                network_params = self._calculate_predator_network(genetic_core)
                
                neural_net = NeuralAdaptiveNetwork(
                    input_size=network_params['input_size'],
                    hidden_size=network_params['hidden_size'],
                    output_size=network_params['output_size'],
                    memory_size=network_params['memory_size'],
                    learning_rate=genetic_core.brain_genetics.learning_rate,
                    plasticity=genetic_core.brain_genetics.neural_plasticity
                )
                
                position = (
                    random.uniform(0, self.size[0]),
                    random.uniform(0, self.size[1])
                )
                
                predator = AdaptiveAgent(  # Using same base class as agents
                    genetic_core=genetic_core,
                    neural_net=neural_net,
                    position=position,
                    is_predator=True  # New flag to distinguish predators
                )
                
                # Initialize predator-specific traits
                predator.hunting_efficiency = genetic_core.physical_genetics.energy_efficiency
                predator.detection_range = genetic_core.physical_genetics.sensor_sensitivity * 1.5
                predator.attack_strength = genetic_core.physical_genetics.action_precision
                
                self.current_state.predators.append(predator)
                logger.info(f"Created predator {i+1} with network: {network_params}")
                
            except Exception as e:
                logger.error(f"Failed to create predator {i+1}: {str(e)}")
    
    def _calculate_predator_network(self, genetic_core: GeneticCore) -> Dict[str, int]:
        """Calculate neural network architecture for predators"""
        # Enhanced sensory inputs for predators
        sensor_inputs = int(24 * (1 + genetic_core.physical_genetics.sensor_sensitivity))
        memory_inputs = int(12 * genetic_core.brain_genetics.memory_capacity)
        
        # Larger network for complex hunting behaviors
        return {
            'input_size': sensor_inputs + memory_inputs,
            'hidden_size': int(48 * (1 + genetic_core.brain_genetics.neural_plasticity)),
            'output_size': int(16 * (1 + genetic_core.mind_genetics.creativity)),
            'memory_size': memory_inputs
        }
