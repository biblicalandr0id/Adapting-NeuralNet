import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
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
# ... other local imports as needed

# Module logger
logger = logging.getLogger(__name__)

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import AdaptiveAgent
    from predator import AdaptivePredator

from environment_protocol import EnvironmentProtocol
from predator import PredatorType, Predator, add_predator #Importing

class ResourceType(Enum):  # From agent-architecture.py
    ENERGY = "energy"
    INFORMATION = "information"
    MATERIALS = "materials"


@dataclass  # From agent-architecture.py
class Resource:
    type: ResourceType  # Changed from str to ResourceType enum
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
    predators: List['AdaptivePredator'] = field(default_factory=list)
    time_step: int = 0
    complexity_level: float = 0.5
    agents: List['AdaptiveAgent'] = field(default_factory=list)
    size: Tuple[int, int] = (0, 0)
    weather_conditions: Dict = field(default_factory=dict)

    def get_context_vector(self) -> torch.Tensor:
        """Get environmental context as tensor for neural processing"""
        context = [
            self.complexity_level,
            len(self.resources),
            len(self.predators),
            len(self.agents),
            self.time_step / 1000.0,  # Normalize time
            self.weather_conditions.get('temperature', 0.5),
            self.weather_conditions.get('humidity', 0.5)
        ]
        return torch.tensor(context, dtype=torch.float32)


class AdaptiveEnvironment(EnvironmentProtocol): #Implementing the protocol
    def __init__(self, size: Tuple[int, int], complexity: float):
        self.size = size
        self.complexity = complexity
        self.config = {
            'max_resources': 100,
            'resource_spawn_rate': 0.01 * complexity,
            'predator_spawn_rate': 0.005 * complexity,
            'resource_limit': int(50 * complexity),
            'predator_limit': int(10 * complexity)
        }
        self.current_state = EnvironmentalState(
            resources=[],
            predators=[],  # Now stores AdaptivePredator instances
            time_step=0,
            complexity_level=complexity,
            size=size,
            weather_conditions=self._initialize_weather()  # Initialize weather
        )

    def step(self, agents: List['AdaptiveAgent']) -> EnvironmentalState:
        """Advance the environment by one time step"""
        try:
            self.current_state.time_step += 1
            self.current_state.agents = agents

            # Update resources with bounds checking
            for resource in self.current_state.resources[:]:  # Copy list to allow modifications
                if resource.quantity <= 0:
                    self.current_state.resources.remove(resource)
                    continue

                # Calculate environmental factors
                terrain_factor = max(0.1, min(2.0, self._get_terrain_factor(resource.position)))
                weather_factor = max(0.1, min(2.0, self._get_weather_factor(resource.position)))
                
                # Update resource quantity with regeneration
                if resource.quantity < resource.max_quantity:
                    regen_rate = resource.regeneration_rate
                    if resource.quantity < resource.depletion_threshold:
                        regen_rate *= 1.5  # Faster regeneration when depleted
                    
                    resource.quantity = min(
                        resource.max_quantity,
                        resource.quantity + (regen_rate * terrain_factor * weather_factor)
                    )

            # Spawn new resources based on rate and current count
            if (len(self.current_state.resources) < self.config['max_resources'] and 
                random.random() < self.config['resource_spawn_rate']):
                self.add_resources(1)

            # Update predators with error handling
            for predator in self.current_state.predators[:]:  # Copy list to allow modifications
                try:
                    movement = self._calculate_threat_movement(predator.position)
                    terrain_gradient = self._calculate_terrain_gradient(predator.position)
                    
                    # Combine movement vectors with terrain influence
                    final_movement = (
                        movement[0] + terrain_gradient[0] * 0.3,
                        movement[1] + terrain_gradient[1] * 0.3
                    )
                    
                    # Update position with bounds checking
                    predator.position = (
                        max(0, min(self.size[0], predator.position[0] + final_movement[0])),
                        max(0, min(self.size[1], predator.position[1] + final_movement[1]))
                    )
                    
                except Exception as e:
                    logger.error(f"Error updating predator: {str(e)}")
                    continue

            # Update weather conditions
            self.current_state.weather_conditions = self._update_weather(
                self.current_state.weather_conditions
            )

            return self.current_state

        except Exception as e:
            logger.error(f"Error in environment step: {str(e)}")
            return self.current_state

    def _update_state(self):
        """Update environment state - resource dynamics, threats, etc."""
        self.current_state.time_step += 1

        # Example: Resource regeneration (simplified)
        for resource in self.current_state.resources:
            if resource.quantity < 100:
                terrain_factor = self._get_terrain_factor(tuple(map(int, resource.position)))
                weather_factor = self._get_weather_factor(tuple(map(int, resource.position)))
                resource.quantity += random.uniform(0, 0.5) * terrain_factor * weather_factor

        # Example: Threat movement (simplified random walk)
        for i in range(len(self.current_state.predators)):  # Changed from threats
            predator = self.current_state.predators[i]  # Changed from threats
            movement = self._calculate_threat_movement(predator.position)  # Changed from threats
            terrain_gradient = self._calculate_terrain_gradient(tuple(map(int, predator.position)))
            
            # Adjust movement based on terrain gradient
            movement = (movement[0] + terrain_gradient[0], movement[1] + terrain_gradient[1])
            
            new_position = (
                predator.position[0] + movement[0] * predator.speed,
                predator.position[1] + movement[1] * predator.speed
            )
            
            # Keep threats within bounds
            predator.position = (max(0, min(self.size[0]-1, new_position[0])), max(0, min(self.size[1]-1, new_position[1])))
            self.current_state.predators[i] = predator

        # Example: New resource spawning (probability based on complexity)
        if random.random() < 0.01 * self.current_state.complexity_level:
            position = (random.randint(0, self.size[0]-1), random.randint(0, self.size[1]-1))
            terrain_factor = self._get_terrain_factor(position)
            weather_factor = self._get_weather_factor(position)
            
            self.current_state.resources.append(
                Resource(
                    type=random.choice(list(ResourceType)),
                    quantity=random.uniform(10, 50) * terrain_factor * weather_factor,
                    position=position,
                    complexity=random.uniform(0.1, 0.9),
                    regeneration_rate = 0.1,
                    max_quantity = 100,
                    depletion_threshold = 10,
                    quality = 0.5,
                    seasonal_factor = 0.5
                )
            )

    def _generate_perlin_noise(self, size: Tuple[int, int]) -> np.ndarray:
        return np.zeros(size)

    def _calculate_threat_movement(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate movement vector for a threat based on its current position"""
        # Get nearest agent
        nearest_agent = self._find_nearest_agent(position)
        
        if nearest_agent:
            # Move towards agent if found
            dx = nearest_agent.position[0] - position[0]
            dy = nearest_agent.position[1] - position[1]
            # Normalize vector
            magnitude = (dx**2 + dy**2)**0.5
            if magnitude > 0:
                return (dx/magnitude, dy/magnitude)
                
        # Random movement if no agent found
        return (random.uniform(-1, 1), random.uniform(-1, 1))

    def _initialize_weather(self) -> Dict:
        """Initialize weather conditions"""
        return {
            "temperature": random.uniform(15, 30),
            "humidity": random.uniform(0.3, 0.8),
            "wind_speed": random.uniform(0, 20),
            "precipitation": random.uniform(0, 1)
        }

    def _update_weather(self, current_weather: Dict) -> Dict:
        """Update weather conditions with slight random changes"""
        new_weather = current_weather.copy()
        new_weather["temperature"] += random.uniform(-2, 2)
        new_weather["humidity"] = max(0, min(1, new_weather["humidity"] + random.uniform(-0.1, 0.1)))
        new_weather["wind_speed"] = max(0, new_weather["wind_speed"] + random.uniform(-5, 5))
        new_weather["precipitation"] = max(0, min(1, new_weather["precipitation"] + random.uniform(-0.2, 0.2)))
        return new_weather

    def _get_terrain_factor(self, position: Tuple[int, int]) -> float:
        """Calculate terrain difficulty factor based on position"""
        x, y = position
        # Use perlin noise or similar for terrain generation
        terrain_height = np.sin(x/10) * np.cos(y/10)  # Simple terrain function
        return max(0.5, min(1.5, 1 + terrain_height))

    def _get_weather_factor(self, position: Tuple[int, int]) -> float:
        """Calculate weather impact factor based on position"""
        if not hasattr(self.current_state, 'weather_conditions'):
            return 1.0
        # Consider local weather conditions
        weather = self.current_state.weather_conditions
        return 1.0 - (weather.get("precipitation", 0) * 0.3)

    def _calculate_terrain_gradient(self, position: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate terrain gradient for movement effects"""
        x, y = position
        dx = np.cos(x/10) * 0.1
        dy = np.cos(y/10) * 0.1
        return (dx, dy)

    def _find_nearest_agent(self, pos: Tuple[float, float]) -> Optional['AdaptiveAgent']:
        """Find the nearest agent to a given position"""
        if not self.current_state.agents:
            return None
            
        nearest = None
        min_dist = float('inf')
        
        for agent in self.current_state.agents:
            dist = ((pos[0] - agent.position[0])**2 + (pos[1] - agent.position[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest = agent
            
        return nearest

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
        add_predator(self, count)

    def get_state(self) -> EnvironmentalState:
        return self.current_state

    def get_resources(self) -> List[Resource]:
        return self.current_state.resources

    def get_agents(self) -> List['AdaptiveAgent']:
        return self.current_state.agents

    def get_size(self) -> Tuple[int, int]:
        return self.size
