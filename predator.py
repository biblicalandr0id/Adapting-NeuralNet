import uuid
from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Dict, Tuple

from genetics import GeneticCore
from neural_networks import NeuralAdaptiveNetwork
import logging

logger = logging.getLogger(__name__)

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

def add_predator(environment, count: int) -> None:
    """Add predators to the environment"""
    for i in range(count):
        try:
            genetic_core = GeneticCore()
            network_params = _calculate_predator_network(genetic_core)
            
            neural_net = NeuralAdaptiveNetwork(
                input_size=network_params['input_size'],
                output_size=network_params['output_size'],
                genetic_core = genetic_core
            )
            
            position = (
                random.uniform(0, environment.size[0]),
                random.uniform(0, environment.size[1])
            )
            
            from agent import AdaptiveAgent  # Import here to avoid circular dependency
            predator = AdaptiveAgent(  # Using same base class as agents
                environment = environment,
                genetic_core=genetic_core,
                neural_net=neural_net,
                position=position,
                is_predator=True  # New flag to distinguish predators
            )
            
            # Initialize predator-specific traits
            predator.hunting_efficiency = genetic_core.physical_genetics.energy_efficiency
            predator.detection_range = genetic_core.physical_genetics.sensor_sensitivity * 1.5
            predator.attack_strength = genetic_core.physical_genetics.action_precision
            
            environment.current_state.predators.append(predator)
            logger.info(f"Created predator {i+1} with network: {network_params}")
            
        except Exception as e:
            logger.error(f"Failed to create predator {i+1}: {str(e)}")

def _calculate_predator_network(genetic_core: GeneticCore) -> Dict[str, int]:
    """Calculate neural network architecture for predators"""
    # Enhanced sensory inputs for predators
    sensor_inputs = int(24 * (1 + genetic_core.physical_genetics.sensor_sensitivity))
    memory_inputs = int(12 * genetic_core.brain_genetics.memory_capacity)
    
    # Larger network for complex hunting behaviors
    return {
        'input_size': sensor_inputs + memory_inputs,
        'output_size': int(16 * (1 + genetic_core.mind_genetics.creativity)),
    }