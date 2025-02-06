from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import random
import uuid

from genetics import GeneticCore
from neural_networks import NeuralAdaptiveNetwork

@dataclass
class Predator:
    """Base class for predator entities"""
    position: Tuple[float, float]
    strength: float = 1.0
    speed: float = 1.0
    perception: float = 1.0
    stamina: float = 1.0
    hunt_strategy: str = "patrol"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    energy: float = 100.0
    
    def update(self, env_state) -> bool:
        """Update predator state"""
        self.energy -= 0.1  # Base energy consumption
        return self.energy > 0

class AdaptivePredator(Predator):
    def __init__(self, 
                 genetic_core: GeneticCore, 
                 neural_net: NeuralAdaptiveNetwork,
                 position: Tuple[float, float],
                 parent: Optional['AdaptivePredator'] = None):
        
        super().__init__(position=position)
        self.genetic_core = genetic_core
        if parent:
            self.genetic_core.inherit_from(parent.genetic_core)
        self.neural_net = neural_net
        self.age = 0
        self.max_age = int(100 * genetic_core.physical_genetics.longevity)
        
        # Predator-specific attributes
        self.hunting_strategy = self._initialize_hunting_strategy()
        self.target = None
        self.last_kill_time = 0
        self.successful_hunts = 0
        
        # Evolution tracking
        self.hunt_stats = {
            'successful_hunts': 0,
            'failed_attempts': 0,
            'energy_efficiency': 0.0,
            'average_pursuit_time': 0.0
        }
        
    def _initialize_hunting_strategy(self) -> Dict:
        """Initialize hunting strategy based on genetics"""
        return {
            'aggression': self.genetic_core.mind_genetics.creativity,
            'stealth': self.genetic_core.physical_genetics.energy_efficiency,
            'pack_tendency': self.genetic_core.mind_genetics.adaptation_rate,
            'territory_size': self.genetic_core.physical_genetics.sensor_sensitivity
        }
        
    def decide_action(self, env_state) -> Tuple[str, Dict]:
        """Use neural network to decide hunting action"""
        # Create input vector from environmental state and internal state
        inputs = self._process_sensory_input(env_state)
        
        # Get neural network output
        outputs = self.neural_net.forward(inputs)
        
        # Convert outputs to action decision
        action, params = self._decode_neural_output(outputs)
        
        return action, params
        
    def _process_sensory_input(self, env_state) -> np.ndarray:
        """Process environmental state into neural network inputs"""
        # Include target position, energy levels, nearby agents, etc.
        inputs = []
        # ...implement sensory processing...
        return np.array(inputs)
        
    def _decode_neural_output(self, outputs: np.ndarray) -> Tuple[str, Dict]:
        """Convert neural network outputs into concrete actions"""
        # Map neural outputs to hunting actions (stalk, chase, attack, etc.)
        actions = {
            'stalk': outputs[0],
            'chase': outputs[1],
            'attack': outputs[2],
            'retreat': outputs[3]
        }
        
        # Choose action with highest activation
        action = max(actions.items(), key=lambda x: x[1])[0]
        
        # Generate parameters for the chosen action
        params = self._generate_action_params(action, outputs)
        
        return action, params