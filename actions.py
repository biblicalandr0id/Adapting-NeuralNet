from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Callable, Any, Union
import torch
import torch.nn.functional as F
import numpy as np
from genetics import GeneticCore
from adaptive_environment import EnvironmentalState, Resource
import logging
import time
import uuid

logger = logging.getLogger(__name__)

@dataclass
class ActionResult:
    """Represents the result of an action"""
    success: bool
    reward: float
    energy_cost: float
    new_state: Optional[Dict[str, Any]]

class ActionPrototype:
    """Represents a prototype action with genetic influence"""
    def __init__(self, name: str, vector: torch.Tensor, base_energy_cost: float):
        self.name = name
        self.vector = vector.float()  # Ensure float tensor
        self.base_energy_cost = float(base_energy_cost)
        self.usage_count = 0
        self.success_rate = 0.0
        self.average_reward = 0.0

class ActionDecoder:
    """Decodes action selection and parameters from neural output"""
    def __init__(self, hidden_size: int = 32):
        self.hidden_size = hidden_size
        self.prototypes: Dict[str, ActionPrototype] = {}
        self.action_methods: Dict[str, Callable[[Dict, bool, EnvironmentalState], float]] = {}

    def add_action(self, name: str, prototype: ActionPrototype, method: Callable):
        """Add new action to the decoder"""
        self.prototypes[name] = prototype
        self.action_methods[name] = method

    def decode_selection(self, selection_vector: torch.Tensor) -> Tuple[str, float]:
        """Decode action selection from selection vector"""
        best_similarity = -1
        selected_action = None

        for name, prototype in self.prototypes.items():
            similarity = F.cosine_similarity(
                selection_vector.unsqueeze(0),
                prototype.vector.unsqueeze(0)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                selected_action = name

        return selected_action, best_similarity.item()

class ActionSystem:
    """Handles action selection, execution and mutation"""
    def __init__(self, genetic_core: GeneticCore):
        if not isinstance(genetic_core, GeneticCore):
            raise ValueError("genetic_core must be an instance of GeneticCore")
            
        self.genetic_core = genetic_core
        self.action_decoder = ActionDecoder(hidden_size=32)
        self.actions: Dict[str, Callable] = {}
        self.action_history: List[Dict] = []
        self.mutation_rate = max(0.0, min(1.0, genetic_core.mind_genetics.creativity * 0.1))
        self.unknown_action_count = 0

    def validate_action_params(self, action_name: str, params: Dict) -> bool:
        """Validate action parameters before execution"""
        if not action_name or not isinstance(action_name, str):
            logger.error("Invalid action name")
            return False
        if not isinstance(params, dict):
            logger.error("Parameters must be a dictionary")
            return False
        if action_name not in self.actions:
            logger.error(f"Unknown action: {action_name}")
            return False
        return True
        
    def add_action(self, name: str, action_func: Callable, base_energy_cost: float = 1.0):
        """Add new action with genetic influence on prototype"""
        self.actions[name] = action_func
        
        # Create genetically influenced prototype vector
        creativity = self.genetic_core.mind_genetics.creativity
        intelligence = self.genetic_core.brain_genetics.processing_speed
        prototype_vector = torch.randn(32) * creativity * intelligence
        
        # Create action prototype
        prototype = ActionPrototype(
            name=name,
            vector=prototype_vector,
            base_energy_cost=base_energy_cost
        )
        
        self.action_decoder.add_action(name, prototype, action_func)

    def create_unknown_action(self, stimulus_vector: torch.Tensor) -> str:
        """Create a new unknown action based on environmental stimulus"""
        # Get genetic influences for unknown action creation
        creativity = self.genetic_core.mind_genetics.creativity
        adaptability = self.genetic_core.mind_genetics.adaptation_rate
        intelligence = self.genetic_core.brain_genetics.processing_speed
        
        # Generate action characteristics based on genetics and stimulus
        action_complexity = torch.norm(stimulus_vector).item() * creativity
        learning_factor = adaptability * intelligence
        base_energy_cost = action_complexity * (1.0 / learning_factor)
        
        # Create unknown action function with dynamic behavior
        def unknown_action(params: Dict, success: bool, env_state: EnvironmentalState) -> float:
            # Dynamic response based on environmental state and genetic traits
            env_context = env_state.get_context_vector()
            environmental_match = F.cosine_similarity(
                stimulus_vector.unsqueeze(0),
                env_context.unsqueeze(0)
            ).item()
            
            # Calculate reward based on environmental match and genetic factors
            base_reward = environmental_match * learning_factor
            uncertainty_factor = np.random.normal(0, 0.2 * creativity)
            adaptation_bonus = adaptability * 0.1
            
            return float(base_reward * (1.0 + uncertainty_factor + adaptation_bonus))
        
        # Generate unique name for unknown action
        self.unknown_action_count += 1
        unknown_name = f"unknown_action_{self.unknown_action_count}"
        
        # Create prototype vector influenced by stimulus and genetics
        prototype_vector = stimulus_vector * creativity + torch.randn_like(stimulus_vector) * adaptability
        
        # Create and register unknown action prototype
        unknown_prototype = ActionPrototype(
            name=unknown_name,
            vector=prototype_vector,
            base_energy_cost=base_energy_cost
        )
        
        self.actions[unknown_name] = unknown_action
        self.action_decoder.add_action(unknown_name, unknown_prototype, unknown_action)
        
        logger.info(f"Created new unknown action: {unknown_name} with complexity {action_complexity:.2f}")
        return unknown_name
        
    def execute_action(self, action_name: str, params: Dict, 
                      env_state: EnvironmentalState) -> ActionResult:
        """Execute action with genetic traits influence and validation"""
        try:
            if not self.validate_action_params(action_name, params):
                return ActionResult(
                    success=False,
                    reward=-1.0,
                    energy_cost=0.0,
                    new_state={'error': 'invalid_parameters'}
                )

            # Get genetic modifiers with bounds checking
            energy_efficiency = max(0.1, self.genetic_core.physical_genetics.energy_efficiency)
            precision = max(0.0, min(1.0, self.genetic_core.physical_genetics.action_precision))
            adaptation = max(0.0, min(1.0, self.genetic_core.mind_genetics.adaptation_rate))
            
            # Calculate bounded energy cost
            base_cost = self.action_decoder.prototypes[action_name].base_energy_cost
            energy_cost = max(0.1, base_cost * (1.0 / energy_efficiency))
            
            # Calculate bounded success probability
            success_prob = min(1.0, (precision + adaptation) / 2.0)
            success = np.random.random() < success_prob
            
            # Execute action with error handling
            action_func = self.actions[action_name]
            try:
                result = action_func(params, success, env_state)
                result = float(result)  # Ensure numeric result
            except Exception as e:
                logger.error(f"Action execution failed: {str(e)}")
                return ActionResult(
                    success=False,
                    reward=-1.0,
                    energy_cost=energy_cost,
                    new_state={'error': f'execution_failed: {str(e)}'}
                )

            # Update prototype statistics safely
            prototype = self.action_decoder.prototypes[action_name]
            prototype.usage_count += 1
            prototype.success_rate = ((prototype.success_rate * (prototype.usage_count - 1) + int(success)) 
                                    / prototype.usage_count)
            prototype.average_reward = ((prototype.average_reward * (prototype.usage_count - 1) + result) 
                                      / prototype.usage_count)

            # Record action history
            self.action_history.append({
                'action': action_name,
                'success': success,
                'energy_cost': energy_cost,
                'timestamp': time.time()
            })

            return ActionResult(
                success=success,
                reward=result,
                energy_cost=energy_cost,
                new_state={'action_completed': True}
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in execute_action: {str(e)}")
            return ActionResult(
                success=False,
                reward=-1.0,
                energy_cost=0.0,
                new_state={'error': f'unexpected_error: {str(e)}'}
            )
    
    def mutate_action(self, action_name: str) -> Optional[str]:
        """Create mutated version of existing action"""
        if action_name not in self.actions:
            return None
            
        # Get genetic influences
        creativity = self.genetic_core.mind_genetics.creativity
        intelligence = self.genetic_core.brain_genetics.processing_speed
        
        # Original action components
        original_action = self.actions[action_name]
        original_prototype = self.action_decoder.prototypes[action_name]
        
        # Create mutated action function
        def mutated_action(params: Dict, success: bool, env_state: EnvironmentalState) -> float:
            base_result = original_action(params, success, env_state)
            mutation_effect = np.random.normal(0, creativity * 0.2)
            efficiency_bonus = intelligence * 0.1
            return float(base_result * (1.0 + mutation_effect + efficiency_bonus))
            
        # Create mutated prototype
        mutation_strength = creativity * 0.3
        mutated_vector = original_prototype.vector + torch.randn_like(original_prototype.vector) * mutation_strength
        
        # Register mutated action
        mutated_name = f"{action_name}_mutated_{len(self.actions)}"
        mutated_prototype = ActionPrototype(
            name=mutated_name,
            vector=mutated_vector,
            base_energy_cost=original_prototype.base_energy_cost * 1.1
        )
        
        self.actions[mutated_name] = mutated_action
        self.action_decoder.add_action(mutated_name, mutated_prototype, mutated_action)
        
        return mutated_name