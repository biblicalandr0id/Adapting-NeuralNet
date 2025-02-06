# mind.py
import random
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import time
import torch
import logging
from neural_networks import NeuralAdaptiveNetwork
from genetics import GeneticCore


@dataclass
class GrowthMetrics:
    """Tracks developmental progress of the agent"""
    cognitive_complexity: float = 0.0
    adaptation_rate: float = 0.0
    learning_capacity: float = 0.0
    growth_stage: str = "embryonic"
    age: float = 0.0


@dataclass
class Memory:
    """Stores experiences and learned patterns"""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)
    developmental_milestones: List[str] = field(default_factory=list)


@dataclass
class MindMetrics:
    age: float = 0.0
    growth_stage: str = "embryonic"
    cognitive_complexity: float = 0.1
    adaptation_rate: float = 0.1
    learning_capacity: float = 0.1


@dataclass
class MindState:
    """Current state of mind processing"""
    consciousness_level: float = 1.0
    awareness_focus: str = "environment"
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        'curiosity': 0.5,
        'confidence': 0.5,
        'stress': 0.0
    })
    current_task: Optional[str] = None
    energy_consumption: float = 0.0


class AgentEmbryo:
    def __init__(self):
        self.metrics = GrowthMetrics()
        self.memory = Memory()
        self.growth_rate = 0.01
        self.adaptation_threshold = 0.5
        self.maturity_threshold = 10.0
        self.last_update = time.time()

        # Initialize basic neural patterns
        self.neural_patterns = torch.rand(10, 10)

    def process_stimulus(self, stimulus: Dict[str, Any]) -> Tuple[Dict[str, Any], 'AgentEmbryo']:
        """Process environmental input and adapt neural patterns"""
        # Store in short-term memory
        self.memory.short_term.append(stimulus)
        if len(self.memory.short_term) > 100:
            self._consolidate_memory()

        # Adapt neural patterns based on stimulus
        try:
            stimulus_vector = torch.tensor(list(stimulus.values())).float()
        except Exception as e:
            print(
                f"Error processing stimulus: {e}, please ensure all stimulus values are numbers: {stimulus}")
            stimulus_vector = torch.zeros(len(stimulus))

        self._adapt_neural_patterns(stimulus_vector)

        return self._generate_response(stimulus), self

    def grow(self) -> None:
        """Handle natural growth and development"""
        current_time = time.time()
        time_delta = current_time - self.last_update

        # Update age and metrics
        self.metrics.age += time_delta
        self.metrics.cognitive_complexity += self.growth_rate * time_delta
        self.metrics.adaptation_rate = min(
            1.0, self.metrics.adaptation_rate + 0.001 * time_delta)
        self.metrics.learning_capacity = min(
            1.0, self.metrics.learning_capacity + 0.002 * time_delta)

        # Check for developmental milestones
        self._check_developmental_stage()
        self.last_update = current_time

    def _adapt_neural_patterns(self, stimulus_vector: torch.Tensor) -> None:
        """Adapt neural patterns based on environmental input"""
        # Simple Hebbian-inspired learning
        pattern_activation = self.neural_patterns @ stimulus_vector
        self.neural_patterns += self.metrics.adaptation_rate * \
            torch.outer(pattern_activation, stimulus_vector)
        self.neural_patterns = torch.clip(self.neural_patterns, 0, 1)

    def _consolidate_memory(self) -> None:
        """Move important patterns from short-term to long-term memory"""
        if len(self.memory.short_term) < 10:
            return

        # Simple pattern recognition and consolidation
        patterns = {}
        for memory in self.memory.short_term:
            key = str(sorted(memory.items()))
            patterns[key] = patterns.get(key, 0) + 1

        # Store frequently occurring patterns
        for pattern, count in patterns.items():
            if count >= 3:  # threshold for importance
                self.memory.long_term[pattern] = self.memory.long_term.get(
                    pattern, 0) + 1

        self.memory.short_term = []

    def _check_developmental_stage(self) -> None:
        """Update developmental stage based on metrics"""
        if self.metrics.cognitive_complexity >= self.maturity_threshold:
            if self.metrics.growth_stage == "embryonic":
                self.metrics.growth_stage = "juvenile"
                self.memory.developmental_milestones.append(
                    f"Reached juvenile stage at age {self.metrics.age:.2f}")
                self.growth_rate *= 1.5  # Accelerated growth in juvenile stage

        if self.metrics.cognitive_complexity >= self.maturity_threshold * 2:
            if self.metrics.growth_stage == "juvenile":
                self.metrics.growth_stage = "adolescent"
                self.memory.developmental_milestones.append(
                    f"Reached adolescent stage at age {self.metrics.age:.2f}")
                # Increase adaptation capabilities
                self.adaptation_threshold *= 0.8

    def _generate_response(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response based on current development level and stimulus"""
        # Basic response generation based on development level
        response_complexity = min(
            1.0, self.metrics.cognitive_complexity / self.maturity_threshold)
        response_vector = self.neural_patterns @ torch.rand(10)

        return {
            "response_type": self.metrics.growth_stage,
            "complexity": response_complexity,
            "pattern_activation": response_vector.tolist(),
            "developmental_state": {
                "age": self.metrics.age,
                "stage": self.metrics.growth_stage,
                "cognitive_complexity": self.metrics.cognitive_complexity
            }
        }


class EmbronicMind:
    def __init__(self):
        self.metrics = MindMetrics()
        self.state = MindState(metrics=self.metrics)
        
    def process_stimulus(self, stimulus: Dict[str, Any]) -> Tuple[Dict[str, Any], MindState]:
        """Process incoming stimulus and return response with updated state"""
        try:
            # Update metrics based on stimulus
            self.metrics.age += 0.1
            self.metrics.cognitive_complexity += 0.01
            
            # Generate response
            response = {
                "complexity": self.metrics.cognitive_complexity,
                "adaptation": self.metrics.adaptation_rate,
                "learning": self.metrics.learning_capacity,
                "focus": stimulus.get("type", "unknown")
            }
            
            return response, self.state
            
        except Exception as e:
            logging.error(f"Error processing stimulus: {e}")
            return {}, self.state


class Mind:
    def __init__(self, genetic_core: GeneticCore, neural_net: NeuralAdaptiveNetwork):
        self.genetic_core = genetic_core
        self.neural_net = neural_net
        self.state = MindState()
        self.memory = []
        self.processing_queue = []
        
    def process_state(self, state_input: Dict) -> Dict:
        """Process current state through mind systems"""
        # Convert input to tensor
        state_tensor = torch.tensor([float(v) for v in state_input.values()])
        
        # Process through neural network
        output, importance = self.neural_net(state_tensor)
        
        # Update mind state
        self.state.energy_consumption = importance.mean().item()
        self.state.consciousness_level = max(0.1, min(1.0, 
            self.state.consciousness_level - self.state.energy_consumption * 0.1
        ))
        
        return {
            'output': output.detach().numpy(),
            'consciousness': self.state.consciousness_level,
            'energy': self.state.energy_consumption
        }


def create_embryo() -> AgentEmbryo:
    """Factory function to create a new agent embryo"""
    return AgentEmbryo()


def create_embryo() -> EmbronicMind:
    """Factory function to create embryonic mind"""
    return EmbronicMind()
