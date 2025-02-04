# mind.py
import random
from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
import time
import torch


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


def create_embryo() -> AgentEmbryo:
    """Factory function to create a new agent embryo"""
    return AgentEmbryo()
