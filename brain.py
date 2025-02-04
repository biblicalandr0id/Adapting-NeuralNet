from typing import Dict, List, Tuple, Optional
import torch
import logging
from dataclasses import dataclass, field
from datetime import datetime
from mind import MindState
from neural_networks import NeuralAdaptiveNetwork

@dataclass
class BrainState:
    """Represents the current state of the brain's cognitive functions"""
    attention_focus: str = "idle"
    cognitive_load: float = 0.0
    processing_depth: float = 1.0
    memory_utilization: float = 0.0
    current_task: Optional[str] = None
    last_update: datetime = field(default_factory=datetime.now)
    neural_activity: float = 0.0
    energy_consumption: float = 0.0

class Brain:
    def __init__(self, genetic_core, neural_net: NeuralAdaptiveNetwork):
        self.genetic_core = genetic_core
        self.neural_net = neural_net
        self.state = BrainState()
        self.mind_state = MindState()
        self.short_term_memory = []
        self.working_memory = {}
        self.processing_queue = []

    def _calculate_cognitive_load(self) -> float:
        """Calculate current cognitive load based on processing"""
        base_load = len(self.processing_queue) / 10.0  # 10 items = full load
        genetic_modifier = self.genetic_core.brain_genetics.cognitive_efficiency
        return min(1.0, base_load / genetic_modifier)

    def _calculate_processing_depth(self) -> float:
        """Calculate processing depth based on brain genetics"""
        base_depth = self.genetic_core.brain_genetics.pattern_recognition
        focus_modifier = 1.0 if self.state.attention_focus != "idle" else 0.5
        return base_depth * focus_modifier

    def _consolidate_memory(self, experience: Dict) -> None:
        """Consolidate experience into memory"""
        memory_strength = self.genetic_core.brain_genetics.memory_capacity
        if len(self.short_term_memory) >= int(10 * memory_strength):
            self.short_term_memory.pop(0)
        self.short_term_memory.append(experience)

    def _preprocess_input(self, sensory_input: Dict, sensitivity: float) -> torch.Tensor:
        """Preprocess sensory input based on genetic sensitivity"""
        if isinstance(sensory_input, dict):
            processed = torch.tensor([float(v) * sensitivity for v in sensory_input.values()])
        else:
            processed = torch.tensor(sensory_input) * sensitivity
        return processed

    def process_input(self, sensory_input: Dict) -> Tuple[torch.Tensor, BrainState]:
        """Process sensory input through neural network and update brain state"""
        # Adjust input based on genetic traits
        sensitivity = self.genetic_core.brain_genetics.pattern_recognition
        processed_input = self._preprocess_input(sensory_input, sensitivity)
        
        # Neural processing
        with torch.no_grad():
            output = self.neural_net(processed_input)
        
        # Update brain state
        self.state.cognitive_load = self._calculate_cognitive_load()
        self.state.processing_depth = self._calculate_processing_depth()
        
        return output, self.state
    
    def dream_cycle(self, experiences: List[Dict]) -> None:
        """Process experiences during rest/dream state"""
        plasticity = self.genetic_core.brain_genetics.neural_plasticity
        
        for experience in experiences:
            self.neural_net.process_dream(
                experience,
                learning_rate=plasticity * 0.1
            )
            self._consolidate_memory(experience)