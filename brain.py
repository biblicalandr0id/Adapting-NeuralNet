from typing import Dict, Optional
import torch
import logging
from dataclasses import dataclass, field
from datetime import datetime
from neural_networks import NeuralAdaptiveNetwork

logger = logging.getLogger(__name__)

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
    """Handles cognitive processing and memory consolidation"""
    def __init__(self, neural_net: NeuralAdaptiveNetwork, genetic_core, plasticity: float, learning_rate: float):
        self.neural_net = neural_net
        self.genetic_core = genetic_core
        self.plasticity = plasticity
        self.learning_rate = learning_rate
        self.state = BrainState()
        
        # Initialize reasoning system based on genetics
        self.reasoning_type = self._determine_reasoning_type()
        
        # Set up cognitive parameters from genetics
        self.cognitive_params = {
            'processing_speed': genetic_core.brain_genetics.processing_speed,
            'pattern_recognition': genetic_core.brain_genetics.pattern_recognition,
            'attention_capacity': genetic_core.brain_genetics.multi_tasking,
            'neural_plasticity': genetic_core.brain_genetics.neural_plasticity,
            'memory_capacity': genetic_core.brain_genetics.memory_capacity,
            'decision_speed': genetic_core.brain_genetics.decision_speed,
            'spatial_awareness': genetic_core.brain_genetics.spatial_awareness,
            'temporal_processing': genetic_core.brain_genetics.temporal_processing,
            'error_correction': genetic_core.brain_genetics.error_correction
        }
        
    def _determine_reasoning_type(self) -> str:
        """Determine reasoning type based on genetic traits"""
        if self.genetic_core.mind_genetics.problem_solving > 1.5:
            return 'gnn'
        elif self.genetic_core.mind_genetics.creativity > 1.5:
            return 'transformer'
        elif self.genetic_core.brain_genetics.pattern_recognition > 1.0:
            return 'attention'
        return 'linear'
        
    def process_input(self, input_data: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process input through neural network with genetic influence"""
        # Apply processing speed to input
        input_data = input_data * self.cognitive_params['processing_speed']
        
        # Get reasoning context if using advanced reasoning
        if self.reasoning_type != 'linear':
            if context is None:
                context = self.state.get_context()
                
        # Process through neural network
        output, importance = self.neural_net(input_data, context)
        
        # Update brain state
        self.state.update(
            attention_focus=importance.mean().item(),
            cognitive_load=output.abs().mean().item(),
            processing_depth=self.cognitive_params['pattern_recognition']
        )
        
        return output
        
    def learn(self, experience: Dict):
        """Learn from experience with plasticity influence"""
        # Extract experience data
        state = experience['state']
        action = experience['action']
        reward = experience.get('reward', 0.0)
        
        # Scale learning parameters by plasticity
        effective_learning_rate = self.learning_rate * (1.0 + self.plasticity)
        
        # Process through neural network with plasticity influence
        self.neural_net.backward(
            state, 
            action,
            learning_rate=effective_learning_rate,
            plasticity=self.plasticity
        )
        
        # Update plasticity based on reward
        if abs(reward) > 0.5:
            self.plasticity = min(2.0, self.plasticity * (1.0 + abs(reward) * 0.1))
        
    def consolidate_memory(self):
        """Consolidate memories during rest with genetic influence"""
        if not hasattr(self.neural_net, 'memory_cells'):
            return
            
        memory_retention = self.genetic_core.mind_genetics.memory_retention
        pattern_recognition = self.cognitive_params['pattern_recognition']
        
        # Process each memory cell
        for cell in self.neural_net.memory_cells:
            # Scale retention by pattern recognition
            retention_threshold = 1.0 - (memory_retention * pattern_recognition)
            cell.retention_threshold = retention_threshold
            
            # Consolidate important memories
            cell.consolidate_memories()