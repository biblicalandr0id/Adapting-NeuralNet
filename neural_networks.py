# neural_networks.py
import numpy as np
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Type
import torch
import torch.nn as nn
import torch.optim as optim
from base_types import BaseGeneticCore  # Changed from genetics import
from genetics import GeneticCore  # This creates the cycle
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ImprovedAdaptiveNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        # Dynamic multi-layer architecture
        # Experience buffer for learning
        # Genetic modifier integration
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.experience_buffer = []
        self.max_buffer_size = 1000

    def forward(self, x: torch.Tensor, genetic_modifiers: Dict[str, float]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        activations = [x]

        x = x * genetic_modifiers['sensor_sensitivity']

        for i, layer in enumerate(self.layers[:-1]):
            z = layer(activations[-1])
            z = z * genetic_modifiers['processing_speed']

            a = nn.LeakyReLU(0.01)(z)
            activations.append(a)

        z = self.layers[-1](activations[-1])
        output = torch.softmax(z, dim=-1)
        activations.append(output)

        return output, activations

    def backward(self, x: torch.Tensor, y: torch.Tensor, learning_rate: float, plasticity: float):
        """
        Backward pass with genetic trait influence
        Args:
            x: Input tensor
            y: Target tensor 
            learning_rate: Base learning rate modified by genetics
            plasticity: Neural plasticity from genetics
        """
        # Forward pass to get outputs and store activations
        outputs, hidden_states = self.forward(x, self.state_manager.get_context())
        
        # Calculate loss with genetic influence
        loss = self.criterion(outputs, y) * (1.0 + plasticity)
        
        # Zero existing gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Apply genetic traits to gradient updates
        for param in self.parameters():
            if param.grad is not None:
                # Scale gradients by plasticity and learning rate
                param.grad *= plasticity
                param.grad *= learning_rate
                
                # Apply adaptive noise based on plasticity
                noise_scale = 0.01 * plasticity
                param.grad += torch.randn_like(param.grad) * noise_scale
        
        # Update weights
        self.optimizer.step()
        
        # Update memory cells with new learning
        self.state_manager.update_memory(hidden_states, loss.item())
        
        return loss.item()


class GeneticLayer(nn.Module):
    def __init__(self, in_features, out_features, genetic_traits):
# Layer normalization
        # Compression gate
        # Importance generator
        # Adaptive activation functions
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.activation = self._get_activation(genetic_traits.pattern_recognition)
        self.adaptation_rate = genetic_traits.neural_plasticity
        self.processing_speed = genetic_traits.processing_speed
        
        # Compression gate influenced by pattern recognition
        compression_size = int(out_features * genetic_traits.pattern_recognition)
        self.compression_gate = nn.Sequential(
            nn.Linear(out_features + 1, compression_size),
            nn.LayerNorm(compression_size),
            nn.Sigmoid()
        )
        
        # Importance generator scaled by learning efficiency
        self.importance_generator = nn.Sequential(
            nn.Linear(compression_size, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.Sigmoid()
        )
        
    def _get_activation(self, pattern_recognition):
        if pattern_recognition > 1.5:
            return nn.GELU()
        elif pattern_recognition > 1.0:
            return nn.ReLU()
        else:
            return nn.LeakyReLU(0.1)
    
    def forward(self, x, context):
        # Base transformation
        x = self.layer(x)
        x = x * self.processing_speed
        x = self.layer_norm(x)
        x = self.activation(x)
        
        # Compression and importance based on context
        compression = self.compression_gate(torch.cat([x, context], dim=-1))
        importance = self.importance_generator(compression)
        
        # Apply importance and adaptation
        x = x * importance
        if self.training and self.adaptation_rate > 1.0:
            x = x + torch.randn_like(x) * 0.1 * (self.adaptation_rate - 1.0)
        
        return x, importance


class GeneticMemoryCell(nn.Module):
    def __init__(self, input_size, hidden_size, genetic_traits):
        super().__init__()
        self.memory_capacity = genetic_traits.memory_capacity
        self.learning_efficiency = genetic_traits.learning_efficiency
        
        # Scale LSTM size based on memory capacity
        self.lstm_size = int(hidden_size * self.memory_capacity)
        self.lstm = nn.LSTMCell(input_size, self.lstm_size)
        
        # Adaptive gates influenced by learning efficiency
        self.memory_gate = nn.Sequential(
            nn.Linear(self.lstm_size, self.lstm_size),
            nn.Sigmoid()
        )
        
    def forward(self, x, state=None):
        if state is None:
            h_state = torch.zeros(x.shape[0], self.lstm_size, device=x.device)
            c_state = torch.zeros(x.shape[0], self.lstm_size, device=x.device)
        else:
            h_state, c_state = state
            
        # LSTM processing scaled by learning efficiency
        h_state, c_state = self.lstm(x, (h_state, c_state))
        
        # Apply memory gating
        memory_importance = self.memory_gate(h_state)
        h_state = h_state * memory_importance * self.learning_efficiency
        
        return h_state, c_state


class NeuralAdaptiveNetwork(nn.Module):
    def __init__(self, input_size, output_size, genetic_core):
        super().__init__()
        self.mind_genetics = genetic_core.mind_genetics
        self.brain_genetics = genetic_core.brain_genetics
        self.output_size = output_size

        # Base network parameters from genetics
        hidden_size = max(64, int(64 * self.mind_genetics.memory_capacity))
        num_layers = max(2, int(2 * self.mind_genetics.pattern_recognition))
        
        # Create state manager for memory handling
        self.state_manager = AdaptiveStateManager(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            adaptive_rate=self.brain_genetics.learning_rate,
            memory_type='lstm'
        )
        
        # Create genetic layers with state management
        self.layers = nn.ModuleList()
        current_size = input_size
        
        for i in range(num_layers):
            layer_size = int(hidden_size * (1 + 0.2 * (i - num_layers/2)))
            genetic_layer = GeneticLayer(current_size, layer_size, self.brain_genetics)
            self.layers.append(genetic_layer)
            current_size = layer_size
        
        # Memory cells from GeneticMemoryCell
        num_memory_cells = max(1, int(2 * self.mind_genetics.cognitive_growth_rate))
        self.memory_cells = nn.ModuleList([
            GeneticMemoryCell(current_size, hidden_size, self.mind_genetics)
            for _ in range(num_memory_cells)
        ])
        
        self.output = nn.Linear(hidden_size, output_size)
        self.reset_states()
        
        # Add optimizer initialization with genetic traits influence
        learning_rate = self.brain_genetics.learning_rate
        beta1 = max(0.5, min(0.99, self.mind_genetics.pattern_recognition))
        beta2 = max(0.8, min(0.999, self.mind_genetics.learning_efficiency))
        
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=1e-8 * self.brain_genetics.neural_plasticity
        )
        
        # Loss criterion with genetic scaling
        self.criterion = nn.MSELoss()
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device
        
        if context is None:
            context = torch.zeros(batch_size, 1, device=device)
            
        # Process through genetic layers
        importance_signals = []
        for layer in self.layers:
            x, importance = layer(x, context)
            importance_signals.append(importance)
            
        # Process through memory cells
        for i, cell in enumerate(self.memory_cells):
            if i >= len(self.h_states):  # Handle dynamic memory cell count
                self.h_states.append(torch.zeros(batch_size, cell.lstm_size, device=device))
                self.c_states.append(torch.zeros(batch_size, cell.lstm_size, device=device))
            elif self.h_states[i].shape[0] != batch_size:  # Handle batch size changes
                self.h_states[i] = torch.zeros(batch_size, cell.lstm_size, device=device)
                self.c_states[i] = torch.zeros(batch_size, cell.lstm_size, device=device)
                
            self.h_states[i], self.c_states[i] = cell(x, (self.h_states[i], self.c_states[i]))
            x = self.h_states[i]
            
        # Final output with proper shape for action selection
        output = self.output(x)
        if output.shape[-1] != self.output_size:
            raise ValueError(f"Output shape {output.shape} doesn't match required size {self.output_size}")
        
        # Return output and mean importance (maintaining original interface)
        return output, torch.stack(importance_signals).mean(0)
        
    def reset_states(self):
        """Reset memory cell states"""
        self.h_states = []
        self.c_states = []
        if hasattr(self, 'memory_cells'):
            device = next(self.parameters()).device
            batch_size = 1  # Default batch size, will adapt in forward pass
            for cell in self.memory_cells:
                self.h_states.append(torch.zeros(batch_size, cell.lstm_size, device=device))
                self.c_states.append(torch.zeros(batch_size, cell.lstm_size, device=device))
                
    def backward(self, x: torch.Tensor, y: torch.Tensor, learning_rate: float, plasticity: float):
        """
        Backward pass with genetic trait influence
        Args:
            x: Input tensor
            y: Target tensor 
            learning_rate: Base learning rate modified by genetics
            plasticity: Neural plasticity from genetics
        """
        # Forward pass to get outputs and store activations
        outputs, hidden_states = self.forward(x, self.state_manager.get_context())
        
        # Calculate loss with genetic influence
        loss = self.criterion(outputs, y) * (1.0 + plasticity)
        
        # Zero existing gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Apply genetic traits to gradient updates
        for param in self.parameters():
            if param.grad is not None:
                # Scale gradients by plasticity and learning rate
                param.grad *= plasticity
                param.grad *= learning_rate
                
                # Apply adaptive noise based on plasticity
                noise_scale = 0.01 * plasticity
                param.grad += torch.randn_like(param.grad) * noise_scale
        
        # Update weights
        self.optimizer.step()
        
        # Update memory cells with new learning
        self.state_manager.update_memory(hidden_states, loss.item())
        
        return loss.item()

    def process_dream(self, experience: Dict):
        """Process experiences during dream state for memory consolidation"""
        # Extract experience data
        state = torch.tensor(experience['state'], dtype=torch.float32)
        action = torch.tensor(experience['action'], dtype=torch.float32)
        reward = torch.tensor(experience['reward'], dtype=torch.float32)
        
        # Dream-state plasticity modifier
        dream_plasticity = self.mind_genetics.creativity * 1.5
        
        # Process through network with higher plasticity
        output, importance = self.forward(state)
        self.backward(state, action, learning_rate=0.01, plasticity=dream_plasticity)
        
        # Update memory cells with experience
        with torch.no_grad():
            for i, cell in enumerate(self.memory_cells):
                if i >= len(self.h_states):  # Handle dynamic memory cell count
                    self.h_states.append(torch.zeros(state.shape[0], cell.lstm_size, device=state.device))
                    self.c_states.append(torch.zeros(state.shape[0], cell.lstm_size, device=state.device))
                elif self.h_states[i].shape[0] != state.shape[0]:  # Handle batch size changes
                    self.h_states[i] = torch.zeros(state.shape[0], cell.lstm_size, device=state.device)
                    self.c_states[i] = torch.zeros(state.shape[0], cell.lstm_size, device=state.device)
                    
                self.h_states[i], self.c_states[i] = cell(state, (self.h_states[i], self.c_states[i]))
                state = self.h_states[i]

    def adapt_network(self):
        """Adapt network architecture based on genetic traits"""
        # Update layer complexity
        hidden_size = max(64, int(64 * self.mind_genetics.memory_capacity))
        num_layers = max(2, int(2 * self.mind_genetics.pattern_recognition))
        
        # Adjust memory cells
        num_memory_cells = max(1, int(2 * self.mind_genetics.cognitive_growth_rate))
        if len(self.memory_cells) != num_memory_cells:
            new_cells = nn.ModuleList([
                GeneticMemoryCell(hidden_size, hidden_size, self.mind_genetics)
                for _ in range(num_memory_cells - len(self.memory_cells))
            ])
            self.memory_cells.extend(new_cells)
        
        # Reset states for new configuration
        self.reset_states()

    def _calculate_network_architecture(self, genetic_core):
        """Calculate network architecture based on genetic parameters"""
        return {
            'input_size': 64,  # Base input size
            'output_size': 32,  # Base output size
            'memory_size': int(64 * genetic_core.brain_genetics.memory_capacity)  # Dynamic memory size
        }
    def _create_neural_network(self, genetic_core: GeneticCore) -> 'NeuralAdaptiveNetwork':
        """Create neural network with genetic parameters"""
        try:
            # Get network architecture
            architecture = self._calculate_network_architecture(genetic_core)
            # Get network dimensions from the genetic core's traits 
            input_layer_scale = genetic_core.brain_genetics.processing_power
            output_layer_scale = genetic_core.brain_genetics.pattern_recognition

            architecture = {
                'input_size': int(32 * max(1.0, input_layer_scale)),  # Dynamic input scaling
                'output_size': int(16 * max(1.0, output_layer_scale)), # Dynamic output scaling
                'memory_size': int(64 * genetic_core.brain_genetics.memory_capacity)  # Dynamic memory size
            }            
            # Debug logging
            logger.debug(f"Network architecture: {architecture}")
            logger.debug(f"Using brain genetics: {vars(genetic_core.brain_genetics)}")
            
            # Initialize network attributes using self
            self.input_size = architecture['input_size']
            self.output_size = architecture['output_size'] 
            self.memory_size = architecture['memory_size']
            self.learning_rate = genetic_core.brain_genetics.learning_rate
            self.plasticity = genetic_core.brain_genetics.neural_plasticity
            
            return self
        except Exception as e:
            logger.error(f"Neural network creation failed: {str(e)}", exc_info=True)
            raise

    def adapt_to_genetics(self, genetic_core: BaseGeneticCore):
        # Implementation
        pass


class NeuralLayer(nn.Module):
    def __init__(self, in_features, out_features, genetic_traits):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.activation = self._get_activation(genetic_traits.pattern_recognition)
        self.adaptation_rate = genetic_traits.neural_plasticity
        
    def _get_activation(self, pattern_recognition):
        # Higher pattern recognition -> more complex activation
        if (pattern_recognition > 1.5):
            return nn.GELU()
        elif (pattern_recognition > 1.0):
            return nn.ReLU()
        else:
            return nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x


class AdaptiveStateManager(nn.Module):
    def __init__(self, input_dim, hidden_dim, adaptive_rate=0.01,
                 memory_layers=2, memory_type='lstm'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adaptive_rate = adaptive_rate
        self.memory_type = memory_type

        if memory_type == 'lstm':
            self.memory_cells = nn.ModuleList([
                nn.LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(memory_layers)
            ])
            self.h_state = [torch.zeros(1, hidden_dim)
                            for _ in range(memory_layers)]
            self.c_state = [torch.zeros(1, hidden_dim)
                            for _ in range(memory_layers)]
        else:
            self.memory_cells = nn.ModuleList([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(memory_layers)
            ])

        self.compression_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )

        self.importance_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )

        self.register_buffer('state_importance', torch.ones(1, hidden_dim))
        self.register_buffer('memory_allocation', torch.ones(2))

    def forward(self, current_state, context):
        batch_size = current_state.shape[0]

        if self.memory_type == 'lstm':
            if not hasattr(self, 'h_state') or self.h_state[0].shape[0] != batch_size:
                self.h_state = [torch.zeros(batch_size, self.hidden_dim, device=current_state.device)
                                for _ in range(len(self.memory_cells))]
                self.c_state = [torch.zeros(batch_size, self.hidden_dim, device=current_state.device)
                                for _ in range(len(self.memory_cells))]

        processed_state = current_state

        if self.memory_type == 'lstm':
            for i, cell in enumerate(self.memory_cells):
                self.h_state[i], self.c_state[i] = cell(processed_state,
                                                        (self.h_state[i], self.c_state[i]))
                processed_state = self.h_state[i]
        else:
            for cell in self.memory_cells:
                processed_state = torch.relu(cell(processed_state))

        if context.dim() == 2 and context.shape[1] != 1:
            context = context[:, :1]

        compression_signal = self.compression_gate(
            torch.cat([processed_state, context], dim=-1)
        )
        compressed_state = processed_state * compression_signal

        importance_signal = self.importance_generator(compressed_state)

        with torch.no_grad():
            expanded_importance = self.state_importance.expand(batch_size, -1)
            self.state_importance = expanded_importance + self.adaptive_rate * (
                torch.abs(importance_signal) - expanded_importance
            )

            memory_allocation_update = torch.abs(
                importance_signal.mean(dim=-1))
            memory_allocation_update = memory_allocation_update.mean().view(
                1, 1).expand(1, len(self.memory_cells))
            self.memory_allocation += self.adaptive_rate * (
                memory_allocation_update - self.memory_allocation
            )

        return compressed_state, self.state_importance
