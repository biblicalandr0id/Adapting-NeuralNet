# neural_networks.py
import numpy as np
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Type, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from base_types import BaseGeneticCore  # Changed from genetics import
from genetics import GeneticCore  # This creates the cycle
# Remove import for EnvironmentalState
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NeuralAdaptiveNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, genetic_core):
        super().__init__()
        self.mind_genetics = genetic_core.mind_genetics
        self.brain_genetics = genetic_core.brain_genetics
        self.output_size = output_size
        self.genetic_core = genetic_core # Store genetic core
        self.hidden_size = hidden_size

        # Base network parameters from genetics
        hidden_size = max(64, int(getattr(self.mind_genetics, 'memory_capacity', 1.0) * 64))
        num_layers = max(2, int(getattr(self.mind_genetics, 'pattern_recognition', 1.0) * 2))
        
        # Genetic parameters
        context_dim = max(1, int(getattr(self.brain_genetics, 'sensor_sensitivity', 1.0) * 10))
        use_skip_connections = getattr(self.mind_genetics, 'creativity', 0.0) > 0.7
        gnn_layers = max(0, int(getattr(self.mind_genetics, 'cognitive_growth_rate', 1.0) - 1))

        # Dynamically adjust hidden size
        reasoning_hidden_size = int(hidden_size * (1 + 0.5 * getattr(self.mind_genetics, 'pattern_recognition', 1.0)))
        
        # Create state manager for memory handling
        self.state_manager = AdaptiveStateManager(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            adaptive_rate=getattr(self.brain_genetics, 'learning_rate', 0.01),
            memory_type='lstm',
            context_dim=context_dim,  # Genetically controlled context dimension
            genetic_core = genetic_core # Pass genetic core
        )

        # Reasoning Module parameters
        reasoning_layers = max(1, int(getattr(self.mind_genetics, 'cognitive_growth_rate', 1.0)))  # Genetic control
        reasoning_type = 'linear'  # Default reasoning type
        if getattr(self.mind_genetics, 'creativity', 0.0) > 1.5:
            reasoning_type = 'transformer'
        elif getattr(self.mind_genetics, 'pattern_recognition', 0.0) > 1.0:
            reasoning_type = 'attention'
        elif getattr(self.mind_genetics, 'cognitive_growth_rate', 0.0) > 2.0:
            reasoning_type = 'gnn'
            
        self.reasoning_module = ReasoningModule(hidden_size, reasoning_hidden_size, reasoning_layers, 
                                                genetic_core, reasoning_type, use_skip_connections, gnn_layers)
        
        # Genetic control for feedback and influence
        self.reasoning_influence = nn.Parameter(torch.tensor(getattr(self.mind_genetics, 'reasoning_influence', 0.5)))
        self.memory_feedback_strength = nn.Parameter(torch.tensor(getattr(self.mind_genetics, 'memory_feedback_strength', 0.5)))
        
        # Create genetic layers with state management
        self.layers = nn.ModuleList()
        current_size = input_size
        
        for i in range(num_layers):
            layer_size = int(hidden_size * (1 + 0.2 * (i - num_layers/2)))
            genetic_layer = GeneticLayer(current_size, layer_size, self.brain_genetics)
            self.layers.append(genetic_layer)
            current_size = layer_size
        
        # Memory cells from GeneticMemoryCell
        num_memory_cells = max(1, int(getattr(self.mind_genetics, 'cognitive_growth_rate', 1.0) * 2))
        self.memory_cells = nn.ModuleList([
            GeneticMemoryCell(current_size, hidden_size, self.mind_genetics)
            for _ in range(num_memory_cells)
        ])
        
        self.output = nn.Linear(hidden_size, output_size)
        self.reset_states()
        
        # Add optimizer initialization with genetic traits influence
        learning_rate = getattr(self.brain_genetics, 'learning_rate', 0.001)
        beta1 = max(0.5, min(0.99, getattr(self.mind_genetics, 'pattern_recognition', 0.9)))
        beta2 = max(0.8, min(0.999, getattr(self.mind_genetics, 'learning_efficiency', 0.99)))
        
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=1e-8 * getattr(self.brain_genetics, 'neural_plasticity', 0.1)
        )
        
        # Loss criterion with genetic scaling
        self.criterion = nn.MSELoss()
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, 
                adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device
        
        if context is None:
            context = torch.zeros(batch_size, 1, device=device)
            
        # Process through genetic layers
        importance_signals = []
        for layer in self.layers:
            x, importance = layer(x, context)
            importance_signals.append(importance)
            
        # Process through state manager
        compressed_state, _ = self.state_manager(x, context)
        # Process through reasoning module and return output
        reasoned_state = self.reasoning_module(compressed_state)
        state_importance = torch.mean(torch.abs(compressed_state), dim=0)
        # Memory feedback
        memory_feedback = torch.zeros_like(compressed_state)  # Initialize feedback
        for i, cell in enumerate(self.memory_cells):
            if i >= len(self.h_states):  # Handle dynamic memory cell count
                self.h_states.append(torch.zeros(batch_size, cell.lstm_size, device=device))
                self.c_states.append(torch.zeros(batch_size, cell.lstm_size, device=device))
            elif self.h_states[i].shape[0] != batch_size:  # Handle batch size changes
                self.h_states[i] = torch.zeros(batch_size, cell.lstm_size, device=device)
                self.c_states[i] = torch.zeros(batch_size, cell.lstm_size, device=device)
                
            self.h_states[i], self.c_states[i] = cell(compressed_state + memory_feedback * getattr(self.genetic_core.mind_genetics, 'memory_feedback_strength', 0.5), (self.h_states[i], self.c_states[i]))
            memory_feedback = self.h_states[i]  # Use h_state as feedback

        # Process through reasoning module
        if self.reasoning_module.reasoning_type == 'gnn':
            if adj_matrix is None:
                logger.warning("Adjacency matrix not provided, using identity matrix as default.")
                adj_matrix = torch.eye(compressed_state.size(1), device=device)  # Identity matrix
            reasoned_state = self.reasoning_module(compressed_state, adj_matrix)
        else:
            reasoned_state = self.reasoning_module(compressed_state)
            
        # Influence state manager with reasoning
        enhanced_context = context + reasoned_state.mean(dim=-1, keepdim=True) * getattr(self.genetic_core.mind_genetics, 'reasoning_influence', 0.5)
        compressed_state, state_importance = self.state_manager(x, enhanced_context)
        
        # Final output with proper shape for action selection
        output = self.output(memory_feedback)
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
        
        # Dream-state plasticity modifier
        reward = experience.get('reward', 0.0)  # Get reward if available
        dream_plasticity = self.mind_genetics.learning_efficiency * 1.5 * (1.0 + abs(reward))  # Plasticity influenced by reward
     
        # Process through network with higher plasticity
        _, _ = self.forward(state)
        self.backward(state, action, learning_rate=0.01, plasticity=dream_plasticity)
        output, hidden_states = self.forward(state)  # Get dream processing outputs
        dream_loss = self.backward(state, action, learning_rate=0.01, plasticity=dream_plasticity)

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

        # Return dream processing results 
        return {
            'output': output.detach().numpy(),
            'loss': dream_loss,
            'plasticity': dream_plasticity
        }        

    def adapt_network(self):
        """Adapt network architecture based on genetic traits"""
        # Update layer complexity
        hidden_size = max(64, int(getattr(self.mind_genetics, 'memory_capacity', 1.0) * 64))
        num_layers = max(2, int(getattr(self.mind_genetics, 'pattern_recognition', 1.0) * 2))
        
        # Adjust memory cells
        num_memory_cells = max(1, int(getattr(self.mind_genetics, 'cognitive_growth_rate', 1.0) * 2))
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
            'memory_size': int(getattr(self.brain_genetics, 'memory_capacity', 1.0) * 64)  # Dynamic memory size
        }
    def _create_neural_network(self, genetic_core: GeneticCore) -> 'NeuralAdaptiveNetwork':
        """Create neural network with genetic parameters"""
        try:
            # Get network architecture
            architecture = self._calculate_network_architecture(genetic_core)
            # Get network dimensions from the genetic core's traits 
            input_layer_scale = getattr(self.brain_genetics, 'processing_power', 1.0)
            output_layer_scale = getattr(self.brain_genetics, 'pattern_recognition', 1.0)

            architecture = {
                'input_size': int(32 * max(1.0, input_layer_scale)),  # Dynamic input scaling
                'output_size': int(16 * max(1.0, output_layer_scale)), # Dynamic output scaling
                'memory_size': int(getattr(self.brain_genetics, 'memory_capacity', 1.0) * 64)  # Dynamic memory size
            }            
            # Debug logging
            logger.debug(f"Network architecture: {architecture}")
            logger.debug(f"Using brain genetics: {vars(genetic_core.brain_genetics)}")
            
            # Initialize network attributes using self
            self.input_size = architecture['input_size']
            self.output_size = architecture['output_size'] 
            self.memory_size = architecture['memory_size']
            self.learning_rate = getattr(self.brain_genetics, 'learning_rate', 0.001)
            self.plasticity = getattr(self.brain_genetics, 'neural_plasticity', 0.1)
            
            return self
        except Exception as e:
            logger.error(f"Neural network creation failed: {str(e)}", exc_info=True)
            raise

    def adapt_to_genetics(self, genetic_core: GeneticCore):
        # Implementation
        pass

class GeneticLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, genetic_traits):
        super().__init__()
        self.input_size = input_size 
        self.output_size = output_size
        
        # Genetic trait influences
        self.pattern_recognition = getattr(genetic_traits, 'pattern_recognition', 1.0)
        self.neural_plasticity = getattr(genetic_traits, 'neural_plasticity', 0.1)
        self.processing_speed = getattr(genetic_traits, 'processing_speed', 1.0)
        self.trait_stability = getattr(genetic_traits, 'trait_stability', 0.8)
        self.development_plasticity = getattr(genetic_traits, 'development_plasticity', 0.6)
        self.mutation_chance = getattr(genetic_traits, 'mutation_chance', 0.05)
        
        # Main processing layers
        self.linear = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Pattern detection influenced by genetics 
        self.pattern_detector = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
            nn.Sigmoid()
        )
        
        # Importance estimation
        self.importance_estimator = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size),
            nn.Sigmoid()
        )
        
        # Adaptive noise scale based on plasticity
        self.noise_scale = nn.Parameter(torch.tensor(0.1 * self.neural_plasticity))
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Main processing path
        main = self.linear(x)
        main = self.layer_norm(main)
        
        # Pattern detection path
        patterns = self.pattern_detector(x)
        patterns = patterns * self.pattern_recognition
        
        # Apply processing speed to main path
        main = main * self.processing_speed
        
        # Add adaptive noise based on plasticity
        if self.training:
            noise = torch.randn_like(main) * self.noise_scale * self.neural_plasticity
            main = main + noise
        
        # Combine paths
        output = main * patterns
        
        # Generate importance signals
        importance = self.importance_estimator(output)
        
        # Apply activation based on pattern recognition level
        if self.pattern_recognition > 1.5:
            output = F.gelu(output)
        elif self.pattern_recognition > 1.0:
            output = F.relu(output)
        else:
            output = F.leaky_relu(output, 0.1)
            
        return output, importance
        
    def adapt_weights(self, learning_rate: float):
        """Adapt weights based on genetic traits"""
        with torch.no_grad():
            # Scale weights based on plasticity
            self.linear.weight.data *= (1.0 + self.neural_plasticity * learning_rate)
            
            # Add small random adaptations
            noise = torch.randn_like(self.linear.weight) * self.noise_scale
            self.linear.weight.data += noise
            
            # Normalize to prevent explosion
            self.linear.weight.data = F.normalize(self.linear.weight.data, dim=1)
            
        # LSTM processing scaled by learning efficiency
        h_state, c_state = self.lstm(x, (h_state, c_state))
        
        # Apply memory gating
        memory_importance = self.memory_gate(h_state)
        h_state = h_state * memory_importance * self.learning_efficiency
        
        return h_state, c_state


class GraphNeuralLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(torch.matmul(adj_matrix, x))
        return x

class GeneticMemoryCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, genetic_traits):
        super().__init__()
        self.input_size = input_size
        self.memory_capacity = genetic_traits.memory_capacity
        self.learning_efficiency = genetic_traits.learning_efficiency
        self.lstm_size = int(hidden_size * self.memory_capacity)
        
        self._initialize_lstm(hidden_size)
        self._initialize_memory_gate()
        
    def _initialize_lstm(self, hidden_size):
        self.lstm = nn.LSTMCell(self.input_size, self.lstm_size)
        
    def _initialize_memory_gate(self):
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


class ReasoningModule(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 genetic_core: GeneticCore, reasoning_type: str = 'linear', use_skip_connections: bool = True,
                 num_gnn_layers: int = 1):
        super().__init__()
        self.reasoning_type = reasoning_type
        self.use_skip_connections = use_skip_connections
        self.layers = nn.ModuleList()
        self.num_gnn_layers = num_gnn_layers
        self.genetic_core = genetic_core

        # Ensure input_size is divisible by num_heads for attention/transformer
        self.num_heads = 4
        self.adjusted_size = ((input_size + self.num_heads - 1) // self.num_heads) * self.num_heads
        self.input_projection = nn.Linear(input_size, self.adjusted_size) if input_size != self.adjusted_size else nn.Identity()

        if reasoning_type == 'linear':
            for i in range(num_layers):
                self.layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        elif reasoning_type == 'attention':
            self.attention = nn.MultiheadAttention(self.adjusted_size, num_heads=self.num_heads)
            for i in range(num_layers - 1):
                self.layers.append(nn.Linear(self.adjusted_size if i == 0 else hidden_size, hidden_size))
        elif reasoning_type == 'transformer':
            self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.adjusted_size, nhead=self.num_heads)
            self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        elif reasoning_type == 'gnn':
            self.gnn_layers = nn.ModuleList()
            for i in range(num_gnn_layers):
                self.gnn_layers.append(GraphNeuralLayer(input_size if i == 0 else hidden_size, hidden_size))
        else:
            raise ValueError(f"Unknown reasoning type: {reasoning_type}")

        self.output = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.reasoning_type in ['attention', 'transformer']:
            # Ensure input is properly shaped and sized for attention
            x = self.input_projection(x)
            # Add batch dimension if needed
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # Ensure sequence dimension is present
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
                
        if self.reasoning_type == 'linear':
            residual = x  # Initial residual for skip connection
            return x
        elif self.reasoning_type == 'attention':
            # Add sequence dimension if needed
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            x, _ = self.attention(x, x, x)
            for layer in self.layers:
                x = torch.relu(layer(x))
            x = self.output(x)
            return x.squeeze(0)
        elif self.reasoning_type == 'transformer':
            # Add sequence dimension if needed
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            x = self.transformer(x)
            return x.squeeze(0)
        elif self.reasoning_type == 'gnn':
            if adj_matrix is None:
                logger.warning("Adjacency matrix not provided, using identity matrix as default.")
                adj_matrix = torch.eye(x.size(1), device=x.device)  # Identity matrix
            for layer in self.gnn_layers:
                x = layer(x, adj_matrix)
            x = self.output(x)
            return x
        else:
            raise ValueError(f"Unknown reasoning type: {self.reasoning_type}")

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
                 memory_layers=2, memory_type='lstm', context_dim=1,
                 genetic_core: Optional[GeneticCore] = None):  # Added genetic_core
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adaptive_rate = adaptive_rate
        self.memory_type = memory_type
        self.context_dim = context_dim # Store context dimension
        self.genetic_core = genetic_core # Store genetic core

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
            nn.Linear(hidden_dim + context_dim, hidden_dim),  # Use context_dim
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

        # Removed this check
        # if context.dim() == 2 and context.shape[1] != 1:
        #     context = context[:, :1]

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


class MemorySystem(nn.Module):
    def __init__(self, size: int, retention: float, processing_speed: float):
        super().__init__()
        self.memory_bank = nn.Parameter(torch.zeros(size, 512))
        self.retention = retention
        self.processing_speed = processing_speed
        
    def query(self, x: torch.Tensor) -> torch.Tensor:
        # Similarity-based memory retrieval
        similarities = F.cosine_similarity(x.unsqueeze(1), self.memory_bank, dim=2)
        attention = F.softmax(similarities * self.processing_speed, dim=1)
        return torch.matmul(attention, self.memory_bank)
        
    def update(self, x: torch.Tensor, importance: torch.Tensor) -> None:
        # Update memory based on importance and retention
        update_mask = importance > (1 - self.retention)
        self.memory_bank.data[update_mask] = x[update_mask]

class PatternDetectionLayer(nn.Module):
    def __init__(self, input_size: int, sensitivity: float):
        super().__init__()
        self.sensitivity = sensitivity
        self.pattern_bank = nn.Parameter(torch.randn(32, input_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pattern detection with genetic sensitivity
        similarities = F.cosine_similarity(x.unsqueeze(1), 
                                        self.pattern_bank, dim=2)
        return similarities * self.sensitivity

class AdaptiveLayer(nn.Module):
    def __init__(self, size: int, creativity: float, adaptation_rate: float):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(size, size))
        self.creativity = creativity
        self.adaptation_rate = adaptation_rate
        
    def forward(self, x: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
        # Adaptive processing influenced by genetics
        adaptation = patterns.mean() * self.adaptation_rate
        creative_noise = torch.randn_like(x) * self.creativity
        return x + torch.matmul(x, self.weights) * adaptation + creative_noise

class ActionResult:
    """Represents the result of an action"""
    def __init__(self, reward: float, next_state: np.ndarray):
        self.reward = reward
        self.next_state = next_state

class NeuralBehavior:
    """Handles neural network based decision making"""
    def __init__(self, neural_net: NeuralAdaptiveNetwork, genetic_core: GeneticCore):
        self.neural_net = neural_net
        self.genetic_core = genetic_core
        
    def process_perception(self, env_state: Any) -> np.ndarray:
        # Convert any environment state to neural input
        if isinstance(env_state, (np.ndarray, list, torch.Tensor)):
            state_vector = torch.tensor(env_state, dtype=torch.float32)
        elif hasattr(env_state, 'get_state_vector'):
            state_vector = torch.tensor(env_state.get_state_vector(), dtype=torch.float32)
        elif hasattr(env_state, '__array__'):
            state_vector = torch.tensor(np.array(env_state), dtype=torch.float32)
        else:
            try:
                state_vector = torch.tensor(list(env_state), dtype=torch.float32)
            except:
                raise ValueError(f"Cannot convert environment state of type {type(env_state)} to tensor")

        processed_state = self.neural_net.state_manager.forward(state_vector, torch.zeros(1))[0]
        return processed_state.detach().numpy()
        
    def decide_action(self, sensor_data: np.ndarray) -> Tuple[str, Dict]:
        # Process sensor data through neural network
        input_tensor = torch.tensor(sensor_data, dtype=torch.float32)
        output, importance = self.neural_net(input_tensor)
        action_idx = torch.argmax(output).item()
        return str(action_idx), {"importance": importance.detach().numpy()}
        
    def learn_from_experience(self, state: np.ndarray, action: str, 
                            result: ActionResult) -> None:
        # Update neural network based on experience
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(int(action), dtype=torch.long)
        
        # Scale learning rate and plasticity based on reward
        reward_scaling = max(0.1, min(2.0, abs(result.reward)))
        adjusted_learning_rate = self.genetic_core.brain_genetics.learning_rate * reward_scaling
        adjusted_plasticity = self.genetic_core.brain_genetics.neural_plasticity * reward_scaling
        
        self.neural_net.backward(
            state_tensor,
            action_tensor,
            learning_rate=adjusted_learning_rate,
            plasticity=adjusted_plasticity
        )

    def process_dream(self, experience: Dict):
        # Add trait stability influence
        dream_stability = self.genetic_core.embryo_genetics.trait_stability
        mutation_chance = self.genetic_core.embryo_genetics.mutation_chance
        
        # Extract experience data
        state = torch.tensor(experience['state'], dtype=torch.float32)
        action = torch.tensor(experience['action'], dtype=torch.float32)
        
        # Dream-state plasticity modifier
        reward = experience.get('reward', 0.0)  # Get reward if available
        dream_plasticity = (
            self.mind_genetics.learning_efficiency * 1.5 * 
            (1.0 + abs(reward)) * 
            self.genetic_core.embryo_genetics.development_plasticity
        )
     
        # Process through network with higher plasticity
        _, _ = self.forward(state)
        self.backward(state, action, learning_rate=0.01, plasticity=dream_plasticity)
        output, hidden_states = self.forward(state)  # Get dream processing outputs
        dream_loss = self.backward(state, action, learning_rate=0.01, plasticity=dream_plasticity)

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

        # Return dream processing results 
        return {
            'output': output.detach().numpy(),
            'loss': dream_loss,
            'plasticity': dream_plasticity
        }
