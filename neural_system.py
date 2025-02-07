import torch
import torch.nn as nn
import numpy as np
from neural_diagnostics import NeuralDiagnostics
from neural_executor import AdaptiveExecutor
from neural_augmentation import AdaptiveDataAugmenter
import torch.optim as optim
import torch.nn.functional as F  # Import for activation functions


class NeuralAdaptiveSystem:
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers=2,  # Configurable network depth
                 activation_function='relu',  # Configurable activation
                 memory_type='lstm',
                 learning_rates=[1e-3, 1e-4, 1e-5],
                 noise_levels=[0.1, 0.3, 0.5, 0.7]):
        """
        Upgraded Neural Adaptive System with enhanced flexibility and features.

        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden layers
            output_size: Dimension of output
            num_layers: Number of network layers (configurable depth)
            activation_function: Activation function to use (relu, leaky_relu, tanh, sigmoid, elu)
            memory_type: Type of memory cell (lstm or alternative)
            learning_rates: List of learning rates for optimization
            noise_levels: List of noise levels for data augmentation
        """
        self.network = NeuralAdaptiveNetwork(
            input_size, hidden_size, output_size,
            num_layers=num_layers,
            activation_function=activation_function,
            memory_type=memory_type
        )

        self.diagnostics = NeuralDiagnostics(self.network)

        self.executor = AdaptiveExecutor(
            self.network,
            learning_rates=learning_rates
        )

        self.augmenter = AdaptiveDataAugmenter(
            noise_levels=noise_levels
        )

        self.training_history = {
            'losses': [],
            'validation_losses': [],
            'augmentation_strategies': [],
            'diagnostic_metrics': [],
            'optimizer_indices': [] # Track optimizer index
        }

        self.validation_losses = []
        self.current_epoch = 0 # Track epoch


    def train_step(self, inputs, targets, context=None, validation_inputs=None, validation_targets=None):
        """
        Enhanced training step with validation, diagnostics, and adaptive mechanisms.
        """
        self.current_epoch += 1 # Increment epoch counter

        augmented_inputs = self.augmenter.augment(inputs, context)

        diagnostics = self.diagnostics.monitor_network_health(
            augmented_inputs, targets, context, epoch=self.current_epoch # Pass epoch info
        )

        loss, outputs, state_importance = self.executor.execute(
            augmented_inputs, targets, context, diagnostics=diagnostics # Pass diagnostics to executor
        )

        # Adaptive Learning Rate Adjustment based on Anomalies (Automated Response)
        if diagnostics['anomalies']:
            self.executor.adjust_learning_rate(diagnostics['anomalies'])

        # Validation Step and Adaptive Augmentation
        if validation_inputs is not None and validation_targets is not None:
            validation_loss = self._validation_step(validation_inputs, validation_targets, context)
            noise_level = self.augmenter.adjust_augmentation(
                validation_loss, diagnostics=diagnostics
            )
            self.training_history['validation_losses'].append(validation_loss)
        else:
            validation_loss = None # Set to None if no validation
            noise_level = self.augmenter.adjust_augmentation(
                loss, diagnostics=diagnostics
            )

        # Record Training History
        self.training_history['losses'].append(loss)
        self.training_history['augmentation_strategies'].append(
            self.augmenter.augmentation_history['applied_strategies'][-1]
        )
        self.training_history['diagnostic_metrics'].append(diagnostics)
        self.training_history['optimizer_indices'].append(self.executor.current_optimizer_index)

        validation_loss_to_return = self.training_history['validation_losses'][-1] if self.training_history['validation_losses'] else None # Get validation loss to return

        return {
            'loss': loss,
            'outputs': outputs,
            'state_importance': state_importance,
            'diagnostics': diagnostics,
            'noise_level': noise_level,
            'optimizer_index': self.executor.current_optimizer_index,
            'validation_loss': validation_loss_to_return # Include validation_loss here!
        }

    def _validation_step(self, validation_inputs, validation_targets, context):
        """Validation step remains the same"""
        self.network.eval()
        with torch.no_grad():
            outputs, _ = self.network(validation_inputs, context)
            criterion = nn.CrossEntropyLoss() # Changed to CrossEntropyLoss for character classification
            loss = criterion(outputs, validation_targets.squeeze(1)) # Targets are now index tensors, squeeze to match CrossEntropyLoss input
        self.network.train()
        return loss.item()

    def evaluate(self, test_inputs, test_targets, context=None):
        """Evaluation function remains the same"""
        self.network.eval()
        with torch.no_grad():
            outputs, state_importance = self.network(test_inputs, context)
            criterion = nn.CrossEntropyLoss() # Changed to CrossEntropyLoss for character classification
            loss = criterion(outputs, test_targets.squeeze(1)) # Targets are now index tensors, squeeze to match CrossEntropyLoss input
            diagnostic_report = self.diagnostics.get_comprehensive_report()
            augmentation_report = self.augmenter.get_augmentation_report()

        return {
            'loss': loss.item(),
            'outputs': outputs,
            'state_importance': state_importance,
            'diagnostic_report': diagnostic_report,
            'augmentation_report': augmentation_report
        }

    def get_training_summary(self):
        """Enhanced training summary with optimizer indices"""
        summary = {
            'total_losses': self.training_history['losses'],
            'validation_losses': self.training_history['validation_losses'],
            'augmentation_strategies': self.training_history['augmentation_strategies'],
            'optimizer_indices': self.training_history['optimizer_indices'], # Include optimizer indices in summary
            'final_diagnostic_metrics': self.training_history['diagnostic_metrics'][-1]
                if self.training_history['diagnostic_metrics'] else None
        }
        return summary


class NeuralAdaptiveNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=2, activation_function='relu', memory_type='lstm'): # Added activation_function and num_layers
        super().__init__()
        self.state_manager = AdaptiveStateManager(
            hidden_size, hidden_size, memory_type=memory_type # input_size changed to hidden_size as input to state manager is now hidden layer output
        )
        self.num_layers = num_layers # Store num_layers
        self.activation_function_str = activation_function # Store activation function name

        self.activation_function = self._get_activation_function(activation_function) # Get activation function

        self.embedding = nn.Linear(input_size, hidden_size) # Embedding layer for one-hot input

        core_layers = [] # Layers before the final output layer
        current_size = hidden_size
        for _ in range(num_layers): # Use configurable num_layers
            core_layers.append(nn.Linear(current_size, hidden_size))
            core_layers.append(nn.BatchNorm1d(hidden_size))
            core_layers.append(self.activation_function) # Use configurable activation function
            current_size = hidden_size

        self.network = nn.Sequential(*core_layers) # Assign sequential layers to self.network
        self.final_layer = nn.Linear(current_size, output_size) # Final output layer


    def _get_activation_function(self, activation_function_str):
        """Helper to get activation function by name"""
        activation_function_str = activation_function_str.lower()
        if activation_function_str == 'relu':
            return nn.ReLU()
        elif activation_function_str == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_function_str == 'tanh':
            return nn.Tanh()
        elif activation_function_str == 'sigmoid':
            return nn.Sigmoid()
        elif activation_function_str == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Activation function '{activation_function_str}' not supported.")


    def forward(self, x, context):
        x = self.embedding(x) # Apply embedding to one-hot input
        x = self.network(x)  # Use self.network for the core layers
        adaptive_state, importance = self.state_manager(x, context)
        output = self.final_layer(adaptive_state) # Apply final linear layer separately
        return output, importance


class AdaptiveStateManager(nn.Module):
    def __init__(self, input_dim, hidden_dim, adaptive_rate=0.01,
                 memory_layers=2, memory_type='lstm'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adaptive_rate = adaptive_rate
        self.memory_type = memory_type

        if memory_type == 'lstm':
            self.memory_cells = nn.ModuleList([
                nn.LSTMCell(hidden_dim, hidden_dim)
                for _ in range(memory_layers)
            ])
        else:
            self.memory_cells = nn.ModuleList([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
                for _ in range(memory_layers)
            ])

        self.compression_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
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
        self.register_buffer('memory_allocation', torch.ones(memory_layers))

    def forward(self, current_state, context):
        processed_state = current_state # Input from NeuralAdaptiveNetwork, shape [batch_size, seq_len, hidden_dim]

        if self.memory_type == 'lstm':
            batch_size, seq_len, feature_dim = processed_state.size() # Get sequence length and feature dimension
            h_state = [torch.zeros(batch_size, self.hidden_dim).to(processed_state.device) for _ in range(len(self.memory_cells))]
            c_state = [torch.zeros(batch_size, self.hidden_dim).to(processed_state.device) for _ in range(len(self.memory_cells))]

            # Process the sequence step-by-step
            for step in range(seq_len):
                step_input = processed_state[:, step, :] # Get input for the current step: [batch_size, feature_dim]
                for i, cell in enumerate(self.memory_cells):
                    h_state[i], c_state[i] = cell(step_input, (h_state[i], c_state[i])) # LSTMCell expects 2D input
                    step_input = h_state[i] # Output of current LSTM layer becomes input to the next

            processed_state = h_state[-1] # Use the hidden state of the last LSTM layer at the final step as the processed state

        else: # For non-LSTM memory types, keep the existing logic
            for cell in self.memory_cells:
                processed_state = torch.relu(cell(processed_state))

        compression_signal = self.compression_gate(
            torch.cat([processed_state, context], dim=-1)
        )
        compressed_state = processed_state * compression_signal

        importance_signal = self.importance_generator(compressed_state)

        # Take mean across batch and hidden dimensions for memory_allocation update
        memory_allocation_signal = torch.abs(importance_signal).mean(dim=[0, 1]) # Mean across batch and hidden dims
        memory_allocation_update = self.adaptive_rate * (
            memory_allocation_signal - self.memory_allocation.mean() # Compare scalar to scalar mean
        )
        self.memory_allocation += memory_allocation_update # Scalar update


        self.state_importance += self.adaptive_rate * ( # state_importance update remains the same (mean across batch dim)
            torch.abs(importance_signal).mean(dim=0, keepdim=True) - self.state_importance
        )
        return compressed_state, self.state_importance