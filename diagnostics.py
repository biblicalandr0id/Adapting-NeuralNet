# diagnostics.py
from __future__ import annotations
import torch
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Tuple
import logging
import time
from collections import defaultdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# --- NEURAL DIAGNOSTICS (neural_diagnostics.py) ---


class NeuralDiagnostics:
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.last_update = time.time()
        self.performance_stats = {
            'forward_pass_time': [],
            'backward_pass_time': [],
            'memory_usage': []
        }
        self.alert_thresholds = {
            'gradient_norm': 10.0,
            'activation_saturation': 0.9,
            'dead_neurons': 0.1
        }
        
    def monitor_network_health(self, inputs: torch.Tensor, targets: torch.Tensor, 
                             context: torch.Tensor, epoch: int) -> Dict:
        current_time = time.time()
        metrics = {}
        
        try:
            # Basic network statistics
            metrics['input_range'] = (inputs.min().item(), inputs.max().item())
            metrics['target_range'] = (targets.min().item(), targets.max().item())
            metrics['context_value'] = context.mean().item()
            metrics['epoch'] = epoch
            metrics['timestamp'] = current_time
            
            # Performance tracking
            metrics['time_since_last_update'] = current_time - self.last_update
            self.last_update = current_time
            
            # Store metrics
            for key, value in metrics.items():
                self.metrics_history[key].append({
                    'value': value,
                    'epoch': epoch,
                    'timestamp': current_time
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in network health monitoring: {str(e)}")
            return {'error': str(e)}
            
    def close(self):
        """Cleanup and save final diagnostics"""
        try:
            # Save final metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"simulation_stats/neural_diagnostics_{timestamp}.json"
            
            with open(save_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving neural diagnostics: {str(e)}")


class AgentDiagnostics:
    """Monitors and analyzes agent behavior and performance"""
    def __init__(self):
        self.metrics_history = {
            'energy_levels': [],
            'action_distribution': {},
            'survival_time': [],
            'reproduction_success': [],
            'genetic_diversity': [],
            'neural_complexity': []
        }
        
    def update_metrics(self, agent: 'AdaptiveAgent'):
        """Record agent metrics for analysis"""
        self.metrics_history['energy_levels'].append(agent.energy)
        
        # Track action choices
        action = agent.last_action
        self.metrics_history['action_distribution'][action] = \
            self.metrics_history['action_distribution'].get(action, 0) + 1
            
        # Calculate neural complexity
        neural_complexity = self._calculate_neural_complexity(agent.neural_net)
        self.metrics_history['neural_complexity'].append(neural_complexity)
        
    def _calculate_neural_complexity(self, neural_net) -> float:
        """Measure complexity of neural network"""
        total_params = sum(p.numel() for p in neural_net.parameters())
        connection_density = self._calculate_connection_density(neural_net)
        return total_params * connection_density


class PopulationDiagnostics:
    """Analyzes population-level metrics and trends"""
    def __init__(self):
        self.generation_metrics = {}
        self.population_trends = {
            'size': [],
            'avg_fitness': [],
            'genetic_diversity': [],
            'species_distribution': {}
        }
    
    def update_generation_metrics(self, generation: int, agents: List['AdaptiveAgent']):
        """Record metrics for current generation"""
        if not agents:
            return
            
        metrics = {
            'population_size': len(agents),
            'avg_energy': np.mean([a.energy for a in agents]),
            'avg_age': np.mean([a.age for a in agents]),
            'genetic_diversity': self._calculate_genetic_diversity(agents)
        }
        
        self.generation_metrics[generation] = metrics
