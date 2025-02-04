# diagnostics.py
import torch
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# --- NEURAL DIAGNOSTICS (neural_diagnostics.py) ---


class NeuralDiagnostics:
    def __init__(self, network, diagnostic_config=None):
        self.network = network

        self.diagnostic_config = diagnostic_config or {
            'gradient_norm_threshold': 10.0,
            'activation_sparsity_threshold': 0.3,
            'weight_divergence_threshold': 5.0,
            'anomaly_detection_sensitivity': 0.95
        }

        self.diagnostic_history = {
            'gradient_norms': [],
            'activation_sparsity': [],
            'weight_distributions': [],
            'loss_curvature': [],
            'feature_importance': []
        }

        self.anomaly_detectors = {
            'gradient_norm': self._detect_gradient_anomalies,
            'activation_sparsity': self._detect_sparsity_anomalies,
            'weight_distribution': self._detect_weight_distribution_anomalies
        }
        self.metrics_history = {}  # Track full history for comprehensive report

    # Added epoch info
    def monitor_network_health(self, inputs, targets, context, epoch=None):
        """Enhanced network health monitoring with comprehensive metrics."""
        diagnostics = {
            'gradient_analysis': self._analyze_gradients(),
            'activation_analysis': self._analyze_activations(inputs, context),
            'weight_analysis': self._analyze_weight_distributions(),
            # Pass context here now!
            'loss_landscape': self._analyze_loss_landscape(inputs, targets, context),
            'anomalies': self._detect_network_anomalies()
        }

        # Update diagnostic history and metrics_history
        for key, value in diagnostics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    # Create unique history key
                    history_key = f"{key}_{subkey}"
                    if history_key not in self.diagnostic_history:
                        self.diagnostic_history[history_key] = []
                        self.metrics_history[history_key] = []
                    self.diagnostic_history[history_key].append(subvalue)
                    self.metrics_history[history_key].append(
                        {'epoch': epoch, 'value': subvalue})  # Track with epoch
            elif isinstance(value, list):
                for item in value:
                    if 'name' in item:  # For weight distributions and gradient details
                        # Unique key per weight layer/grad
                        history_key = f"{key}_{item['name']}"
                        if history_key not in self.diagnostic_history:
                            self.diagnostic_history[history_key] = []
                            self.metrics_history[history_key] = []
                        metric_value = item.get('grad_norm') or item.get(
                            'anomaly_score') or item.get('mean') or 0  # Handle different metrics
                        self.diagnostic_history[history_key].append(
                            metric_value)
                        self.metrics_history[history_key].append(
                            {'epoch': epoch, 'value': metric_value})  # Track with epoch
            else:  # Scalar values, though unlikely in current diagnostics
                if key not in self.diagnostic_history:
                    self.diagnostic_history[key] = []
                    self.metrics_history[key] = []
                self.diagnostic_history[key].append(value)
                self.metrics_history[key].append(
                    {'epoch': epoch, 'value': value})  # Track with epoch

        return diagnostics

    def _analyze_gradients(self):
        """Detailed gradient analysis"""
        grad_details = []
        total_grad_norm = 0

        for param in self.network.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                grad_details.append({
                    'name': param.name if hasattr(param, 'name') else 'unnamed',
                    'grad_norm': grad_norm,
                    'grad_variance': torch.var(param.grad).item()
                })

        return {
            'total_grad_norm': total_grad_norm,
            'grad_details': grad_details,
            'grad_norm_ratio': self._compute_gradient_ratio()
        }

    def _compute_gradient_ratio(self):
        """Compute gradient norm ratios for comparative analysis"""
        grad_norms = [
            p.grad.norm().item()
            for p in self.network.parameters()
            if p.grad is not None
        ]
        return np.std(grad_norms) / (np.mean(grad_norms) + 1e-8)

    def _analyze_activations(self, inputs, context):
        """Comprehensive activation analysis"""
        outputs, state_importance = self.network(inputs, context)

        activation_sparsity = torch.sum(outputs == 0).float() / outputs.numel()

        feature_importance = state_importance.cpu().detach().numpy()

        return {
            'activation_sparsity': activation_sparsity.item(),
            'feature_importance': feature_importance,
            'activation_distribution':  outputs.mean().item()  # <-- Return scalar mean here!
        }

    def _analyze_weight_distributions(self):
        """Advanced weight distribution analysis"""
        weight_stats = []
        for name, param in self.network.named_parameters():
            if param.data is not None:
                weight_stats.append({
                    'name': name,
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'skewness': stats.skew(param.data.cpu().numpy().flatten()),
                    'kurtosis': stats.kurtosis(param.data.cpu().numpy().flatten())
                })

        return weight_stats

    # Added context parameter
    def _analyze_loss_landscape(self, inputs, targets, context):
        """Loss landscape analysis remains similar, now takes context"""
        outputs, _ = self.network(inputs, context)  # Now using context

        outputs.requires_grad_(True)
        first_order_grad = torch.autograd.grad(
            outputs.sum(), inputs, create_graph=True
        )[0]

        loss_curvature = torch.norm(first_order_grad).item()

        return {
            'loss_curvature': loss_curvature,
            'gradient_complexity': torch.std(first_order_grad).item()
        }

    def _detect_network_anomalies(self):
        """Comprehensive anomaly detection"""
        anomalies = {}

        for detector_name, detector_func in self.anomaly_detectors.items():
            anomaly_result = detector_func()
            if anomaly_result:  # Only add if anomalies are detected
                # anomalies is now a dict of lists
                anomalies[detector_name] = anomaly_result

        return anomalies

    def _detect_gradient_anomalies(self):
        """Detect gradient-related anomalies using a z-score based approach."""
        grad_norms = [
            p.grad.norm().item()
            for p in self.network.parameters()
            if p.grad is not None
        ]

        if not grad_norms:
            return []

        mean_grad_norm = np.mean(grad_norms)
        std_grad_norm = np.std(grad_norms)

        if std_grad_norm == 0:
            return []

        z_scores = [(grad_norm - mean_grad_norm) /
                    std_grad_norm for grad_norm in grad_norms]
        threshold = stats.norm.ppf(
            self.diagnostic_config['anomaly_detection_sensitivity'])

        anomalous_gradients = [
            {'name': p.name, 'grad_norm': p.grad.norm().item(), 'z_score': z}
            for p, z in zip(self.network.parameters(), z_scores)
            if p.grad is not None and z > threshold
        ]
        # Return None if no anomalies
        return anomalous_gradients if anomalous_gradients else None

    def _detect_sparsity_anomalies(self):
        """Detect activation sparsity anomalies"""
        sparsity_history = self.diagnostic_history.get(
            'activation_sparsity', [])

        if not sparsity_history:
            return []

        mean_sparsity = np.mean(sparsity_history)
        std_sparsity = np.std(sparsity_history)

        if std_sparsity == 0:
            return []

        z_scores = [(sp - mean_sparsity) /
                    std_sparsity for sp in sparsity_history]
        threshold = stats.norm.ppf(
            self.diagnostic_config['anomaly_detection_sensitivity'])

        anomalous_sparsity = [
            {'sparsity': sp, 'z_score': z}
            for sp, z in zip(sparsity_history, z_scores)
            if z > threshold
        ]
        # Return None if no anomalies
        return anomalous_sparsity if anomalous_sparsity else None

    def _detect_weight_distribution_anomalies(self):
        """Weight distribution anomaly detection remains similar"""
        weight_distributions = self.diagnostic_history.get(
            'weight_distributions', [])

        if not weight_distributions:
            return []

        anomalies = []
        for dist_list in weight_distributions:
            for dist in dist_list:
                if 'skewness' not in dist or 'kurtosis' not in dist:
                    continue

                anomaly_score = self._compute_distribution_anomaly(dist)
                if anomaly_score > self.diagnostic_config['anomaly_detection_sensitivity']:
                    anomalies.append(
                        {'name': dist['name'], 'anomaly_score': anomaly_score})

        return anomalies if anomalies else None  # Return None if no anomalies

    def _compute_distribution_anomaly(self, distribution):
        """Distribution anomaly score computation remains similar"""
        skewness = distribution.get('skewness', 0)
        kurtosis = distribution.get('kurtosis', 0)

        expected_skewness = 0
        expected_kurtosis = 3

        chi2_skew = (skewness - expected_skewness)**2 / \
            (expected_skewness**2 + 1e-8)
        chi2_kurt = (kurtosis - expected_kurtosis)**2 / \
            (expected_kurtosis**2 + 1e-8)

        anomaly_score = np.sqrt(chi2_skew + chi2_kurt)

        return anomaly_score

    def get_comprehensive_report(self):
        """Enhanced comprehensive report with more stats and full history access."""
        report = {}
        for metric, history in self.metrics_history.items():  # Use metrics_history for epoch-wise tracking
            if history:
                values = [item['value'] for item in history]  # Extract values

                # Debugging: Print metric name and type of values *before* np.mean
                print(
                    f"Debugging Metric: {metric}, Type of values: {type(values)}")

                # General flattening logic (keep this for robustness)
                if values and isinstance(values[0], list):
                    flattened_values = []
                    for item_value in values:
                        flattened_values.extend(item_value)
                    values = flattened_values

                try:  # Keep the try-except block for robustness
                    report[metric] = {
                        'mean': np.mean(values) if values else np.nan,
                        'std': np.std(values) if values else np.nan,
                        'min': np.min(values) if values else np.nan,
                        'max': np.max(values) if values else np.nan,
                        'median': np.median(values),
                        'percentile_25': np.percentile(values, 25),
                        'percentile_75': np.percentile(values, 75),
                        'history': history
                    }
                except TypeError as e:  # Catch TypeError specifically
                    # Print metric name in case of error
                    print(
                        f"TypeError encountered for metric: {metric}, Error: {e}")
                    raise  # Re-raise the error after printing debug info

            return report


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
