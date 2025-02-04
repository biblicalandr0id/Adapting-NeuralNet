import torch
import numpy as np
import random
import torchvision.transforms as transforms


# --- ADAPTIVE DATA AUGMENTER (neural_augmentation.py) ---
class AdaptiveDataAugmenter:
    def __init__(self,
                 noise_levels=[0.1, 0.3, 0.5, 0.7],
                 augmentation_strategies=None):
        self.noise_levels = noise_levels
        self.current_noise_index = 0

        self.augmentation_strategies = augmentation_strategies or [
            'gaussian_noise',
            'dropout',
            'mixup',
            'feature_space_augmentation',
            'horizontal_flip',  # New Augmentation
            'vertical_flip',   # New Augmentation
            # 'temporal_warping' # Temporal warping removed from defaults
        ]
        self.current_strategy_index = 0

        self.augmentation_history = {
            'applied_strategies': [],
            'noise_levels': [],
            'performance_impact': []
        }
        self.horizontal_flip_transform = transforms.RandomHorizontalFlip(
            p=0.5)  # Pre-initialize transforms
        self.vertical_flip_transform = transforms.RandomVerticalFlip(p=0.5)

    def augment(self, data, context=None):
        """
        Augment data with multiple strategies, including new flips.
        """
        strategy = self.augmentation_strategies[self.current_strategy_index]
        noise_scale = self.noise_levels[self.current_noise_index]

        augmented_data = self._apply_augmentation(
            data, strategy, noise_scale, context)

        self.augmentation_history['applied_strategies'].append(strategy)
        self.augmentation_history['noise_levels'].append(noise_scale)

        return augmented_data

    def _apply_augmentation(self, data, strategy, noise_scale, context=None):
        """Apply augmentation strategies, including flips."""
        if strategy == 'gaussian_noise':
            return data + torch.randn_like(data) * noise_scale

        elif strategy == 'dropout':
            mask = torch.bernoulli(torch.full(
                data.shape, 1 - noise_scale)).bool()
            return data.masked_fill(mask, 0)

        elif strategy == 'mixup':
            batch_size = data.size(0)
            shuffle_indices = torch.randperm(batch_size)
            lam = np.random.beta(noise_scale, noise_scale)
            return lam * data + (1 - lam) * data[shuffle_indices]

        elif strategy == 'feature_space_augmentation':
            transform_matrix = torch.randn_like(
                data) * noise_scale * data.std()
            return data + transform_matrix

        elif strategy == 'horizontal_flip':  # Horizontal Flip Augmentation
            return self.horizontal_flip_transform(data)

        elif strategy == 'vertical_flip':   # Vertical Flip Augmentation
            return self.vertical_flip_transform(data)

        elif strategy == 'temporal_warping':  # Temporal Warping - kept, but not in defaults
            seq_len = data.size(1)
            warp_points = np.sort(random.sample(
                range(seq_len), int(seq_len*noise_scale)))
            warped_data = data.clone()
            for point in warp_points:
                offset = random.choice([-1, 1])
                if 0 <= point + offset < seq_len:
                    warped_data[:, point] = (
                        data[:, point] + data[:, point + offset]) / 2
            return warped_data

        return data

    def adjust_augmentation(self, network_performance, diagnostics=None):
        """Adjust augmentation based on performance and diagnostics."""
        base_adjustment = self._compute_adjustment(
            network_performance, diagnostics)

        if base_adjustment < 0.4:  # Adjusted thresholds for more frequent changes
            self.current_noise_index = min(
                self.current_noise_index + 1,
                len(self.noise_levels) - 1
            )
            if base_adjustment < 0.2:  # More aggressive strategy switch at lower performance
                self.current_strategy_index = (
                    self.current_strategy_index + 1
                ) % len(self.augmentation_strategies)
        elif base_adjustment > 0.7:  # Adjusted thresholds
            self.current_noise_index = max(
                self.current_noise_index - 1,
                0
            )

        self.augmentation_history['performance_impact'].append(base_adjustment)
        return self.noise_levels[self.current_noise_index]

    def _compute_adjustment(self, performance, diagnostics=None):
        """Compute adjustment score based on performance and diagnostics."""
        if diagnostics is None:
            return performance

        sparsity = diagnostics.get('activation_analysis', {}).get(
            'activation_sparsity', 0)
        curvature = diagnostics.get(
            'loss_landscape', {}).get('loss_curvature', 0)
        grad_norm_ratio = diagnostics.get('gradient_analysis', {}).get(
            'grad_norm_ratio', 0)  # Use grad norm ratio

        # More weight on sparsity and grad_norm_ratio in adjustment
        adjustment_score = (performance + (1 - sparsity) * 2 -
                            curvature/10 + (1-grad_norm_ratio)) / 4.5  # Adjusted weights
        return adjustment_score

    def get_augmentation_report(self):
        """Generate augmentation report."""
        return self.augmentation_history

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

# --- AGENT IMPLEMENTATIONS (system-improvements.py + agent-implementations.py - Combined and Improved Neural Network and Environment) ---


class ResourceType(Enum):  # From agent-architecture.py
    ENERGY = "energy"
    INFORMATION = "information"
    MATERIALS = "materials"


@dataclass  # From agent-architecture.py
class Resource:
    type: ResourceType
    quantity: float
    position: Tuple[int, int]
    complexity: float  # How difficult it is to extract/process
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass  # From agent-architecture.py
class EnvironmentalState:
    """Represents the current state of the environment"""
    resources: List[Resource]
    threats: List[Tuple[int, int]]  # Positions of hazards/threats
    time_step: int
    complexity_level: float  # Overall difficulty/complexity of current state
    agents: List['AdaptiveAgent'] = field(
        default_factory=list)  # Added agents to state


class AgentAction(Enum):  # From agent-architecture.py
    MOVE = "move"
    GATHER = "gather"
    PROCESS = "process"
    SHARE = "share"
    DEFEND = "defend"
    EXECUTE_TOOL = "execute_tool"  # New tool execution action


@dataclass  # From agent-architecture.py
class ActionResult:
    """Outcome of an agent's action"""
    success: bool
    reward: float
    energy_cost: float
    new_state: Optional[Dict]


# From agent-architecture.py + agent-implementations.py + system-improvements.py (Combined and Improved)
class AdaptiveAgent:
    def __init__(self, genetic_core, neural_net, position: Tuple[int, int]):
        self.genetic_core = genetic_core
        self.neural_net = neural_net
        self.position = position

        # Internal state
        self.energy = 100.0
        self.resources = {rt: 0.0 for rt in ResourceType}
        self.knowledge_base = {}  # Learned patterns/information

        # Performance metrics
        self.total_resources_gathered = 0.0
        self.successful_interactions = 0
        self.survival_time = 0
        self.efficiency_score = 0.0
        self.name = EmbryoNamer().generate_random_name()  # Assign name on creation
        self.data_augmenter = AdaptiveDataAugmenter()  # Initialize Data Augmenter
        self.neural_diagnostics = NeuralDiagnostics(
            neural_net)  # Initialize Diagnostics

    def augment_perception(self, inputs, context=None):  # New Augmentation Method
        """Augment sensor data"""
        return self.data_augmenter.augment(inputs, context)

    def perceive_environment(self, env_state: EnvironmentalState) -> np.ndarray:
        """Process environmental inputs based on genetic sensory traits"""
        sensor_sensitivity = self.genetic_core.physical_genetics.sensor_sensitivity

        # Basic sensor inputs
        inputs = []

        # Resource detection (affected by sensor sensitivity)
        for resource in env_state.resources:
            distance = self._calculate_distance(resource.position)
            detection_threshold = 10.0 / sensor_sensitivity

            if distance <= detection_threshold:
                # Clearer detection of closer resources
                clarity = 1.0 - (distance / detection_threshold)
                inputs.extend([
                    1.0,  # Resource Detected Flag
                    self._normalize_distance(
                        distance, detection_threshold),  # Normalized Distance
                    self._normalize_quantity(
                        resource.quantity),  # Normalized Quantity
                    self._normalize_complexity(
                        resource.complexity)  # Normalized Complexity
                ])
        if not inputs:
            inputs.extend([0.0] * 4)  # No Resource Detected Input

        # Threat detection
        threat_sensitivity = self.genetic_core.heart_genetics.security_sensitivity
        threat_inputs = []
        for threat_pos in env_state.threats:
            distance = self._calculate_distance(threat_pos)
            threat_detection_threshold = 15.0 * threat_sensitivity
            if distance <= threat_detection_threshold:
                threat_inputs.extend([
                    1.0,  # Threat Detected Flag
                    # Normalized Threat Distance
                    self._normalize_distance(
                        distance, threat_detection_threshold)
                ])
        if not threat_inputs:
            threat_inputs.extend([0.0] * 2)  # No Threat Detected Input

        # Agent's Internal State
        internal_inputs = [
            self._normalize_energy(self.energy),  # Normalized Energy Level
        ]
        augmented_inputs = self.augment_perception(torch.tensor(
            inputs + threat_inputs + internal_inputs).float())

        return augmented_inputs.numpy()  # Return numpy array from augmented tensor

    def decide_action(self, env_state: EnvironmentalState) -> Tuple[AgentAction, Dict]:
        """Determine next action using neural network and genetic traits"""
        # Get environmental inputs
        sensor_data = self.perceive_environment(env_state)

        # Genetic Modifiers for Neural Network
        genetic_modifiers = {
            'processing_speed': self.genetic_core.brain_genetics.processing_speed,
            'sensor_sensitivity': self.genetic_core.physical_genetics.sensor_sensitivity
        }

        # Process using neural network, modulated by genetic traits
        network_output, activations = self.neural_net.forward(
            x=sensor_data.reshape(1, -1),  # Reshape for network input
            context=torch.tensor(0)  # Placeholder for context
        )
        network_output = network_output.flatten()  # Flatten to 1D array

        # Decision making influenced by genetic traits
        action_precision = self.genetic_core.physical_genetics.action_precision
        trust_baseline = self.genetic_core.heart_genetics.trust_baseline

        # Action selection logic
        return self._select_action(network_output, action_precision, trust_baseline)

    def execute_action(self, action: AgentAction, params: Dict) -> ActionResult:
        """Execute chosen action with genetic trait influences"""
        energy_efficiency = self.genetic_core.physical_genetics.energy_efficiency
        structural_integrity = self.genetic_core.physical_genetics.structural_integrity

        # Base energy cost modified by efficiency
        energy_cost = self._calculate_energy_cost(action) / energy_efficiency

        if self.energy < energy_cost:
            # Negative reward for failed action
            return ActionResult(False, -1.0, 0.0, None)

        # Action execution logic influenced by genetic traits...
        success_prob = self._calculate_success_probability(
            action, structural_integrity)

        # Execute action and return results...
        action_result = self._process_action_result(
            action, params, energy_cost, success_prob)
        self.energy -= energy_cost  # Deduct energy after processing result
        return action_result

    def learn_from_experience(self, env_state: EnvironmentalState, action: AgentAction, result: ActionResult):
        """Update knowledge and adapt based on action outcomes"""
        learning_efficiency = self.genetic_core.mind_genetics.learning_efficiency
        neural_plasticity = self.genetic_core.mind_genetics.neural_plasticity

        # Prepare training data
        sensor_data = self.perceive_environment(env_state)
        target_output = np.zeros(len(AgentAction))
        action_index = list(AgentAction).index(action)
        target_output[action_index] = result.reward  # Reward as target
        diagnostics = self.neural_diagnostics.monitor_network_health(
            inputs=torch.tensor(sensor_data).float().reshape(1, -1),
            targets=torch.tensor(target_output).float().reshape(1, -1),
            context=torch.tensor(0),
            epoch=env_state.time_step
        )
        # Train Neural Network
        self.neural_net.backward(x=sensor_data.reshape(1, -1),  # Reshape for network input
                                 # Reshape for target output
                                 y=target_output.reshape(1, -1),
                                 activations=None,  # Activations not used in current backprop
                                 learning_rate=learning_efficiency,
                                 plasticity=neural_plasticity
                                 )
        self.data_augmenter.adjust_augmentation(
            network_performance=result.reward,
            diagnostics=diagnostics
        )
        # Update performance metrics
        self._update_metrics(result)

    def get_fitness_score(self) -> float:
        """Calculate comprehensive fitness score including energy management"""
        # Base fitness from previous metrics
        base_fitness = (
            self.total_resources_gathered * 0.3 +
            self.successful_interactions * 0.2 +
            self.survival_time * 0.2 +
            self.efficiency_score * 0.2
        )

        # Energy management component
        energy_ratio = self.energy / 100.0  # Normalized to starting energy
        energy_stability = 0.1 * energy_ratio

        return base_fitness + energy_stability

    def _select_action(self, network_output: np.ndarray,
                       action_precision: float,
                       trust_baseline: float) -> Tuple[AgentAction, Dict]:
        """Detailed action selection logic"""
        # Get action probabilities from network output
        action_probs = network_output

        # Modify probabilities based on genetic traits
        # Higher action precision reduces randomness
        temperature = 1.0 / action_precision
        modified_probs = np.power(action_probs, 1/temperature)
        modified_probs /= modified_probs.sum()

        # Select action based on modified probabilities
        action_idx = np.random.choice(len(AgentAction), p=modified_probs)
        selected_action = list(AgentAction)[action_idx]

        # Generate action parameters based on selected action
        params = self._generate_action_params(selected_action, trust_baseline)

        return selected_action, params

    def _generate_action_params(self, action: AgentAction, trust_baseline: float) -> Dict:
        """Generate specific parameters for each action type with genetic influence"""
        params = {}
        # Get physical parameters for action execution
        genetic_params = self.genetic_core.get_physical_parameters()
        # Get brain parameters for processing
        brain_params = self.genetic_core.get_brain_parameters()

        if action == AgentAction.MOVE:
            # Calculate optimal direction based on resources and threats
            visible_resources = self._get_visible_resources()
            visible_threats = self._get_visible_threats()

            # Weight attractors and repulsors based on genetic traits
            direction_vector = np.zeros(2)

            for resource in visible_resources:
                weight = resource.quantity * \
                    genetic_params['sensor_resolution']
                direction = self._calculate_direction_to(
                    resource.position, env_state)
                direction_vector += direction * weight

            for threat in visible_threats:
                weight = genetic_params['security_sensitivity']
                direction = self._calculate_direction_to(threat, env_state)
                direction_vector -= direction * weight  # Repulsion

            params['direction'] = self._normalize_vector(direction_vector)
            params['speed'] = min(2.0, self.energy / 50.0) * \
                genetic_params['energy_efficiency']

        elif action == AgentAction.GATHER:
            resources = self._get_visible_resources()
            if resources:
                # Score resources based on quantity, distance, and complexity
                scored_resources = []
                for resource in resources:
                    distance = self._calculate_distance(resource.position)
                    gathering_difficulty = resource.complexity / \
                        genetic_params['action_precision']
                    energy_cost = distance * gathering_difficulty

                    expected_value = (resource.quantity *
                                      genetic_params['energy_efficiency'] /
                                      energy_cost)
                    scored_resources.append((expected_value, resource))

                best_resource = max(scored_resources, key=lambda x: x[0])[1]
                params['resource_id'] = best_resource.id
                params['gather_rate'] = genetic_params['action_precision']

        elif action == AgentAction.PROCESS:
            params['resource_type'] = self._select_resource_to_process()
            params['processing_efficiency'] = brain_params['processing_speed']

        elif action == AgentAction.SHARE:
            params['share_amount'] = self.resources[ResourceType.ENERGY] * \
                trust_baseline
            params['target_agent'] = self._select_sharing_target(env_state)

        elif action == AgentAction.DEFEND:
            params['defense_strength'] = self.genetic_core.heart_genetics.security_sensitivity
            params['energy_allocation'] = min(self.energy * 0.3, 30.0)
        elif action == AgentAction.EXECUTE_TOOL:
            params['tool_name'] = 'codebase_search'
            params['tool_params'] = {
                "Query": "self.energy", "TargetDirectories": ['']}
            params['security_level'] = 'LOW'

        return params

    def _process_action_result(self, action: AgentAction, params: Dict, energy_cost: float, success_prob: float) -> ActionResult:
        """Process the outcome of an action"""
        success = False
        reward = 0.0
        new_state = {}

        # Assume action succeeds based on probability
        if random.random() < success_prob:
            success = True

        if action == AgentAction.GATHER:
            reward = self._process_gathering(
                params, success, env_state)  # Pass env_state
        elif action == AgentAction.PROCESS:
            reward = self._process_resources(params, success)
        elif action == AgentAction.SHARE:
            reward = self._process_sharing(params, success)
        elif action == AgentAction.DEFEND:
            reward = self._process_defense(params, success)
        elif action == AgentAction.MOVE:
            reward = self._process_movement(params, success)
        elif action == AgentAction.EXECUTE_TOOL:
            reward = self._process_tool_execution(params, success)

        # Update efficiency score based on reward/energy cost ratio
        if energy_cost > 0:
            # Ensure non-negative reward for efficiency
            self.efficiency_score = (
                self.efficiency_score + max(0, reward)/energy_cost) / 2

        return ActionResult(success, reward, energy_cost, new_state)

    def _calculate_energy_cost(self, action: AgentAction) -> float:
        """Base energy cost for actions - can be adjusted based on action and genetics"""
        base_costs = {
            AgentAction.MOVE: 1.0,
            AgentAction.GATHER: 2.0,
            AgentAction.PROCESS: 5.0,
            AgentAction.SHARE: 1.5,
            AgentAction.DEFEND: 3.0,
            AgentAction.EXECUTE_TOOL: 7.0
        }
        return base_costs.get(action, 1.0)

    def _calculate_success_probability(self, action: AgentAction, structural_integrity: float) -> float:
        """Probability of action success influenced by structural integrity"""
        base_probabilities = {
            AgentAction.MOVE: 0.95,
            AgentAction.GATHER: 0.8,
            AgentAction.PROCESS: 0.7,
            AgentAction.SHARE: 0.99,
            AgentAction.DEFEND: 0.6,
            AgentAction.EXECUTE_TOOL: 0.9
        }
        return base_probabilities.get(action, 0.8) * structural_integrity

    def _update_metrics(self, result: ActionResult):
        """Update agent performance metrics based on action result"""
        if result.success:
            self.successful_interactions += 1
            # Only count positive resource rewards
            self.total_resources_gathered += max(0, result.reward)
        self.survival_time += 1

    # From agent-architecture.py
    def _calculate_distance(self, target_pos: Tuple[int, int]) -> float:
        return np.sqrt(
            (self.position[0] - target_pos[0])**2 +
            (self.position[1] - target_pos[1])**2
        )

    def _normalize_distance(self, distance, max_distance):
        """Normalize distance to a 0-1 range"""
        return 1.0 - min(1.0, distance / max_distance) if max_distance > 0 else 0.0

    def _normalize_quantity(self, quantity):
        """Normalize quantity to a 0-1 range (assuming max quantity is around 100)"""
        return min(1.0, quantity / 100.0)

    def _normalize_complexity(self, complexity):
        """Normalize complexity to a 0-1 range (assuming max complexity is around 2.0)"""
        return min(1.0, complexity / 2.0)

    def _normalize_energy(self, energy):
        """Normalize energy level to a 0-1 range (assuming max energy is 100)"""
        return min(1.0, energy / 100.0)

    def _get_visible_resources(self, env_state: EnvironmentalState) -> List[Resource]:
        """Placeholder: Get list of resources visible to the agent"""
        visible_resources = []
        for resource in env_state.resources:
            distance = self._calculate_distance(resource.position)
            if distance <= 20:  # Visibility range
                visible_resources.append(resource)
        return visible_resources

    def _get_visible_threats(self, env_state: EnvironmentalState) -> List[Tuple[int, int]]:
        """Placeholder: Get list of threats visible to the agent"""
        visible_threats = []
        for threat_pos in env_state.threats:
            distance = self._calculate_distance(threat_pos)
            if distance <= 15:  # Visibility range for threats
                visible_threats.append(threat_pos)
        return visible_threats

    def _calculate_direction_to(self, target_pos: Tuple[int, int], env_state: EnvironmentalState) -> np.ndarray:
        """Placeholder: Calculate direction vector to target position"""
        agent_pos = np.array(self.position)
        target = np.array(target_pos)
        direction = target - agent_pos
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([0, 0])
        return direction / norm

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Placeholder: Normalize a vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _select_resource_to_process(self) -> Optional[ResourceType]:
        """Placeholder: Select which resource to process"""
        if self.resources[ResourceType.MATERIALS] > 0:
            return ResourceType.MATERIALS
        elif self.resources[ResourceType.INFORMATION] > 0:
            return ResourceType.INFORMATION
        elif self.resources[ResourceType.ENERGY] > 0:
            return ResourceType.ENERGY
        return None

    def _select_sharing_target(self, env_state: EnvironmentalState) -> Optional['AdaptiveAgent']:
        """Placeholder: Select target agent for sharing"""
        nearby_agents = [agent for agent in env_state.agents if agent !=
                         self and self._calculate_distance(agent.position) < 10]
        if nearby_agents:
            return random.choice(nearby_agents)
        return None

    def _process_gathering(self, params: Dict, success: bool, env_state: EnvironmentalState) -> float:
        """Placeholder: Process gathering action and return reward"""
        if success and params['resource_id']:
            resource_id = params['resource_id']
            for resource in env_state.resources:
                if resource.id == resource_id:
                    gathered_quantity = min(
                        resource.quantity, params['gather_rate'])
                    self.resources[resource.type] += gathered_quantity
                    resource.quantity -= gathered_quantity
                    return gathered_quantity  # Reward is the quantity gathered
        return -0.1  # Slight negative reward for failed gathering

    def _process_resources(self, params: Dict, success: bool) -> float:
        """Placeholder: Process resources action and return reward"""
        if success and params['resource_type']:
            resource_type = params['resource_type']
            if self.resources[resource_type] > 0:
                processing_rate = params['processing_efficiency']
                processed_quantity = self.resources[resource_type] * \
                    processing_rate
                self.resources[resource_type] -= processed_quantity
                self.energy += processed_quantity * 10  # Convert resources to energy
                return processed_quantity * 5  # Reward for processing
        return -0.5  # Negative reward for failed processing

    def _process_sharing(self, params: Dict, success: bool) -> float:
        """Placeholder: Process sharing action and return reward"""
        if success and params['target_agent']:
            share_amount = params['share_amount']
            target_agent = params['target_agent']
            if self.resources[ResourceType.ENERGY] >= share_amount:
                self.resources[ResourceType.ENERGY] -= share_amount
                target_agent.energy += share_amount
                return share_amount  # Reward is the amount shared
        return -0.2  # Negative reward for failed sharing

    def _process_defense(self, params: Dict, success: bool) -> float:
        """Placeholder: Process defense action and return reward"""
        if success:
            defense_strength = params['defense_strength']
            energy_invested = params['energy_allocation']
            self.energy -= energy_invested
            return defense_strength  # Reward based on defense strength
        return -0.3  # Negative reward for failed defense

    def _process_movement(self, params: Dict, success: bool) -> float:
        """Placeholder: Process movement action and return reward"""
        if success:
            direction = params['direction']
            speed = params['speed']
            new_position = (
                self.position[0] + direction[0] * speed, self.position[1] + direction[1] * speed)
            self.position = new_position
            return 0.01  # Small reward for movement
        return -0.05  # Small negative reward for failed movement

    def _process_tool_execution(self, params: Dict, success: bool) -> float:
        if success and params['tool_name']:
            tool_name = params['tool_name']
            if tool_name == 'codebase_search':
                return 1.0  # Example positive reward for tool use
        return -0.8  # Negative reward for failed tool execution


# From system-improvements.py (Enhanced Environment)
class EnhancedAdaptiveEnvironment(AdaptiveEnvironment):
    def __init__(self, size: Tuple[int, int], complexity: float):
        super().__init__(size, complexity)
        self.terrain = self._generate_terrain()
        self.weather = self._initialize_weather()
        self.agents = []

    def _generate_terrain(self) -> np.ndarray:
        """Generate terrain heightmap using perlin noise"""
        size_x, size_y = self.size
        terrain = np.zeros(self.size)
        scale = 10  # Adjust scale for feature size
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0

        for i in range(octaves):
            frequency = lacunarity ** i
            amplitude = persistence ** i
            x_coords = np.linspace(0, size_x / scale * frequency, size_x)
            y_coords = np.linspace(0, size_y / scale * frequency, size_y)
            xv, yv = np.meshgrid(x_coords, y_coords)
            sample = perlin.perlin(xv, yv)
            terrain += amplitude * sample

        # Normalize terrain to 0-1 range
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        return terrain

    def _update_state(self):
        """Update environment state for next time step"""
        self.current_state.time_step += 1

        # Resource regeneration and movement (example - customize as needed)
        for resource in self.current_state.resources:
            if resource.quantity < 100:
                resource.quantity += random.uniform(0, 0.5)
            resource.position = (
                max(0, min(self.size[0] - 1, int(resource.position[0] + random.uniform(-1, 1))),
                    max(0, min(
                        self.size[1] - 1, int(resource.position[1] + random.uniform(-1, 1))))
                    )
            )
        # Threat movement - more directed movement
        if not hasattr(self.current_state, 'threats'):
            self.current_state.threats = []  # Initialize empty threats list if none exist
        if self.current_state.threats:
            for i, threat_pos in enumerate(self.current_state.threats):
                nearest_agent = self._find_nearest_agent(threat_pos)
                if nearest_agent:
                    # Calculate direction vector towards agent
                    dx = nearest_agent.position[0] - threat_pos[0]
                    dy = nearest_agent.position[1] - threat_pos[1]
                # Normalize direction vector
                    # Avoid division by zero
                    dist = max(0.1, math.sqrt(dx*dx + dy*dy))
                    dx, dy = dx/dist, dy/dist
                # Update threat position
                    new_x = threat_pos[0] + dx
                    new_y = threat_pos[1] + dy
                # Keep threats within bounds
                    self.current_state.threats[i] = (
                        max(0, min(self.size[0]-1, int(new_x))),
                        max(0, min(self.size[1]-1, int(new_y)))
                    )
            else:
                # Random movement if no agents nearby
                self.current_state.threats[i] = (
                    max(0, min(
                        self.size[0]-1, int(threat_pos[0] + random.uniform(-1, 1)))),
                    max(0, min(
                        self.size[1]-1, int(threat_pos[1] + random.uniform(-1, 1))))
                )

        # New resource spawning (example - adjust conditions)
        if random.random() < 0.01 * self.current_state.complexity_level:
            self.current_state.resources.append(
                Resource(
                    type=random.choice(list(ResourceType)),
                    quantity=random.uniform(10, 50),
                    position=(random.randint(
                        0, self.size[0]-1), random.randint(0, self.size[1]-1)),
                    complexity=random.uniform(0.1, 0.9)
                )
            )
        self.current_state.agents = self.agents

    def _calculate_threat_movement(self, threat_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate threat movement direction - simplified random movement"""
        return (random.uniform(-1, 1), random.uniform(-1, 1))

    def _find_nearest_agent(self, pos: Tuple[float, float]) -> Optional['AdaptiveAgent']:
        """Find the nearest agent to a position"""
        min_distance = float('inf')
        nearest_agent = None
        for agent in self.agents:
            distance = self._calculate_distance(pos, agent.position)
            if distance < min_distance:
                min_distance = distance
                nearest_agent = agent
        return nearest_agent

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _generate_perlin_noise(self, size: Tuple[int, int], scale: float) -> np.ndarray:
        """Placeholder: Generate perlin noise heightmap - replace with actual implementation if needed"""
        return np.zeros(size)

    def _initialize_weather(self) -> Dict:
        """Placeholder: Initialize weather conditions - replace with actual implementation if needed"""
        return {}

    def _update_weather(self, current_weather: Dict) -> Dict:
        """Placeholder: Update weather patterns - replace with actual implementation if needed"""
        return current_weather

    def _get_terrain_factor(self, position: Tuple[int, int]) -> float:
        """Placeholder: Get terrain influence factor at position - replace with actual implementation if needed"""
        return 1.0

    def _get_weather_factor(self, position: Tuple[int, int]) -> float:
        """Placeholder: Get weather influence factor at position - replace with actual implementation if needed"""
        return 1.0

    def _calculate_terrain_gradient(self, position: Tuple[int, int]) -> Tuple[float, float]:
        """Placeholder: Calculate terrain gradient at position - replace with actual implementation if needed"""
        return (0.0, 0.0)

    def _find_nearest_agent(self, pos: Tuple[float, float]) -> Optional['AdaptiveAgent']:
        """Placeholder: Find the nearest agent to a position - replace with actual implementation if needed"""
        return None


if __name__ == "__main__":
    genetic_core = create_genetic_core(seed=42)
    embryo_namer = EmbryoNamer()
    generator = EmbryoGenerator(genetic_core, embryo_namer)
    embryo_file = generator.generate_embryo_file()
    size = (100, 100)
    complexity = 1.0
    environment = EnhancedAdaptiveEnvironment(size, complexity)

    input_size = 7
    hidden_sizes = [64, 64]
    output_size = len(AgentAction)
    neural_net = NeuralAdaptiveNetwork(
        input_size, hidden_sizes[0], output_size)

    diagnostics = NeuralDiagnostics(neural_net)

    agent_position = (50, 50)
    agent = AdaptiveAgent(genetic_core, neural_net, agent_position)
    environment.agents.append(agent)
    executor = AdaptiveExecutor(neural_net)

    env_state = environment.current_state
    env_state.resources = [
        Resource(type=ResourceType.ENERGY, quantity=100,
                 position=(20, 20), complexity=0.2),
        Resource(type=ResourceType.MATERIALS, quantity=50,
                 position=(70, 70), complexity=0.8)
    ]
    env_state.threats = [(80, 80)]

    for step in range(2):
        env_state = environment.current_state
        action, params = agent.decide_action(env_state)
        result = agent.execute_action(action, params)
        executor.record_validation_loss(result.reward)
        loss, outputs, importance = executor.execute(
            inputs=torch.tensor(agent.perceive_environment(
                env_state)).float().reshape(1, -1),
            targets=torch.tensor(np.zeros(len(AgentAction))
                                 ).float().reshape(1, -1),
            context=torch.tensor(0),
            diagnostics=diagnostics
        )
        diagnostics = agent.neural_diagnostics.monitor_network_health(
            inputs=torch.tensor(agent.perceive_environment(
                env_state)).float().reshape(1, -1),
            targets=torch.tensor(np.zeros(len(AgentAction))
                                 ).float().reshape(1, -1),
            context=torch.tensor(0),
            epoch=env_state.time_step
        )

        print(json.dumps(diagnostics, indent=2))
        print(
            f"Step: {step+1}, Action: {action}, Reward: {result.reward:.4f}, Energy: {agent.energy:.2f}")
        # Step the environment forward, passing in list of agents
        environment.step([agent])

    print("\nAgent Status after 2 steps:", agent.get_status())
