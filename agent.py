# agent.py
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import uuid
import numpy as np
import math
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from genetics import GeneticCore
from neural_networks import NeuralAdaptiveNetwork
from executor import AdaptiveExecutor
from diagnostics import NeuralDiagnostics
from embryo_namer import EmbryoNamer
from adaptive_environment import AdaptiveEnvironment, ResourceType, Resource, EnvironmentalState
from mind import AgentEmbryo, create_embryo
from dna import DNAGuide, create_dna_guide
from heart import HeartSecurity, create_heart_security
from brainui import create_brain_interface, BrainInterface
from functools import partial



@dataclass
class ActionResult:
    success: bool
    reward: float
    energy_cost: float
    new_state: Optional[Dict]


class ActionVector:
    def __init__(self, selection, parameters, hidden_size=128):
        self.selection_size = 32  # Base action type encoding
        self.parameter_size = 96  # Action parameters encoding

        # Total hidden_size = selection_size + parameter_size
        self.hidden_size = hidden_size

        self.selection = selection
        self.parameters = parameters


class ActionDecoder:
    def __init__(self, hidden_size=32):
        # Dictionary mapping action names to their prototype vectors
        self.action_prototypes = {}
        # The actual functions for each action
        self.action_methods = {}

    def add_action(self, name: str, prototype_vector: torch.Tensor, method: callable):
        self.action_prototypes[name] = prototype_vector
        self.action_methods[name] = method

    def decode_selection(self, selection_vector: torch.Tensor) -> tuple[str, float]:
        # Find closest prototype using cosine similarity
        best_similarity = -1
        selected_action = None

        for name, prototype in self.action_prototypes.items():
            similarity = F.cosine_similarity(
                selection_vector.unsqueeze(0),
                prototype.unsqueeze(0)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                selected_action = name

        return selected_action, best_similarity.item()

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
    
    def adjust_augmentation(self, network_performance: float, diagnostics: Dict):
        pass
    
    
    def get_augmentation_report(self):
        """Generate augmentation report."""
        return self.augmentation_history


class AdaptiveAgent:
    def __init__(self, genetic_core, neural_net, position: Tuple[int, int]):
        self.genetic_core = genetic_core
        self.neural_net = neural_net
        self.position = position

        self.energy = 100.0
        self.resources = {rt: 0.0 for rt in ResourceType}
        self.knowledge_base = {}

        self.total_resources_gathered = 0.0
        self.successful_interactions = 0
        self.survival_time = 0
        self.efficiency_score = 0.0
        self.name = EmbryoNamer().generate_random_name()
        self.data_augmenter = AdaptiveDataAugmenter()
        self.neural_diagnostics = NeuralDiagnostics(neural_net)
        self.action_decoder = ActionDecoder()
        self.actions = {
            "move": self._process_movement,
            "gather": self._process_gathering,
            "process": self._process_resources,
            "share": self._process_sharing,
            "defend": self._process_defense,
            "execute_tool": self._process_tool_execution
        }
        for action, method in self.actions.items():
            vector = torch.randn(32)
            self.action_decoder.add_action(action, vector, method)

        # Initialize internal components
        self.mind = create_embryo()
        self.dna = create_dna_guide()
        self.component_paths = {
            "mind": "mind.py",
            "dna": "dna.py",
            "agent": "agent.py"
        }
        self.heart = create_heart_security(self.component_paths)
        self.brain_interface = create_brain_interface(self.mind, self.dna)

    def augment_perception(self, inputs, context=None):
        return self.data_augmenter.augment(inputs, context)

    def perceive_environment(self, env_state: EnvironmentalState) -> np.ndarray:
        sensor_sensitivity = self.genetic_core.physical_genetics.sensor_sensitivity

        inputs = []

        for resource in env_state.resources:
            distance = self._calculate_distance(resource.position)
            detection_threshold = 10.0 / sensor_sensitivity

            if distance <= detection_threshold:
                clarity = 1.0 - (distance / detection_threshold)
                inputs.extend([
                    1.0,
                    self._normalize_distance(distance, detection_threshold),
                    self._normalize_quantity(resource.quantity),
                    self._normalize_complexity(resource.complexity)
                ])
        if not inputs:
            inputs.extend([0.0] * 4)

        threat_sensitivity = self.genetic_core.heart_genetics.security_sensitivity
        threat_inputs = []
        for threat_pos in env_state.threats:
            distance = self._calculate_distance(threat_pos)
            threat_detection_threshold = 15.0 * threat_sensitivity
            if distance <= threat_detection_threshold:
                threat_inputs.extend([
                    1.0,
                    self._normalize_distance(
                        distance, threat_detection_threshold)
                ])
        if not threat_inputs:
            threat_inputs.extend([0.0] * 2)

        internal_inputs = [
            self._normalize_energy(self.energy),
        ]
        augmented_inputs = self.augment_perception(torch.tensor(
            inputs + threat_inputs + internal_inputs).float())

        return augmented_inputs.numpy()
     
    def get_status(self) -> Dict:
        return {
            "energy": self.energy,
            "resources": self.resources,
            "position": self.position,
            "total_resources_gathered": self.total_resources_gathered,
            "successful_interactions": self.successful_interactions,
            "survival_time": self.survival_time,
            "efficiency_score": self.efficiency_score
        }

    def decide_action(self, env_state: EnvironmentalState) -> Tuple[str, Dict]:
        """Determine next action using neural network and genetic traits"""
        # Get environmental inputs
        sensor_data = self.perceive_environment(env_state)
        sensor_tensor = torch.from_numpy(sensor_data).float().unsqueeze(0)
        # Genetic Modifiers for Neural Network
        genetic_modifiers = {
            'processing_speed': self.genetic_core.brain_genetics.processing_speed,
            'sensor_sensitivity': self.genetic_core.physical_genetics.sensor_sensitivity
        }
        network_output, _ = self.neural_net.forward(
            x=sensor_tensor,
            context=torch.tensor([[0.0]])
        )
        action_vector = ActionVector(
            hidden_size=self.neural_net.output_size, selection=None, parameters=None)
        selection, parameters = action_vector.decode_action(network_output)

        action_precision = self.genetic_core.physical_genetics.action_precision
        trust_baseline = self.genetic_core.heart_genetics.trust_baseline

        return self._select_action(selection, action_precision, trust_baseline, env_state)

    def _select_action(self, network_output: torch.Tensor,
                       action_precision: float,
                       trust_baseline: float, env_state: EnvironmentalState) -> Tuple[str, Dict]:
        action_vector = ActionVector(selection=network_output.detach(
        ).numpy(), parameters=None, hidden_size=self.neural_net.output_size)
        action_selection, confidence = self.action_decoder.decode_selection(
            action_vector.selection)  # Decide which action to take based on its selection vector

        params = self._generate_action_params(
            action_selection, trust_baseline, env_state)
        return action_selection, params

    def execute_action(self, action_key: str, params: Dict, env_state: EnvironmentalState) -> ActionResult:
        """Execute chosen action with genetic trait influences"""
        energy_efficiency = self.genetic_core.physical_genetics.energy_efficiency
        structural_integrity = self.genetic_core.physical_genetics.structural_integrity

        # Base energy cost modified by efficiency
        energy_cost = self._calculate_energy_cost(
            action_key) / energy_efficiency

        if self.energy < energy_cost:
                        return ActionResult(
                success=False,
                reward=-0.5,
                energy_cost=0.0,
                new_state={'error': 'insufficient_energy'}
)

        # Action execution logic influenced by genetic traits...
        success_prob = self._calculate_success_probability(
            action_key, structural_integrity)

        # Execute action and return results...
        action_result = self._process_action_result(
            action_key, params, energy_cost, success_prob, env_state)
        self.energy -= energy_cost  # Deduct energy after processing result
        return action_result

    def learn_from_experience(self, env_state: EnvironmentalState, action: str, result: ActionResult):
        """Update knowledge and adapt based on action outcomes"""
        learning_efficiency = self.genetic_core.mind_genetics.learning_efficiency
        neural_plasticity = self.genetic_core.mind_genetics.neural_plasticity

        # Prepare training data
        sensor_data = self.perceive_environment(env_state)
        target_output = np.zeros(len(self.actions))
        action_index = list(self.actions.keys()).index(action)
        target_output[action_index] = result.reward  # Reward as target
        diagnostics = self.neural_diagnostics.monitor_network_health(
            inputs=torch.tensor(sensor_data).float(
            ).detach().numpy().reshape(1, -1),
            targets=torch.tensor(target_output).float().reshape(1, -1),
            context=torch.tensor([[0.0]]),
            epoch=env_state.time_step
        )
        # Train Neural Network
        self.neural_net.backward(
            x=torch.tensor(sensor_data).float().reshape(
                1, -1),  # Reshape for network input
            y=torch.tensor(target_output).float().reshape(
                1, -1),  # Reshape for target output
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

    def _generate_action_params(self, action: str, trust_baseline: float, env_state: EnvironmentalState) -> Dict:
        """Generate specific parameters for each action type with genetic influence"""
        params = {}
        genetic_params = self.genetic_core.get_physical_parameters()
        brain_params = self.genetic_core.get_brain_parameters()

        if action == "move":
            # Calculate optimal direction based on resources and threats
            visible_resources = self._get_visible_resources(env_state)
            visible_threats = self._get_visible_threats(env_state)

            # Weight attractors and repulsors based on genetic traits
            direction_vector = np.zeros(2)

            for resource in visible_resources:
                weight = resource.quantity * \
                    genetic_params['sensor_resolution']
                direction = self._calculate_direction_to(
                    resource.position, env_state)
                direction_vector += direction * weight
                distance = self._calculate_distance(resource.position)
                attraction = self._normalize_distance(distance, 20.0)
                direction = self._calculate_direction_to(resource.position, env_state)
                direction_vector += direction * attraction * resource.quantity

            for threat in visible_threats:
                weight = genetic_params['security_sensitivity']
                direction = self._calculate_direction_to(threat, env_state)
                direction_vector -= direction * weight  # Repulsion
                distance = self._calculate_distance(threat)
                repulsion = self._normalize_distance(distance, 15.0)
                direction = self._calculate_direction_to(threat, env_state)
                direction_vector -= direction * repulsion * 2.0  # Threats have stronger influence

            params['direction'] = self._normalize_vector(direction_vector)
            params['speed'] = min(2.0, self.energy / 50.0) * \
                genetic_params['energy_efficiency']

        elif action == "gather":
            resources = self._get_visible_resources(env_state)
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
            else:
                params['resource_id'] = None
                params['gather_rate'] = genetic_params['action_precision']

        elif action == "process":
            params['resource_type'] = self._select_resource_to_process()
            params['processing_efficiency'] = brain_params['processing_speed']

        elif action == "share":
            params['share_amount'] = self.resources[ResourceType.ENERGY] * \
                trust_baseline
            params['target_agent'] = self._select_sharing_target(env_state)

        elif action == "defend":
            params['defense_strength'] = self.genetic_core.heart_genetics.security_sensitivity
            params['energy_allocation'] = min(self.energy * 0.3, 30.0)
        elif action == "execute_tool":
            params['tool_name'] = 'codebase_search'
            params['tool_params'] = {
                "Query": "self.energy", "TargetDirectories": ['']}
            params['security_level'] = 'LOW'

        return params

    def _process_action_result(self, action: str, params: Dict, energy_cost: float, success_prob: float, env_state: EnvironmentalState) -> ActionResult:
        """Process action results using genetic traits and neural network"""
        # Get relevant genetic traits
        learning_efficiency = self.genetic_core.mind_genetics.learning_efficiency
        adaptation_rate = self.genetic_core.mind_genetics.adaptation_rate
        neural_plasticity = self.genetic_core.mind_genetics.neural_plasticity
        
        # Create neural network input
        action_encoding = torch.zeros(len(self.actions))
        action_index = list(self.actions.keys()).index(action)
        action_encoding[action_index] = 1.0
        
        genetic_encoding = torch.tensor([
            learning_efficiency,
            adaptation_rate,
            neural_plasticity,
            energy_cost / 10.0,
            success_prob,
            self.energy / 100.0
        ])
        
        network_input = torch.cat([action_encoding, genetic_encoding])
        
        # Get neural network prediction for outcome
        with torch.no_grad():
            outcome_prediction, hidden_state = self.neural_net.forward(
                network_input.unsqueeze(0),
                context=torch.tensor([[adaptation_rate]])
            )
        
        # Rest of the method implementation...
        success_threshold = torch.sigmoid(outcome_prediction[0])
        genetic_modifier = (learning_efficiency + neural_plasticity) / 2.0
        final_success_prob = float(success_prob * success_threshold * genetic_modifier)
        
        success = random.random() < final_success_prob
        
        if success:
            action_method = self.actions.get(action)
            if action_method:
                if action == "gather":
                    reward = action_method(params, success, env_state)
                else:
                    reward = action_method(params, success)
                reward *= (1.0 + learning_efficiency * 0.2)
            else:
                reward = 0.0
        else:
            base_penalty = -0.1 * energy_cost
            penalty_modifier = (1.0 - neural_plasticity * 0.3)
            reward = base_penalty * penalty_modifier
        
        # Update neural network based on outcome
        self._update_network(network_input, success, reward, hidden_state)
        
        new_state = {
            'energy': self.energy - energy_cost,
            'success': success,
            'reward': reward,
            'action': action,
            'genetic_influence': {
                'learning_efficiency': learning_efficiency,
                'adaptation_rate': adaptation_rate,
                'neural_plasticity': neural_plasticity
            }
        }
        
        return ActionResult(
            success=success,
            reward=reward,
            energy_cost=energy_cost,
            new_state=new_state
        )

    def _calculate_energy_cost(self, action: str) -> float:
        """Calculate energy cost based on genetic traits and neural network prediction"""
        # Get relevant genetic traits
        energy_efficiency = self.genetic_core.physical_genetics.energy_efficiency
        metabolic_rate = self.genetic_core.physical_genetics.metabolic_rate
        processing_speed = self.genetic_core.brain_genetics.processing_speed
        
        # Create neural network input vector
        action_encoding = torch.zeros(len(self.actions))
        action_index = list(self.actions.keys()).index(action)
        action_encoding[action_index] = 1.0
        
        genetic_encoding = torch.tensor([
            energy_efficiency,
            metabolic_rate,
            processing_speed,
            self.energy / 100.0  # Current energy level
        ])
        
        network_input = torch.cat([action_encoding, genetic_encoding])
        
        # Get neural network prediction
        with torch.no_grad():
            cost_prediction, _ = self.neural_net.forward(
                network_input.unsqueeze(0),
                context=torch.tensor([[metabolic_rate]])  # Use metabolic rate as context
            )
        
        # Apply genetic modifiers to base cost
        base_cost = torch.sigmoid(cost_prediction[0]) * 10.0
        modified_cost = base_cost * (1.0 / energy_efficiency) * metabolic_rate
        
        return float(max(0.1, modified_cost))

    def _calculate_success_probability(self, action: str, structural_integrity: float) -> float:
        """Calculate success probability using genetic traits and neural prediction"""
        # Get relevant genetic traits
        adaptation_rate = self.genetic_core.mind_genetics.adaptation_rate
        precision = self.genetic_core.physical_genetics.action_precision
        learning_efficiency = self.genetic_core.mind_genetics.learning_efficiency
        
        # Create neural network input
        action_encoding = torch.zeros(len(self.actions))
        action_index = list(self.actions.keys()).index(action)
        action_encoding[action_index] = 1.0
        
        genetic_encoding = torch.tensor([
            structural_integrity,
            precision,
            adaptation_rate,
            learning_efficiency,
            self.successful_interactions / max(1, self.survival_time)
        ])
        
        network_input = torch.cat([action_encoding, genetic_encoding])
        
        # Get neural network prediction
        with torch.no_grad():
            prob_prediction, _ = self.neural_net.forward(
                network_input.unsqueeze(0),
                context=torch.tensor([[adaptation_rate]])
            )
        
        # Apply genetic modifiers to base probability
        base_prob = torch.sigmoid(prob_prediction[0])
        modified_prob = base_prob * precision * (1.0 + learning_efficiency * 0.2)
        
        # Ensure probability stays in valid range
        return float(max(0.1, min(0.95, modified_prob)))

    def _update_metrics(self, result: ActionResult):
        """Update agent performance metrics based on action result"""
        if result.success:
            self.successful_interactions += 1
            self.total_resources_gathered += max(0, result.reward)
        self.survival_time += 1

    def _calculate_distance(self, target_pos: Tuple[int, int]) -> float:
        return np.sqrt(
            (self.position[0] - target_pos[0])**2 +
            (self.position[1] - target_pos[1])**2
        )

    def _normalize_distance(self, distance, max_distance):
        return 1.0 - min(1.0, distance / max_distance) if max_distance > 0 else 0.0

    def _normalize_quantity(self, quantity):
        return min(1.0, quantity / 100.0)

    def _normalize_complexity(self, complexity):
        return min(1.0, complexity / 2.0)

    def _normalize_energy(self, energy):
        return min(1.0, energy / 100.0)

    def _get_visible_resources(self, env_state: EnvironmentalState) -> List[Resource]:
        visible_resources = []
        for resource in env_state.resources:
            distance = self._calculate_distance(resource.position)
            if distance <= 20:
                visible_resources.append(resource)
        return visible_resources

    def _get_visible_threats(self, env_state: EnvironmentalState) -> List[Tuple[int, int]]:
        visible_threats = []
        for threat_pos in env_state.threats:
            distance = self._calculate_distance(threat_pos)
            if distance <= 15:
                visible_threats.append(threat_pos)
        return visible_threats

    def _calculate_direction_to(self, target_pos: Tuple[int, int], env_state: EnvironmentalState) -> np.ndarray:
        agent_pos = np.array(self.position)
        target = np.array(target_pos)
        direction = target - agent_pos
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([0, 0])
        return direction / norm

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _select_resource_to_process(self) -> Optional[ResourceType]:
        if self.resources[ResourceType.MATERIALS] > 0:
            return ResourceType.MATERIALS
        elif self.resources[ResourceType.INFORMATION] > 0:
            return ResourceType.INFORMATION
        elif self.resources[ResourceType.ENERGY] > 0:
            return ResourceType.ENERGY
        return None

    def _select_sharing_target(self, env_state: EnvironmentalState) -> Optional['AdaptiveAgent']:
        nearby_agents = [agent for agent in env_state.agents if agent !=
                         self and self._calculate_distance(agent.position) < 10]
        if nearby_agents:
            return random.choice(nearby_agents)
        return None

    def learn_action(self, action_name: str, action_function):
        self.actions[action_name] = action_function

    def _process_gathering(self, params: Dict, success: bool, env_state: EnvironmentalState) -> float:
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

    def _process_resources(self, params: Dict, success: bool, env_state: EnvironmentalState) -> float:
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

    def _process_sharing(self, params: Dict, success: bool, env_state: EnvironmentalState) -> float:
        if success and params['target_agent']:
            share_amount = params['share_amount']
            target_agent = params['target_agent']
            if self.resources[ResourceType.ENERGY] >= share_amount:
                self.resources[ResourceType.ENERGY] -= share_amount
                target_agent.energy += share_amount
                return share_amount  # Reward is the amount shared
        return -0.2  # Negative reward for failed sharing

    def _process_defense(self, params: Dict, success: bool, env_state: EnvironmentalState) -> float:
        if success:
            defense_strength = params['defense_strength']
            energy_invested = params['energy_allocation']
            self.energy -= energy_invested
            return defense_strength  # Reward based on defense strength
        return -0.3  # Negative reward for failed defense

    def _process_movement(self, params: Dict, success: bool, env_state: EnvironmentalState) -> float:
        if success:
            direction = params['direction']
            speed = params['speed']
            new_position = (
                self.position[0] + direction[0] * speed, self.position[1] + direction[1] * speed)
            self.position = new_position
            return 0.01  # Small reward for movement
        return -0.05  # Small negative reward for failed movement

    def _process_tool_execution(self, params: Dict, success: bool, env_state: EnvironmentalState) -> float:
        if success and params['tool_name']:
            tool_name = params['tool_name']
            if tool_name == 'codebase_search':
                return 1.0  # Example positive reward for tool use
        return -0.8  # Negative reward for failed tool execution

    def _calculate_distance(self, target_pos: Tuple[int, int]) -> float:
        return np.sqrt(
            (self.position[0] - target_pos[0])**2 +
            (self.position[1] - target_pos[1])**2
        )
    
    def create_action(self, need_type: str, env_state: EnvironmentalState) -> Optional[Tuple[str, callable]]:
        # Get relevant genetic traits for action creation
        creativity = self.genetic_core.mind_genetics.creativity
        adaptability = self.genetic_core.mind_genetics.adaptation_rate
        intelligence = self.genetic_core.brain_genetics.processing_speed
        
        # Calculate creation probability based on genetic traits
        creation_threshold = (creativity + adaptability + intelligence) / 3.0
        
        if random.random() > creation_threshold:
            return None  # Failed to create new action
        
        # Generate action components based on need type
        components = {
            'energy': ['gather', 'convert', 'optimize'],
            'defense': ['evade', 'shield', 'counter'],
            'social': ['communicate', 'collaborate', 'trade'],
            'exploration': ['scan', 'analyze', 'map']
        }
        
        # Create neural network input for action design
        design_input = torch.tensor([
            creativity,
            adaptability,
            intelligence,
            self.energy / 100.0,
            len(self.actions) / 10.0  # Normalized current action count
        ]).float()
        
        # Get network prediction for action design
        with torch.no_grad():
            design_output, _ = self.neural_net.forward(
                design_input.unsqueeze(0),
                context=torch.tensor([[creativity]])
            )
        
        # Generate action name and basic components
        base_components = components.get(need_type, ['generic'])
        action_name = f"{random.choice(base_components)}_{len(self.actions)}"
        
        # Store traits in a closure for the new action method
        traits = {
            'creativity': creativity,
            'intelligence': intelligence,
            'adaptability': adaptability
        }
        
        # Define the new action method
        def new_action_method(self, params: Dict, success: bool, env_state: EnvironmentalState = None) -> float:
            if not success:
                return -0.1
                
            base_reward = 0.2 * traits['creativity']  # Higher creativity = higher potential reward
            efficiency_bonus = 0.1 * traits['intelligence']  # Higher intelligence = better execution
            adaptation_modifier = 0.1 * traits['adaptability']  # Higher adaptability = better environmental fit
            
            # Calculate total reward based on genetic traits and current state
            total_reward = (base_reward + efficiency_bonus) * (1.0 + adaptation_modifier)
            
            # Update agent state based on action execution
            self.energy -= 0.1 * (1.0 - traits['creativity'])  # More creative actions use less energy
            self.efficiency_score += 0.01 * traits['intelligence']
            
            return float(total_reward)
        
        # Create action vector for the new action
        action_vector = torch.randn(32)  # Random initial action vector
        self.action_decoder.add_action(action_name, action_vector, new_action_method)
        
        # Add to available actions
        self.actions[action_name] = new_action_method
        
        # Log action creation
        print(f"Created new action: {action_name} for need: {need_type}")
        
        return action_name, new_action_method
    
    def decide_action(self, env_state: EnvironmentalState) -> Tuple[str, Dict]:
        if self.energy < 30.0:  # Low energy state
            new_action = self.create_action('energy', env_state)
            if new_action:
                action_name, _ = new_action
                params = self._generate_action_params(action_name, trust_baseline, env_state)
        return action_name, params
            