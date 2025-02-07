import sys
import pygame
import random
import math
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import numpy as np
import time
# from agent import AdaptiveAgent # moved inside imports
# from genetics import GeneticCore # moved inside imports
# from neural_networks import NeuralAdaptiveNetwork # moved inside imports
# from adaptive_environment import AdaptiveEnvironment, ResourceType, Resource # moved inside imports
# from visualizer import Visualizer # moved inside imports
import matplotlib.pyplot as plt
# from agent_assembly import AgentAssembler # moved inside imports
# from predator import Predator # moved inside imports
# from birth_registry import BirthRegistry # moved inside imports
# from embryo_generator import EmbryoGenerator, EmbryoToAgentDevelopment # moved inside imports
import json
# from augmentation import NeuralDiagnostics # moved inside imports
# from diagnostics import AgentDiagnostics, PopulationDiagnostics # moved inside imports
# from heart import HeartSystem # moved inside imports
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimulationDebugger:
    def __init__(self):
        self.frame_count = 0
        self.last_agent_count = 0
        self.error_states = []

    def monitor_frame(self, env, agents):
        """Monitor each simulation frame for issues"""
        self.frame_count += 1

        # Check population changes
        if len(agents) != self.last_agent_count:
            logger.info(f"Population changed: {self.last_agent_count} -> {len(agents)}")
            self.last_agent_count = len(agents)

        # Monitor agent states
        for agent in agents:
            if agent.energy < 0:
                logger.warning(f"Agent {agent.name} has negative energy: {agent.energy}")
            if agent.age > agent.max_age:
                logger.warning(f"Agent {agent.name} exceeded max age: {agent.age}/{agent.max_age}")

        # Monitor environment state
        if len(env.current_state.resources) == 0:
            logger.warning("No resources in environment")

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = (800, 600)
DETAIL_PANEL_WIDTH = 200
CELL_SIZE = 10
STATS_HISTORY_LENGTH = 1000

class SimulationStats:
    def __init__(self):
        """Initialize simulation statistics tracking"""
        self.start_time = datetime.now()
        self.population_history = deque(maxlen=STATS_HISTORY_LENGTH)
        self.innovation_history = deque(maxlen=STATS_HISTORY_LENGTH)
        self.generation_stats = {}
        self.evolution_metrics = {
            'mutations': [],
            'novel_actions': [],
            'adaptation_rates': []
        }
        self.performance_metrics = {
            'fps_history': deque(maxlen=60),
            'memory_usage': [],
            'processing_time': []
        }

        # Create stats directory if it doesn't exist
        if not os.path.exists('simulation_stats'):
            os.makedirs('simulation_stats')

    def update(self, agents, time_step):
        """Update simulation statistics"""
        self.population_history.append(len(agents))
        total_innovations = sum(len(a.actions) - 5 for a in agents)  # -5 for initial actions
        self.innovation_history.append(total_innovations)

    def save_graphs(self):
        """Save graphs of simulation statistics"""
        # Population history graph
        plt.figure(figsize=(10, 5))
        plt.plot(list(self.population_history))
        plt.title('Population History')
        plt.xlabel('Time Step')
        plt.ylabel('Population Size')
        plt.savefig('simulation_stats/population_history.png')
        plt.close()

        # Innovation history graph
        plt.figure(figsize=(10, 5))
        plt.plot(list(self.innovation_history))
        plt.title('Innovation History')
        plt.xlabel('Time Step')
        plt.ylabel('Total Innovations')
        plt.savefig('simulation_stats/innovation_history.png')
        plt.close()

class SimulationVisualizer: # Simplified
    def __init__(self, width: int = 800, height: int = 600, debug_mode: bool = False):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Adaptive Agent Simulation")
        self.clock = pygame.time.Clock()

    def draw(self, env, agents):
        self.screen.fill((0, 0, 0))  # Black background

        # Example: Draw agents as green circles
        for agent in agents:
            pygame.draw.circle(self.screen, (0, 255, 0),
                               (int(agent.position[0] * 10), int(agent.position[1] * 10)), 5)  # Scale position

        pygame.display.flip()

    def add_action_creation_effect(self, position):
        pass
    def draw_agent_evolution(self, agent):
        pass
    def _draw_detail_panel(self, env, agents):
        pass
    def _draw_population_graph(self):
        pass
    def _draw_innovation_graph(self):
        pass
    def _draw_evolution_events(self):
        pass
    def add_evolution_event(self, text, event_type='evolution'):
        pass
    def _draw_achievement(self, achievement):
        pass
    def draw_agent_activities(self, agent):
        pass
    def take_screenshot(self, event_name: str = "") -> str:
        pass
    def capture_important_event(self, event_name: str, event_data: dict = None):
        pass

def save_best_agents(agents):
   pass
def calculate_network_architecture(genetic_core: 'GeneticCore') -> Dict[str, int]:
    """Calculate neural network architecture based on genetic traits"""
    # Simplified for demonstration
    return {
        'input_size': 16,
        'output_size': 8,
        'memory_size': 32
    }

class PopulationGenetics:
    def __init__(self):
        self.genetic_diversity = {}
        self.dominant_traits = []

    def track_population_genetics(self, agents):
        pass

    def calculate_mutation_rate(self, population_size: int, genetic_diversity: float) -> float:
        return 0.01

def create_predator(env_size: Tuple[int, int]) -> 'Predator':
    """Create a new predator with its own neural network"""
    pass

# --- Simplified GeneticCore (for demonstration) ---
class GeneticCore:  # You would have a more complete class, just to remove the error
    def __init__(self):
        class Genetics:
            def __init__(self):
                self.creativity = 1.0
                self.adaptation_rate = 1.0
                self.processing_speed = 1.0
                self.energy_efficiency = 1.0
                self.complexity_threshold = 1.0
                self.sensor_sensitivity = 1.0
                self.metabolic_rate = 1.0
                self.vitality = 1.0
                self.trust_baseline = 1.0
                self.empathy_level = 1.0
                self.action_precision = 1.0
                self.structural_integrity = 1.0
                self.gathering_efficiency = 1.0
                self.learning_efficiency = 1.0
                self.neural_plasticity = 1.0
                self.security_sensitivity = 1.0
                self.reproduction_rate = 1.0
                self.longevity = 1.0
                self.risk_tolerance = 1.0

        self.mind_genetics = Genetics()
        self.brain_genetics = Genetics()
        self.physical_genetics = Genetics()
        self.heart_genetics = Genetics()
        self.emergent_traits = {}  # just to remove the error

    def get_physical_parameters(self):
        return {
            'sensor_resolution': self.physical_genetics.sensor_sensitivity,
            'security_sensitivity': self.heart_genetics.security_sensitivity,  # Assuming heart_genetics
            'energy_efficiency': self.physical_genetics.energy_efficiency
        }

    def get_brain_parameters(self):
        return {
            'processing_speed': self.brain_genetics.processing_speed
        }

    def create_offspring(self):
        return GeneticCore()

    def apply_emergent_traits(self):
        pass

    def _generate_emergent_trait(self):  # Added missing method
        return None
    def get_all_traits(self): # Added so the population tracker doesnt fail
        return []

def create_genetic_core():
    """Creates a GeneticCore object (replace with your actual logic)."""
    return GeneticCore()

class AdaptiveEnvironment: # Simplified
    def __init__(self, size, complexity=0.5):
        self.size = size
        self.complexity = complexity
        self.current_state = self.State(time_step=0, resources=[], threats=[])
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def get_state(self):
        return self.current_state
    def add_resources(self, num):
        pass

    def update(self):
        # Simple update: increment time step
        self.current_state.time_step += 1
    def step(self, agent):
        pass

    class State:
        def __init__(self, time_step, resources, threats, agents = None):
            self.time_step = time_step
            self.resources = resources
            self.threats = threats
            self.agents = agents or []

        def get_context_vector(self):
            return torch.tensor([0.0])

class NeuralAdaptiveNetwork:  # Simplified
    def __init__(self, input_size, output_size, hidden_size = None, memory_size = None, learning_rate = None, genetic_core = None, plasticity = None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size # Added missing
        self.memory_size = memory_size
        self.learning_rate = learning_rate or 0.01  # Default value
        self.genetic_core = genetic_core
        self.plasticity = plasticity

    def forward(self, x, context=None):
        # Placeholder forward pass (no actual computation)
        return torch.randn(1, self.output_size), torch.randn(1, self.hidden_size)

    def backward(self, x, y, activations=None, learning_rate=None, plasticity=None):
        pass
    def adapt_network(self):
        pass
    def inherit_weights(self, parent, mutation_rate, adaptation_rate):
        pass
    def process_dream(self, experience):
        pass
    def reset_states(self):
        pass

import torch.nn.functional as F
class NeuralBehavior: # Simplified
    def __init__(self, neural_net, genetic_core):
        self.neural_net = neural_net
        self.genetic_core = genetic_core
    def process_perception(self, env_state):
        return torch.randn(1, self.neural_net.input_size)
    def decide_action(self, state):
        return 'move', 0.5
    def learn_from_experience(self, state, action, result):
        pass

from dataclasses import dataclass
@dataclass
class ActionResult:
    success: bool
    reward: float
    energy_cost: float
    new_state: Optional[Dict]

@dataclass
class Lineage:
    generation: int
    parent_id: Optional[str]
    birth_time: float
    genetic_heritage: List[str]  # List of notable ancestor IDs
    mutations: List[Dict]  # Track significant genetic mutations
    achievements: List[Dict]  # Track notable achievements

class ActionVector:
    def __init__(self, selection, parameters, hidden_size=128):
        pass
    def decode_action(self, network_output: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        pass
class ActionDecoder:
    def __init__(self):
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
    def get_action_method(self, action_name: str) -> Optional[callable]:
        return self.action_methods.get(action_name)

class AdaptiveDataAugmenter:
    def __init__(self,
                 noise_levels=[0.1, 0.3, 0.5, 0.7],
                 augmentation_strategies=None):
        pass
    def augment(self, data, context=None):
        return data
    def adjust_augmentation(self, network_performance, diagnostics=None):
        pass
    def get_augmentation_report(self):
        pass
    def _apply_evolutionary_augmentation(self, data: torch.Tensor, genetic_traits: Dict) -> torch.Tensor:
        pass
    def get_augmentation_stats(self) -> Dict:
        pass
    def _compute_adjustment(self, network_performance, diagnostics):  # Added missing method
        return network_performance

class ActionMutation:
    """Handles mutation of action behaviors and parameters"""
    def __init__(self, genetic_core: 'GeneticCore'):
        pass
    def mutate_action(self, action_func: callable, mutation_strength: float) -> callable:
        pass

class NeuralDiagnostics:  # Simplified
    def monitor_network_health(self, inputs, targets, context, epoch):
        return {}  # Return an empty dictionary

class AdaptiveAgent:  # Simplified
    def __init__(self, birth_id, genetic_core, neural_net, position, environment, **kwargs):
        self.birth_id = birth_id
        self.genetic_core = genetic_core
        self.neural_net = neural_net
        self.position = position
        self.environment = environment
        self.energy = 100.0
        self.age = 0
        self.max_age = 1000
        self.name = "UnnamedAgent" # Added to fix errors
        self.is_creating_action = False
        self.actions = {} # Added to simplify
        self.lineage = Lineage(generation = 0, parent_id = None, birth_time = 0.0, genetic_heritage = [], mutations = [], achievements = [])
        self.neural_diagnostics = NeuralDiagnostics()
        self.data_augmenter = AdaptiveDataAugmenter()
        self.evolution_stats = {}
        self.neural_behavior = NeuralBehavior(neural_net, genetic_core) # Added missing
        self.action_decoder = ActionDecoder()
        self.action_mutator = ActionMutation(genetic_core)
        self._create_initial_action()
        self.mutation_stats = {}


    def _create_initial_action(self):
        """Create the most basic survival action - random movement"""
        def basic_movement(params: Dict, success: bool, env_state) -> float:
            direction = np.random.rand(2) - 0.5  # Random direction
            self.position = (
                self.position[0] + direction[0],
                self.position[1] + direction[1]
            )
            return 0.01 if success else -0.01

        self.learn_action("basic_move", basic_movement)
    def learn_action(self, action_name: str, action_function):
        self.actions[action_name] = action_function
    def update(self, env_state):
        # Simple update: increment age, decrease energy
        self.age += 1
        self.energy -= 0.1
        return self.energy > 0 and self.age < self.max_age
    def is_alive(self):
        return self.energy > 0 and self.age < self.max_age
    def needs_rest(self):
        return False
    def process_rest_cycle(self):
        pass
    def can_reproduce(self):
        return self.energy > 50 and self.age > 20
    def reproduce(self, env_state):
        # Simplified reproduction: create a copy with slight variations
        birth_id = str(uuid.uuid4())
        new_position = (self.position[0] + random.uniform(-2, 2), self.position[1] + random.uniform(-2, 2))

        offspring = AdaptiveAgent(
            birth_id=birth_id,
            genetic_core=self.genetic_core.create_offspring(),  # Create offspring genetic core
            neural_net=self.neural_net,  #  simplified
            position=new_position,
            environment=self.environment
        )

        self.energy -= 30 # reproduction cost
        return offspring
    def calculate_fitness(self) -> float:
        pass
    def decide_action(self, env_state) -> Tuple[str, Dict]:
        # Simplified action selection
        return "basic_move", {}

    def execute_action(self, action_key: str, params: Dict, env_state) -> ActionResult:
        # Simplified action execution
        if action_key == "basic_move":  # Use the string name
            result = self.actions[action_key](params, success = True, env_state = env_state) # Execute
            return ActionResult(success=True, reward=result, energy_cost=0.1, new_state=None)

        return ActionResult(success=False, reward=-0.1, energy_cost=0.1, new_state=None)

    @staticmethod
    def _calculate_network_architecture(genetic_core: GeneticCore) -> Dict[str, int]:
        """Calculate neural network architecture based on genetic traits"""
        # Simplified for demonstration
        return {
            'input_size': 16,
            'output_size': 8,
            'memory_size': 32
        }
    def _calculate_distance(self, target_pos: Tuple[int, int]) -> float:
        return np.sqrt(
            (self.position[0] - target_pos[0])**2 +
            (self.position[1] - target_pos[1])**2
        )
    def perceive_environment(self, env_state):
        pass
    def process_environment(self, env_state):
        return np.zeros(16) # Dummy value

def run_simulation():

    # Initialize components
    logger.info("Starting simulation...")
    env = AdaptiveEnvironment(size=(80, 60))
    visualizer = SimulationVisualizer()
    debugger = SimulationDebugger()
    population_genetics = PopulationGenetics()

    # Ensure minimum resources in environment
    if len(env.current_state.resources) == 0:
        logger.info("Initializing environment with minimum resources...")
        env.add_resources(5)

    # Create initial agents
    agents = []
    for i in range(5):
        try:
            genetic_core = create_genetic_core()

            # Calculate network architecture based on genetics
            network_params = calculate_network_architecture(genetic_core)

            # Create neural network with genetic-based architecture
            neural_net = NeuralAdaptiveNetwork(
                input_size=network_params['input_size'],
                hidden_size=network_params['memory_size'],
                output_size=network_params['output_size'],
                genetic_core=genetic_core  # Added missing parameter
            )

            position = (
                random.randint(0, env.size[0]-1),
                random.randint(0, env.size[1]-1)
            )

            # --- FIX: Provide birth_id and environment ---
            birth_id = str(uuid.uuid4())  # Generate a unique ID
            agent = AdaptiveAgent(
                birth_id=birth_id,  # Pass the birth_id
                genetic_core=genetic_core,
                neural_net=neural_net,
                position=position,
                environment=env  # Pass the environment
            )
            agents.append(agent)
            env.add_agent(agent) # Add to environment
            logger.info(f"Created agent {i+1} with network architecture: {network_params}")

        except Exception as e:
            logger.error(f"Failed to create agent {i+1}: {str(e)}")
            continue

    if not agents:
        logger.error("Failed to create any agents. Exiting simulation.")
        return

    running = True
    paused = False

    # Main simulation loop
    while running:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused  # Toggle pause state
                    elif event.key == pygame.K_ESCAPE:
                        running = False  # Exit simulation

            if not paused:
                # Update environment
                env.update()  # Call the simplified update method
                population_genetics.track_population_genetics(agents)

                # Process each agent
                new_agents = []
                for agent in agents[:]:  # Create a copy of list for modification
                    try:
                        # Update agent
                        if not agent.update(env.current_state):
                            logger.info(f"Agent {agent.name} died")
                            agents.remove(agent)
                            continue

                        # Process agent actions
                        action, params = agent.decide_action(env.current_state)
                        result = agent.execute_action(action, params, env.current_state)

                        # Check reproduction using agent's internal logic
                        if agent.can_reproduce():
                            offspring = agent.reproduce(env.current_state)
                            if offspring:
                                new_agents.append(offspring)
                                env.add_agent(offspring) # Add to environment
                                #if hasattr(visualizer, 'add_evolution_event'):  # Check for method
                                #   visualizer.add_evolution_event(
                                #        f"Agent {agent.name[:8]} reproduced (fitness: {agent.calculate_fitness():.2f})",
                                #        'evolution'
                                #    )

                    except Exception as e:
                        logger.error(f"Error processing agent {agent.name}: {str(e)}")
                        continue

                # Add new agents
                agents.extend(new_agents)

                # Update visualization
                visualizer.draw(env, agents)

                # Update simulation monitoring
                debugger.monitor_frame(env, agents)

                # Save stats periodically
                # if debugger.frame_count % 1000 == 0: # Removed to simplify
                #    visualizer.stats.save_graphs()

            pygame.display.flip()  # Update the display
            pygame.time.Clock().tick(30) # Limit frame rate


        except Exception as e:
            logger.error(f"Error in simulation loop: {str(e)}")
            continue

    # Cleanup
    #visualizer.stats.save_graphs() # Removed to simplify
    pygame.quit()
    logger.info("Simulation ended normally")


if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}")
        sys.exit(1)