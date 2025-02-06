import logging
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
from agent import AdaptiveAgent
from genetics import GeneticCore
from neural_networks import NeuralAdaptiveNetwork
from adaptive_environment import AdaptiveEnvironment, ResourceType, Resource
from visualizer import Visualizer
import matplotlib.pyplot as plt
from predator import AdaptivePredator
from agent_assembly import AgentAssembler
from birth_registry import BirthRegistry
from embryo_generator import EmbryoGenerator

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

class SimulationVisualizer:
    def __init__(self, width: int = 800, height: int = 600, debug_mode: bool = False):
        """Initialize the simulation visualizer"""
        pygame.init()
        
        # Display setup
        self.WINDOW_SIZE = (width, height)
        self.DETAIL_PANEL_WIDTH = 250
        self.CELL_SIZE = 10
        self.screen = pygame.display.set_mode(
            (self.WINDOW_SIZE[0] + self.DETAIL_PANEL_WIDTH, self.WINDOW_SIZE[1])
        )
        pygame.display.set_caption("Adaptive Agent Simulation")
        
        # Font initialization
        self.font = pygame.font.Font(None, 24)
        self.detail_font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 28)
        
        # Performance tracking
        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.last_update = time.time()
        
        # Visual effects
        self.action_creation_effects = []
        self.evolution_events = deque(maxlen=10)
        self.achievements = []
        
        # Color scheme
        self.COLORS = {
            'background': (0, 0, 0),
            'agent': (0, 255, 0),
            'predator': (255, 0, 0),
            'resource': (255, 255, 0),
            'text': (255, 255, 255),
            'stats_background': (30, 30, 30),
            'graph_line': (0, 255, 0),
            'evolution_potential': (0, 255, 255),
            'evolved_action': (255, 165, 0),
            'mutation': (255, 0, 255),
            'adaptation': (0, 255, 255)
        }
        
        # Statistics tracking
        self.stats = SimulationStats()
        
        # Debug information
        self.debug_info = {
            'fps': 0,
            'active_agents': 0,
            'active_resources': 0,
            'evolution_events': 0
        }
        
        self.debug_mode = debug_mode
        self.screenshot_dir = "simulation_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        logger.info("SimulationVisualizer initialized successfully")

    def draw(self, env, agents):
        self.screen.fill(self.COLORS['background'])
        
        # Draw resources
        for resource in env.current_state.resources:
            pygame.draw.circle(
                self.screen,
                self.COLORS['resource'],
                (int(resource.position[0] * CELL_SIZE), 
                 int(resource.position[1] * CELL_SIZE)),
                int(np.sqrt(resource.quantity))
            )
        
        # Draw threats
        for threat in env.current_state.threats:
            pygame.draw.circle(
                self.screen,
                self.COLORS['threat'],
                (int(threat[0] * CELL_SIZE), 
                 int(threat[1] * CELL_SIZE)),
                5
            )
        
        # Draw agents with genetic distinction
        for agent in agents:
            # Determine agent color based on innovation potential
            innovation_potential = (
                agent.genetic_core.mind_genetics.creativity * 0.4 +
                agent.genetic_core.brain_genetics.processing_speed * 0.3 +
                agent.genetic_core.mind_genetics.adaptation_rate * 0.2 +
                agent.genetic_core.physical_genetics.energy_efficiency * 0.1
            )
            
            # Use special color for exceptional agents
            agent_color = (
                self.COLORS['exceptional_agent'] 
                if innovation_potential > 0.85 
                else self.COLORS['agent']
            )
            
            # Draw agent
            pygame.draw.circle(
                self.screen,
                agent_color,
                (int(agent.position[0] * CELL_SIZE), 
                 int(agent.position[1] * CELL_SIZE)),
                7
            )
            
            # Draw lineage indicator for agents with notable heritage
            if len(agent.lineage.genetic_heritage) > 0:
                pygame.draw.circle(
                    self.screen,
                    self.COLORS['lineage_marker'],
                    (int(agent.position[0] * CELL_SIZE), 
                     int(agent.position[1] * CELL_SIZE)),
                    9,
                    1  # Width of circle outline
                )
            
            # Draw agent stats
            stats_text = self.font.render(
                f"E:{int(agent.energy)} G:{agent.lineage.generation}", 
                True, self.COLORS['text']
            )
            self.screen.blit(
                stats_text, 
                (int(agent.position[0] * CELL_SIZE) - 20, 
                 int(agent.position[1] * CELL_SIZE) - 20)
            )
            
            # Draw agent activities
            self.draw_agent_activities(agent)
            
            # Draw agent evolution
            self.draw_agent_evolution(agent)
        
        # Draw action creation effects
        for effect in self.action_creation_effects[:]:
            pos, time_left = effect
            if time_left > 0:
                # Draw expanding circle effect
                radius = (30 - time_left) * 2
                pygame.draw.circle(
                    self.screen,
                    self.COLORS['action_creation'],
                    (int(pos[0] * CELL_SIZE), int(pos[1] * CELL_SIZE)),
                    radius,
                    2  # Width of circle
                )
                effect[1] -= 1
            else:
                self.action_creation_effects.remove(effect)
        
        # Draw global stats with more info
        stats_text = self.font.render(
            f"Time: {env.current_state.time_step} | Agents: {len(agents)} | " +
            f"Innovations: {sum(len(a.actions) - 5 for a in agents)}", # -5 for initial actions
            True, self.COLORS['text']
        )
        self.screen.blit(stats_text, (10, 10))
        
        # Draw detailed stats panel
        self._draw_detail_panel(env, agents)
        
        # Draw mini-graphs
        self._draw_population_graph()
        self._draw_innovation_graph()
        
        # Draw recent evolution events
        self._draw_evolution_events()
        
        # Update stats
        self.stats.update(agents, env.current_state.time_step)
        
        pygame.display.flip()
        self.clock.tick(30)

    def add_action_creation_effect(self, position):
        """Add visual effect for action creation"""
        self.action_creation_effects.append([position, 30])  # 30 frames duration

    def draw_agent_evolution(self, agent):
        """Draw agent's evolutionary status"""
        evolution_potential = (
            agent.genetic_core.mind_genetics.creativity * 0.4 +
            agent.genetic_core.brain_genetics.processing_speed * 0.3 +
            agent.genetic_core.mind_genetics.adaptation_rate * 0.2 +
            agent.genetic_core.physical_genetics.energy_efficiency * 0.1
        )
        
        # Add evolution potential to stats
        if not hasattr(agent.evolution_stats, 'evolution_potential'):
            agent.evolution_stats['evolution_potential'] = evolution_potential
        
        # Draw evolution potential indicator
        potential_height = 4
        potential_width = 20
        potential_rect = pygame.Rect(
            int(agent.position[0] * CELL_SIZE) - potential_width//2,
            int(agent.position[1] * CELL_SIZE) - 15,
            int(potential_width * evolution_potential),
            potential_height
        )
        pygame.draw.rect(self.screen, self.COLORS['evolution_potential'], potential_rect)
        
        # Show number of evolved actions
        evolved_count = len([a for a in agent.actions.keys() if 'evolved' in a])
        if evolved_count > 0:
            evolved_text = self.detail_font.render(
                f"E:{evolved_count}", True, self.COLORS['evolved_action']
            )
            self.screen.blit(
                evolved_text,
                (int(agent.position[0] * CELL_SIZE) + 15,
                 int(agent.position[1] * CELL_SIZE) - 20)
            )
        
        # Show recent mutations
        if agent.lineage.mutations:
            recent_mutations = len(agent.lineage.mutations[-3:])  # Show last 3
            mutation_text = self.detail_font.render(
                f"M:{recent_mutations}", True, self.COLORS['mutation']
            )
            self.screen.blit(
                mutation_text,
                (int(agent.position[0] * CELL_SIZE) - 30,
                 int(agent.position[1] * CELL_SIZE) - 20)
            )

    def _draw_detail_panel(self, env, agents):
        panel_rect = pygame.Rect(WINDOW_SIZE[0], 0, DETAIL_PANEL_WIDTH, WINDOW_SIZE[1])
        pygame.draw.rect(self.screen, self.COLORS['stats_background'], panel_rect)
        
        y_offset = 10
        
        # Simulation title
        title = self.title_font.render("Simulation Details", True, self.COLORS['text'])
        self.screen.blit(title, (WINDOW_SIZE[0] + 10, y_offset))
        y_offset += 40

        # Time stats
        runtime = datetime.now() - self.stats.start_time
        time_stats = [
            f"Runtime: {runtime.seconds//3600:02d}:{(runtime.seconds//60)%60:02d}:{runtime.seconds%60:02d}",
            f"Time Step: {env.current_state.time_step}",
            f"Population: {len(agents)}",
        ]

        # Add generation and innovation stats only if there are agents
        if agents:
            time_stats.extend([
                f"Max Generation: {max(a.lineage.generation for a in agents)}",
                f"Total Innovations: {sum(len(a.actions) - 5 for a in agents)}"
            ])
        else:
            time_stats.extend([
                "Max Generation: N/A",
                "Total Innovations: N/A"
            ])

        for stat in time_stats:
            text = self.detail_font.render(stat, True, self.COLORS['text'])
            self.screen.blit(text, (WINDOW_SIZE[0] + 10, y_offset))
            y_offset += 20

        # Best Agent Stats
        if agents:
            y_offset += 20
            best_agent = max(agents, key=lambda a: a.energy)
            agent_stats = [
                "Best Agent:",
                f"Generation: {best_agent.lineage.generation}",
                f"Energy: {int(best_agent.energy)}",
                f"Age: {best_agent.age}/{best_agent.max_age}",
                f"Actions: {len(best_agent.actions)}",
                f"Heritage: {len(best_agent.lineage.genetic_heritage)}"
            ]

            for stat in agent_stats:
                text = self.detail_font.render(stat, True, self.COLORS['generation_text'])
                self.screen.blit(text, (WINDOW_SIZE[0] + 10, y_offset))
                y_offset += 20
                
        # Add evolution statistics
        if agents:
            y_offset += 30
            evolution_stats = [
                "Evolution Stats:",
                f"Avg Actions: {sum(len(a.actions) for a in agents) / len(agents):.1f}",
                f"Novel Actions: {sum(1 for a in agents for act in a.actions if 'evolved' in act)}",
                f"Recent Mutations: {sum(len(a.lineage.mutations[-3:]) for a in agents)}",
                f"Top Evolution Potential: {max(a.evolution_stats['evolution_potential'] for a in agents):.2f}"
            ]
            
            for stat in evolution_stats:
                text = self.detail_font.render(stat, True, self.COLORS['evolved_action'])
                self.screen.blit(text, (WINDOW_SIZE[0] + 10, y_offset))
                y_offset += 20

    def _draw_population_graph(self):
        if len(self.stats.population_history) < 2:
            return

        graph_rect = pygame.Rect(WINDOW_SIZE[0] + 10, 400, DETAIL_PANEL_WIDTH - 20, 100)
        pygame.draw.rect(self.screen, self.COLORS['background'], graph_rect)
        
        points = []
        max_pop = max(self.stats.population_history)
        if max_pop == 0:
            return
            
        for i, pop in enumerate(self.stats.population_history):
            x = graph_rect.left + (i * graph_rect.width / STATS_HISTORY_LENGTH)
            y = graph_rect.bottom - (pop * graph_rect.height / max_pop)
            points.append((x, y))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.COLORS['graph_line'], False, points, 2)
            
        # Add label
        label = self.detail_font.render("Population", True, self.COLORS['text'])
        self.screen.blit(label, (graph_rect.left, graph_rect.top - 15))

    def _draw_innovation_graph(self):
        if len(self.stats.innovation_history) < 2:
            return

        graph_rect = pygame.Rect(WINDOW_SIZE[0] + 10, 520, DETAIL_PANEL_WIDTH - 20, 100)
        pygame.draw.rect(self.screen, self.COLORS['background'], graph_rect)
        
        points = []
        max_innovations = max(self.stats.innovation_history)
        if max_innovations == 0:
            return
            
        for i, innovations in enumerate(self.stats.innovation_history):
            x = graph_rect.left + (i * graph_rect.width / STATS_HISTORY_LENGTH)
            y = graph_rect.bottom - (innovations * graph_rect.height / max_innovations)
            points.append((x, y))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.COLORS['graph_line'], False, points, 2)
            
        # Add label
        label = self.detail_font.render("Innovations", True, self.COLORS['text'])
        self.screen.blit(label, (graph_rect.left, graph_rect.top - 15))

    def _draw_evolution_events(self):
        """Draw recent evolution events"""
        y_offset = WINDOW_SIZE[1] - 100
        for event in self.evolution_events:
            text = self.detail_font.render(event['text'], True, event['color'])
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

    def add_evolution_event(self, text, event_type='evolution'):
        """Record an evolution event"""
        color = {
            'evolution': self.COLORS['evolved_action'],
            'mutation': self.COLORS['mutation'],
            'adaptation': self.COLORS['adaptation']
        }.get(event_type, self.COLORS['text'])
        
        self.evolution_events.append({
            'text': text,
            'color': color,
            'time': pygame.time.get_ticks()
        })

    def _draw_achievement(self, achievement):
        """Display a temporary achievement notification"""
        self.achievements.append({
            'text': achievement,
            'time': 60,  # Display for 60 frames
            'y_offset': len(self.achievements) * 30
        })
        
    def draw_agent_activities(self, agent):
        """Draw visual indicators for agent activities"""
        if agent.is_creating_action:
            self.add_action_creation_effect(agent.position)
        
        # Show active mutations
        if agent.lineage.mutations:
            mutation_text = self.detail_font.render(
                f"M:{len(agent.lineage.mutations)}", 
                True, self.COLORS['mutation']
            )
            self.screen.blit(
                mutation_text,
                (int(agent.position[0] * CELL_SIZE) - 30, 
                 int(agent.position[1] * CELL_SIZE) - 20)
            )

    def take_screenshot(self, event_name: str = "") -> str:
        """Take a screenshot of the current simulation state"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.screenshot_dir}/screenshot_{event_name}_{timestamp}.png"
        
        try:
            pygame.image.save(self.screen, filename)
            logger.info(f"Screenshot saved: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return ""

    def capture_important_event(self, event_name: str, event_data: dict = None):
        """Capture important simulation events with screenshots"""
        screenshot_path = self.take_screenshot(event_name)
        
        if event_data and self.debug_mode:
            # Save additional event data
            data_path = screenshot_path.replace('.png', '_data.json')
            try:
                with open(data_path, 'w') as f:
                    json.dump(event_data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save event data: {e}")

def save_best_agents(agents):
    """Save best performing agents to file"""
    if not agents:
        return
        
    # Sort agents by various metrics
    best_agents = {
        'energy': max(agents, key=lambda a: a.energy),
        'generation': max(agents, key=lambda a: a.lineage.generation),
        'innovations': max(agents, key=lambda a: len(a.actions))
    }
    
    # Save agent data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for category, agent in best_agents.items():
        filename = f"simulation_stats/best_{category}_agent_{timestamp}.pkl"
        try:
            with open(filename, 'wb') as f:
                pickle.dump(agent, f)
            logger.info(f"Saved best {category} agent to {filename}")
        except Exception as e:
            logger.error(f"Failed to save {category} agent: {e}")

def calculate_network_architecture(genetic_core: GeneticCore) -> Dict[str, int]:
    """Calculate neural network architecture based on genetic traits"""
    # Sensor inputs based on physical genetics
    sensor_inputs = int(16 * (1 + genetic_core.physical_genetics.sensor_sensitivity))
    
    # Memory inputs based on brain capacity
    memory_inputs = int(8 * genetic_core.brain_genetics.memory_capacity)
    base_input = sensor_inputs + memory_inputs
    
    # Output neurons based on brain and mind traits
    base_output = int(12 * (1 + genetic_core.mind_genetics.creativity))
    
    # Brain complexity affects overall network scaling
    brain_complexity = (
        genetic_core.brain_genetics.processing_speed * 0.3 +
        genetic_core.brain_genetics.pattern_recognition * 0.3 +  # Changed from mind_genetics to brain_genetics
        genetic_core.brain_genetics.neural_plasticity * 0.2 +
        genetic_core.brain_genetics.learning_rate * 0.2
    )
    
    # Calculate scaling factors
    input_scaling = 1 + (brain_complexity * 0.5)
    output_scaling = 1 + (brain_complexity * 0.3)
    
    return {
        'input_size': int(base_input * input_scaling),
        'output_size': int(base_output * output_scaling),
        'memory_size': memory_inputs
    }

class PopulationGenetics:
    def __init__(self):
        self.genetic_diversity = {}
        self.dominant_traits = []
        
    def track_population_genetics(self, agents):
        trait_frequencies = defaultdict(int)
        for agent in agents:
            for trait, value in agent.genetic_core.get_all_traits():
                trait_frequencies[trait] += value

    def calculate_mutation_rate(self, population_size: int, genetic_diversity: float) -> float:
        """Dynamically adjust mutation rate based on population health"""
        base_rate = 0.01
        diversity_factor = 1.0 - genetic_diversity  # Higher when diversity is low
        population_factor = math.exp(-population_size / 100)  # Higher when population is small
        
        return base_rate * (1 + diversity_factor + population_factor)

def create_predator(env_size: Tuple[int, int]) -> AdaptivePredator:
    """Create a new predator with its own neural network"""
    genetic_core = GeneticCore()
    
    # Calculate network architecture
    network_params = calculate_network_architecture(genetic_core)
    
    # Create neural network
    neural_net = NeuralAdaptiveNetwork(
        input_size=network_params['input_size'],
        hidden_size=network_params['hidden_size'],
        output_size=network_params['output_size'],
        memory_size=network_params['memory_size'],
        learning_rate=genetic_core.brain_genetics.learning_rate,
        plasticity=genetic_core.brain_genetics.neural_plasticity
    )
    
    position = (
        random.randint(0, env_size[0]-1),
        random.randint(0, env_size[1]-1)
    )
    
    return AdaptivePredator(
        genetic_core=genetic_core,
        neural_net=neural_net,
        position=position
    )

def run_simulation():
    try:
        # Initialize components
        logger.info("Starting simulation...")
        env = AdaptiveEnvironment(size=(80, 60), complexity=0.5)
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
                genetic_core = GeneticCore()
                
                # Calculate network architecture based on genetics
                network_params = calculate_network_architecture(genetic_core)
                
                # Create neural network with genetic-based architecture
                neural_net = NeuralAdaptiveNetwork(
                    input_size=network_params['input_size'],
                    output_size=network_params['output_size'],
                    genetic_core=genetic_core  # Pass genetic_core instead of hidden_size
                )
                
                position = (
                    random.randint(0, env.size[0]-1),
                    random.randint(0, env.size[1]-1)
                )
                
                agent = AdaptiveAgent(
                    genetic_core=genetic_core,
                    neural_net=neural_net,
                    position=position
                )
                agents.append(agent)
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
                    env._update_state()
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
                                    if hasattr(visualizer, 'add_evolution_event'):
                                        visualizer.add_evolution_event(
                                            f"Agent {agent.name[:8]} reproduced (fitness: {agent.calculate_fitness():.2f})",
                                            'evolution'
                                        )
                                
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
                    if debugger.frame_count % 1000 == 0:
                        visualizer.stats.save_graphs()
        
            except Exception as e:
                logger.error(f"Error in simulation loop: {str(e)}")
                continue

        # Cleanup
        visualizer.stats.save_graphs()
        pygame.quit()
        logger.info("Simulation ended normally")
        
    except Exception as e:
        logger.error(f"Fatal simulation error: {str(e)}")
        pygame.quit()
        raise

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}")
        sys.exit(1)


logger = logging.getLogger(__name__)

class SimulationManager:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize simulation components"""
        self.config = config or self._default_config()
        self.agent_assembler = AgentAssembler()
        
        # Initialize core systems
        self.birth_registry = BirthRegistry()
        self.embryo_generator = EmbryoGenerator()
        self.environment = AdaptiveEnvironment(
            size=(self.config['env_width'], self.config['env_height']),
            complexity=self.config['env_complexity']
        )
        
        # Initialize visualization
        self.visualizer = Visualizer(
            width=self.config['display_width'],
            height=self.config['display_height'],
            debug_mode=self.config['debug_mode']
        )
        
        # Population management
        self.agents: List[AdaptiveAgent] = []
        self.generation = 0
        
        # Statistics tracking
        self.stats = {
            'births': 0,
            'deaths': 0,
            'mutations': 0,
            'adaptations': 0
        }

    def initialize_population(self):
        """Create initial population"""
        for _ in range(self.config['initial_population']):
            agent = self._create_agent()
            self.agents.append(agent)
            logger.info(f"Created agent: {agent.id}")

    def _create_agent(self, parent: Optional[AdaptiveAgent] = None) -> AdaptiveAgent:
        """Create a new agent using the assembler"""
        position = (
            random.randint(0, self.environment.size[0]),
            random.randint(0, self.environment.size[1])
        )
        
        assembled = self.agent_assembler.create_agent(position, parent)
        logger.info(f"Created agent with stats: {assembled.stats}")
        return assembled.agent

    def run_simulation(self):
        """Main simulation loop"""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Update environment
            self.environment.update()
            
            # Update each agent
            for agent in self.agents[:]:  # Copy list to allow modifications
                # Update agent systems
                agent.update(self.environment.get_state())
                
                # Process rest cycles
                if agent.needs_rest():
                    agent.process_rest_cycle()
                
                # Handle reproduction
                if agent.can_reproduce():
                    self._handle_reproduction(agent)
                
                # Handle death
                if not agent.is_alive():
                    self._handle_death(agent)
            
            # Update diagnostics
            self._update_diagnostics()
            
            # Update visualization
            self.visualizer.draw(
                self.environment,
                self.agents,
                self.diagnostics
            )
            
            # Maintain frame rate
            clock.tick(self.config['fps'])
        
        self._cleanup()

    def _handle_reproduction(self, parent: AdaptiveAgent):
        """Handle agent reproduction"""
        offspring = self._create_agent(parent)
        self.agents.append(offspring)
        self.stats['births'] += 1
        logger.info(f"New agent born: {offspring.id}, Parent: {parent.id}")

    def _handle_death(self, agent: AdaptiveAgent):
        """Handle agent death"""
        self.agents.remove(agent)
        self.stats['deaths'] += 1
        logger.info(f"Agent died: {agent.id}")

    def _update_diagnostics(self):
        """Update all diagnostic systems"""
        for agent in self.agents:
            self.diagnostics['neural'].update(agent.neural_net)
            self.diagnostics['agent'].update(agent)
        self.diagnostics['population'].update(self.agents)

    def _cleanup(self):
        """Cleanup resources"""
        pygame.quit()
        self._save_simulation_data()

    def _save_simulation_data(self):
        """Save simulation results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = 'simulation_results'
        os.makedirs(save_dir, exist_ok=True)
        
        data = {
            'stats': self.stats,
            'config': self.config,
            'generation': self.generation,
            'timestamp': timestamp
        }
        
        with open(f'{save_dir}/sim_{timestamp}.json', 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _default_config() -> Dict:
        """Default simulation configuration with full system parameters"""
        return {
            # Environment settings
            'env_width': 800,
            'env_height': 600,
            'env_complexity': 0.5,
            'resource_spawn_rate': 0.02,
            'resource_max_quantity': 100,
            'threat_probability': 0.01,
            
            # Display settings
            'display_width': 1024,
            'display_height': 768,
            'debug_mode': True,
            'fps': 60,
            
            # Population settings
            'initial_population': 10,
            'max_population': 100,
            'min_reproduction_energy': 50,
            'reproduction_cost': 30,
            
            # Neural network parameters
            'neural_input_size': 16,
            'neural_output_size': 8,
            'neural_plasticity': 0.3,
            'learning_rate': 0.01,
            'memory_capacity': 32,
            
            # Genetic parameters
            'mutation_rate': 0.05,
            'crossover_rate': 0.7,
            'gene_complexity': 0.4,
            
            # Brain system settings
            'brain_processing_power': 1.0,
            'pattern_recognition_threshold': 0.6,
            'neural_adaptation_rate': 0.2,
            
            # Heart system settings
            'base_metabolism_rate': 1.0,
            'energy_efficiency': 0.8,
            'stamina_recovery_rate': 0.1,
            
            # Mind system settings
            'creativity_factor': 0.5,
            'learning_capacity': 0.7,
            'decision_threshold': 0.3,
            'memory_persistence': 0.8,
            
            # Embryo generation settings
            'development_time': 100,
            'inheritance_strength': 0.7,
            'trait_mutation_chance': 0.1,
            
            # Birth registry settings
            'lineage_tracking_depth': 5,
            'heritage_influence': 0.4,
            
            # Diagnostic settings
            'diagnostic_update_rate': 10,
            'save_interval': 1000,
            'performance_tracking': True,
            
            # Evolution settings
            'adaptation_threshold': 0.6,
            'innovation_requirement': 0.8,
            'genetic_diversity_target': 0.7,
            'selection_pressure': 0.5
        }

    def initialize_display(self):
        """Initialize pygame display with proper caption"""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((
                self.config['display_width'],
                self.config['display_height']
            ))
            pygame.display.set_caption("Adaptive Simulation")  # Changed from setCaption
            
            logger.info(f"Display initialized: {self.config['display_width']}x{self.config['display_height']}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize display: {e}")
            return False