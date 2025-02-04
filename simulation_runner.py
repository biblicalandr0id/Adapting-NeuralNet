import pygame
import sys
import numpy as np
from agent import AdaptiveAgent
from genetics import GeneticCore
from neural_networks import NeuralAdaptiveNetwork
from adaptive_environment import AdaptiveEnvironment, ResourceType, Resource
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import os

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = (800, 600)
CELL_SIZE = 10
DETAIL_PANEL_WIDTH = 200
STATS_HISTORY_LENGTH = 1000
COLORS = {
    'agent': (0, 255, 0),  # Green
    'resource': (255, 255, 0),  # Yellow
    'threat': (255, 0, 0),  # Red
    'background': (0, 0, 0),  # Black
    'text': (255, 255, 255),  # White
    'energy_bar': (50, 205, 50),  # Lime Green
    'age_bar': (135, 206, 235),   # Sky Blue
    'generation_text': (255, 215, 0),  # Gold
    'stats_background': (0, 0, 0, 128),  # Semi-transparent black
    'graph_line': (255, 140, 0),  # Dark Orange
    'achievement': (255, 223, 0)   # Golden achievement color
}

class SimulationStats:
    def __init__(self):
        self.population_history = deque(maxlen=STATS_HISTORY_LENGTH)
        self.innovation_history = deque(maxlen=STATS_HISTORY_LENGTH)
        self.generation_stats = {}
        self.achievements = []
        self.start_time = datetime.now()
        
        # Create stats directory if it doesn't exist
        os.makedirs("simulation_stats", exist_ok=True)

    def update(self, agents, time_step):
        self.population_history.append(len(agents))
        total_innovations = sum(len(a.actions) - 5 for a in agents)
        self.innovation_history.append(total_innovations)
        
        # Update generation stats
        current_gens = [a.lineage.generation for a in agents]
        self.generation_stats[time_step] = {
            'max_gen': max(current_gens, default=0),
            'avg_gen': sum(current_gens) / len(current_gens) if current_gens else 0
        }

    def save_graphs(self):
        plt.figure(figsize=(15, 10))
        
        # Population graph
        plt.subplot(2, 1, 1)
        plt.plot(list(self.population_history), label='Population', color='green')
        plt.title('Population Over Time')
        plt.grid(True)
        
        # Innovation graph
        plt.subplot(2, 1, 2)
        plt.plot(list(self.innovation_history), label='Innovations', color='orange')
        plt.title('Total Innovations Over Time')
        plt.grid(True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"simulation_stats/simulation_graphs_{timestamp}.png")
        plt.close()

class SimulationVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE[0] + DETAIL_PANEL_WIDTH, WINDOW_SIZE[1]))
        pygame.display.set_caption("Adaptive Agent Simulation")
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        self.action_creation_effects = []  # Store visual effects
        self.stats = SimulationStats()
        self.detail_font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 28)
        
        # Add colors for exceptional agents and effects
        self.COLORS.update({
            'exceptional_agent': (0, 255, 255),  # Cyan for creative agents
            'action_creation': (255, 165, 0),    # Orange flash for new action
            'lineage_marker': (147, 112, 219)    # Purple for agents with heritage
        })
        
    def draw(self, env, agents):
        self.screen.fill(COLORS['background'])
        
        # Draw resources
        for resource in env.current_state.resources:
            pygame.draw.circle(
                self.screen,
                COLORS['resource'],
                (int(resource.position[0] * CELL_SIZE), 
                 int(resource.position[1] * CELL_SIZE)),
                int(np.sqrt(resource.quantity))
            )
        
        # Draw threats
        for threat in env.current_state.threats:
            pygame.draw.circle(
                self.screen,
                COLORS['threat'],
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
                COLORS['exceptional_agent'] 
                if innovation_potential > 0.85 
                else COLORS['agent']
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
                    COLORS['lineage_marker'],
                    (int(agent.position[0] * CELL_SIZE), 
                     int(agent.position[1] * CELL_SIZE)),
                    9,
                    1  # Width of circle outline
                )
            
            # Draw agent stats
            stats_text = self.font.render(
                f"E:{int(agent.energy)} G:{agent.lineage.generation}", 
                True, COLORS['text']
            )
            self.screen.blit(
                stats_text, 
                (int(agent.position[0] * CELL_SIZE) - 20, 
                 int(agent.position[1] * CELL_SIZE) - 20)
            )
        
        # Draw action creation effects
        for effect in self.action_creation_effects[:]:
            pos, time_left = effect
            if time_left > 0:
                # Draw expanding circle effect
                radius = (30 - time_left) * 2
                pygame.draw.circle(
                    self.screen,
                    COLORS['action_creation'],
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
            True, COLORS['text']
        )
        self.screen.blit(stats_text, (10, 10))
        
        # Draw detailed stats panel
        self._draw_detail_panel(env, agents)
        
        # Draw mini-graphs
        self._draw_population_graph()
        self._draw_innovation_graph()
        
        # Update stats
        self.stats.update(agents, env.current_state.time_step)
        
        pygame.display.flip()
        self.clock.tick(30)

    def add_action_creation_effect(self, position):
        """Add visual effect for action creation"""
        self.action_creation_effects.append([position, 30])  # 30 frames duration

    def _draw_detail_panel(self, env, agents):
        panel_rect = pygame.Rect(WINDOW_SIZE[0], 0, DETAIL_PANEL_WIDTH, WINDOW_SIZE[1])
        pygame.draw.rect(self.screen, COLORS['stats_background'], panel_rect)
        
        y_offset = 10
        
        # Simulation title
        title = self.title_font.render("Simulation Details", True, COLORS['text'])
        self.screen.blit(title, (WINDOW_SIZE[0] + 10, y_offset))
        y_offset += 40

        # Time stats
        runtime = datetime.now() - self.stats.start_time
        time_stats = [
            f"Runtime: {runtime.seconds//3600:02d}:{(runtime.seconds//60)%60:02d}:{runtime.seconds%60:02d}",
            f"Time Step: {env.current_state.time_step}",
            f"Population: {len(agents)}",
            f"Max Generation: {max(a.lineage.generation for a in agents)}",
            f"Total Innovations: {sum(len(a.actions) - 5 for a in agents)}"
        ]

        for stat in time_stats:
            text = self.detail_font.render(stat, True, COLORS['text'])
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
                text = self.detail_font.render(stat, True, COLORS['generation_text'])
                self.screen.blit(text, (WINDOW_SIZE[0] + 10, y_offset))
                y_offset += 20

    def _draw_population_graph(self):
        if len(self.stats.population_history) < 2:
            return

        graph_rect = pygame.Rect(WINDOW_SIZE[0] + 10, 400, DETAIL_PANEL_WIDTH - 20, 100)
        pygame.draw.rect(self.screen, COLORS['background'], graph_rect)
        
        points = []
        max_pop = max(self.stats.population_history)
        for i, pop in enumerate(self.stats.population_history):
            x = graph_rect.left + (i * graph_rect.width / STATS_HISTORY_LENGTH)
            y = graph_rect.bottom - (pop * graph_rect.height / max_pop)
            points.append((x, y))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, COLORS['graph_line'], False, points, 2)

    def _draw_innovation_graph(self):
        if len(self.stats.innovation_history) < 2:
            return

        graph_rect = pygame.Rect(WINDOW_SIZE[0] + 10, 510, DETAIL_PANEL_WIDTH - 20, 100)
        pygame.draw.rect(self.screen, COLORS['background'], graph_rect)
        
        points = []
        max_innov = max(self.stats.innovation_history)
        for i, innov in enumerate(self.stats.innovation_history):
            x = graph_rect.left + (i * graph_rect.width / STATS_HISTORY_LENGTH)
            y = graph_rect.bottom - (innov * graph_rect.height / max_innov)
            points.append((x, y))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, COLORS['graph_line'], False, points, 2)

    def _draw_achievement(self, achievement):
        """Display a temporary achievement notification"""
        self.achievements.append({
            'text': achievement,
            'time': 60,  # Display for 60 frames
            'y_offset': len(self.achievements) * 30
        })

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
        with open(filename, 'wb') as f:
            pickle.dump(agent, f)

def run_simulation():
    # Initialize environment
    env_size = (80, 60)  # Scaled to fit window
    env = AdaptiveEnvironment(size=env_size, complexity=0.5)
    
    # Initialize agents
    agents = []
    for _ in range(5):  # Start with 5 agents
        genetic_core = GeneticCore()
        neural_net = NeuralAdaptiveNetwork(input_size=10, hidden_size=20, output_size=5)
        position = (
            random.randint(0, env_size[0]-1),
            random.randint(0, env_size[1]-1)
        )
        agent = AdaptiveAgent(genetic_core, neural_net, position)
        agents.append(agent)
    
    # Initialize visualizer
    visualizer = SimulationVisualizer()
    
    running = True
    paused = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if not paused:
            # Update environment
            env._update_state()
            
            # List to store new offspring
            new_agents = []
            
            # Update each agent
            for agent in agents[:]:  # Create a copy of the list to modify
                # Check survival
                if not agent.update(env.current_state):
                    agents.remove(agent)
                    continue
                
                # Check for action creation if energy is low
                if agent.energy < 30.0:
                    if new_action := agent.create_action('energy', env.current_state):
                        action_name, _ = new_action
                        print(f"Agent {agent.name} (Gen {agent.lineage.generation}) created: {action_name}")
                        visualizer.add_action_creation_effect(agent.position)
                
                # Attempt reproduction
                if random.random() < 0.01:  # 1% chance each update
                    offspring = agent.reproduce(env.current_state)
                    if offspring:
                        new_agents.append(offspring)
                
                # Regular action processing
                action, params = agent.decide_action(env.current_state)
                result = agent.execute_action(action, params, env.current_state)
                agent.learn_from_experience(env.current_state, action, result)
            
            # Add new offspring to simulation
            agents.extend(new_agents)
            
            # Save best performing agents periodically
            if env.current_state.time_step % 1000 == 0:
                save_best_agents(agents)
            
            # Draw current state
            visualizer.draw(env, agents)
    
    pygame.quit()

if __name__ == "__main__":
    run_simulation()