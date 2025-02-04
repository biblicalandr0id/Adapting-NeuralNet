import pygame
import numpy as np
from typing import List, Tuple
from agent import AdaptiveAgent
from adaptive_environment import AdaptiveEnvironment, Resource, ResourceType

class Visualizer:
    def __init__(self, width: int, height: int, scale: int = 5):
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale
        self.screen = pygame.display.set_mode((width * scale, height * scale))
        pygame.display.set_caption("Adaptive Agent Simulation")
        
        # Color schemes
        self.colors = {
            'background': (10, 10, 20),
            'agent': (0, 255, 0),
            'energy': (255, 255, 0),
            'information': (0, 255, 255),
            'materials': (255, 128, 0),
            'threat': (255, 0, 0),
            'text': (255, 255, 255)
        }
        
        self.font = pygame.font.Font(None, 24)

    def render(self, env: AdaptiveEnvironment, agents: List[AdaptiveAgent]):
        """Render current simulation state"""
        self.screen.fill(self.colors['background'])
        
        # Draw resources
        for resource in env.current_state.resources:
            color = self.colors[resource.type.value]
            size = int(min(10, resource.quantity / 10))
            pos = (int(resource.position[0] * self.scale), 
                  int(resource.position[1] * self.scale))
            pygame.draw.circle(self.screen, color, pos, size)

        # Draw threats
        for threat in env.current_state.threats:
            pos = (int(threat[0] * self.scale), 
                  int(threat[1] * self.scale))
            pygame.draw.circle(self.screen, self.colors['threat'], pos, 5)

        # Draw agents
        for agent in agents:
            pos = (int(agent.position[0] * self.scale), 
                  int(agent.position[1] * self.scale))
            # Agent color varies based on energy level
            energy_ratio = agent.energy / 100.0
            agent_color = (
                int(255 * (1 - energy_ratio)),
                int(255 * energy_ratio),
                0
            )
            pygame.draw.circle(self.screen, agent_color, pos, 3)

        # Draw stats
        stats = f"Agents: {len(agents)} | Time: {env.current_state.time_step}"
        text = self.font.render(stats, True, self.colors['text'])
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def check_events(self) -> bool:
        """Handle pygame events, return False if should quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def close(self):
        """Clean up pygame"""
        pygame.quit()
