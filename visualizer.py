import pygame
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, width: int = 800, height: int = 600, debug_mode: bool = False):
        """Initialize the visualization system with debugging capabilities"""
        pygame.init()
        
        # Screen setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width + 200, height))  # Extra space for debug info
        pygame.display.set_caption("Adaptive Simulation - Debug Mode" if debug_mode else "Adaptive Simulation")
        
        # Performance monitoring
        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.last_time = pygame.time.get_ticks()
        self.fps_history = []
        
        # Font setup
        self.font = pygame.font.Font(None, 24)
        self.debug_font = pygame.font.Font(None, 20)
        
        # Visual settings
        self.colors = {
            'background': (0, 0, 0),
            'agent': (0, 255, 0),
            'predator': (255, 0, 0),
            'resource': (255, 255, 0),
            'text': (255, 255, 255),
            'debug': (255, 165, 0),  # Orange for debug info
            'warning': (255, 255, 0),
            'error': (255, 0, 0)
        }
        
        # Debug settings
        self.debug_mode = debug_mode
        self.debug_info: Dict[str, any] = {}
        self.error_log: List[str] = []
        
        # Screenshot directory
        self.screenshot_dir = 'screenshots'
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        logger.info(f"Visualizer initialized with resolution: {width}x{height}, Debug mode: {debug_mode}")

    def draw(self, env, agents: List, predators: List):
        """Enhanced draw method with debug information"""
        try:
            # Clear screen
            self.screen.fill(self.colors['background'])
            
            # Draw simulation elements
            self._draw_resources(env.current_state.resources)
            self._draw_agents(agents)
            self._draw_predators(predators)
            
            # Draw debug information if enabled
            if self.debug_mode:
                self._draw_debug_info(env, agents, predators)
            
            # Update display
            pygame.display.flip()
            
            # Track performance
            self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Draw error: {str(e)}")
            self.error_log.append(f"Draw error: {str(e)}")

    def _draw_resources(self, resources: List):
        """Draw resources with quantity indication"""
        for resource in resources:
            try:
                size = int(5 * (1 + resource.quantity/100))  # Scale circle with quantity
                pygame.draw.circle(
                    self.screen,
                    self.colors['resource'],
                    (int(resource.position[0]), int(resource.position[1])),
                    size
                )
            except Exception as e:
                logger.error(f"Resource drawing error: {str(e)}")

    def _draw_agents(self, agents: List):
        """Draw agents"""
        for agent in agents:
            pygame.draw.circle(
                self.screen,
                self.colors['agent'],
                (int(agent.position[0]), int(agent.position[1])),
                7
            )

    def _draw_predators(self, predators: List):
        """Draw predators"""
        for predator in predators:
            pygame.draw.circle(
                self.screen,
                self.colors['predator'],
                (int(predator.position[0]), int(predator.position[1])),
                8
            )

    def _draw_debug_info(self, env, agents: List, predators: List):
        """Draw debug information panel"""
        debug_surface = pygame.Surface((200, self.height))
        debug_surface.fill((50, 50, 50))
        
        y_offset = 10
        stats = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Agents: {len(agents)}",
            f"Predators: {len(predators)}",
            f"Resources: {len(env.current_state.resources)}",
            f"Time: {env.current_state.time_step}"
        ]
        
        for stat in stats:
            text = self.debug_font.render(stat, True, self.colors['text'])
            debug_surface.blit(text, (10, y_offset))
            y_offset += 25
        
        self.screen.blit(debug_surface, (self.width, 0))

    def _update_performance_metrics(self):
        """Update and track performance metrics"""
        self.frame_count += 1
        current_time = pygame.time.get_ticks()
        
        if current_time - self.last_time > 1000:  # Every second
            fps = self.frame_count * 1000 / (current_time - self.last_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > 60:  # Keep last minute
                self.fps_history.pop(0)
            
            self.frame_count = 0
            self.last_time = current_time
            
            # Log performance issues
            if fps < 30:
                logger.warning(f"Low performance detected: {fps:.1f} FPS")

    def take_screenshot(self):
        """Capture current simulation state"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.screenshot_dir}/simulation_{timestamp}.png"
        pygame.image.save(self.screen, filename)
        logger.info(f"Screenshot saved: {filename}")