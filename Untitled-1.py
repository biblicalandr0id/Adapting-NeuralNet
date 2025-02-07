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
