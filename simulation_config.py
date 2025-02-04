"""Configuration settings for the simulation"""

SIMULATION_SETTINGS = {
    'WINDOW_SIZE': (800, 600),
    'CELL_SIZE': 10,
    'FPS': 60,
    'INITIAL_AGENTS': 5,
    'INITIAL_PREDATORS': 2,
    'INITIAL_RESOURCES': 5,
    'MIN_REPRODUCTION_ENERGY': 50.0,
    'REPRODUCTION_COST': 40.0,
    'MIN_REPRODUCTION_AGE': 100,
    'MAX_REPRODUCTION_AGE_FACTOR': 0.8,
    'MUTATION_RATE': 0.1
}

COLORS = {
    'agent': (0, 255, 0),
    'predator': (255, 0, 0),
    'resource': (255, 255, 0),
    'background': (0, 0, 0),
    'text': (255, 255, 255)
}