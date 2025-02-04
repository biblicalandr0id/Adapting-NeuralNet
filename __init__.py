# Standard library imports
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
import json
import uuid

# Third-party imports
import numpy as np
import torch
import pygame

# Local application imports
from .agent import AdaptiveAgent
from .genetics import (
    GeneticCore,
    BaseTraits,
    MindGenetics,
    BrainGenetics,
    PhysicalGenetics,
    HeartGenetics,
    DigitalNucleotide,
    GeneticTrait
)
from .neural_networks import NeuralAdaptiveNetwork
from .adaptive_environment import (
    AdaptiveEnvironment,
    EnvironmentalState,
    ResourceType,
    Resource
)
from .predator import Predator, AdaptivePredator
from .birth_records import BirthRecord, BirthRegistry
from .embryo_generator import EmbryoGenerator, EmbryoNamer
from .diagnostics import (
    NeuralDiagnostics,
    AgentDiagnostics,
    PopulationDiagnostics
)
from .visualizer import Visualizer
from .executor import AdaptiveExecutor
from .simulation_runner import SimulationStats, SimulationVisualizer, SimulationManager
from .logger_config import setup_logging
from .brainui import BrainCore, create_brain_interface, BrainUI
from .mind import (
    EmbronicMind,
    MindMetrics,
    MindState,
    GrowthMetrics,
    Memory,
    AgentEmbryo,
    Mind,
    create_embryo
)
from .brain import Brain, BrainState
from .dna import DNAGuide, PhysicalAttributes
from .heart import HeartSystem, HeartState

# Configure logging
logger = logging.getLogger(__name__)

__all__ = [
    # Agent-related
    'AdaptiveAgent',
    
    # Genetics-related
    'GeneticCore',
    'BaseTraits',
    'MindGenetics',
    'BrainGenetics',
    'PhysicalGenetics',
    'HeartGenetics',
    'DigitalNucleotide',
    'GeneticTrait',
    
    # Neural network
    'NeuralAdaptiveNetwork',
    
    # Environment
    'AdaptiveEnvironment',
    'EnvironmentalState',
    'ResourceType',
    'Resource',
    
    # Predator
    'Predator',
    'AdaptivePredator',
    
    # Birth and embryo
    'BirthRecord',
    'BirthRegistry',
    'EmbryoGenerator',
    'EmbryoNamer',
    
    # Diagnostics and visualization
    'NeuralDiagnostics',
    'AgentDiagnostics',
    'PopulationDiagnostics',
    'Visualizer',
    
    # Execution and simulation
    'AdaptiveExecutor',
    'SimulationStats',
    'SimulationVisualizer',
    'SimulationManager',
    
    # Brain and mind
    'BrainCore',
    'create_brain_interface',
    'EmbronicMind',
    'MindMetrics',
    'MindState',
    
    # DNA and physical
    'DNAGuide',
    'PhysicalAttributes',
    
    # Heart system
    'HeartSystem',
    'HeartState',
    
    # Mind system
    'GrowthMetrics',
    'Memory',
    'MindMetrics',
    'MindState',
    'AgentEmbryo',
    'EmbronicMind',
    'Mind',
    'create_embryo',
    
    # Brain system
    'Brain',
    'BrainState',
    'BrainUI',
    
    # Utilities
    'setup_logging'
]

# Version info
__version__ = '0.1.0'
__author__ = 'Austin Grandstaff'
__email__ = 'Biblicalandr0id@gmail.com'

# Package metadata
name = "adaptive_simulation"
description = "An adaptive agent simulation with evolving neural networks and genetics"