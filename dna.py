#dna.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import math

@dataclass
class PhysicalAttributes:
    """Physical capabilities and limitations"""
    size: float = 0.1  # Base unit size
    energy_capacity: float = 100.0
    processing_speed: float = 0.5
    memory_capacity: float = 1000.0
    sensor_resolution: float = 1.0
    action_precision: float = 1.0

@dataclass
class GrowthGenes:
    """Genetic instructions for physical growth"""
    size_multiplier: float = 1.0
    energy_efficiency: float = 1.0
    processing_multiplier: float = 1.0
    memory_expansion_rate: float = 1.0
    sensor_development_rate: float = 1.0
    action_refinement_rate: float = 1.0
    maturation_rate: float = 1.0

class DNAGuide:
    def __init__(self):
        self.physical_attributes = PhysicalAttributes()
        self.growth_genes = GrowthGenes()
        self.age = 0.0
        self.maturity_age = 100.0
        self.growth_stages = {
            "embryonic": (0, 20),
            "juvenile": (20, 50),
            "adolescent": (50, 80),
            "mature": (80, 100)
        }
        
        # Growth curve parameters
        self.growth_curves = {
            "sigmoid": lambda x: 1 / (1 + math.exp(-x)),
            "exponential": lambda x: 1 - math.exp(-x),
            "logarithmic": lambda x: math.log(x + 1) / math.log(2)
        }
        
    def initialize_growth_pattern(self, pattern_type: str = "balanced") -> None:
        """Initialize growth genes based on desired pattern"""
        if pattern_type == "balanced":
            self.growth_genes = GrowthGenes(
                size_multiplier=1.2,
                energy_efficiency=1.1,
                processing_multiplier=1.15,
                memory_expansion_rate=1.25,
                sensor_development_rate=1.1,
                action_refinement_rate=1.1,
                maturation_rate=1.0
            )
        elif pattern_type == "rapid":
            self.growth_genes = GrowthGenes(
                size_multiplier=1.5,
                energy_efficiency=1.3,
                processing_multiplier=1.4,
                memory_expansion_rate=1.6,
                sensor_development_rate=1.3,
                action_refinement_rate=1.3,
                maturation_rate=1.5
            )
        # Add more patterns as needed
        
    def calculate_growth_phase(self, attribute_value: float, age: float, 
                             growth_rate: float, curve_type: str = "sigmoid") -> float:
        """Calculate growth for a specific attribute using selected growth curve"""
        normalized_age = age / self.maturity_age
        growth_curve = self.growth_curves[curve_type]
        
        # Apply growth curve with gene-specific rate
        growth_factor_input = normalized_age * growth_rate * 10

        if growth_factor_input > 20:
            growth_factor = growth_curve(20)
        else:
            growth_factor = growth_curve(growth_factor_input)
        return attribute_value * (1 + growth_factor)
    
    def update_physical_attributes(self, time_delta: float) -> None:
        """Update physical attributes based on DNA instructions"""
        self.age += time_delta
        
        # Calculate growth stage
        current_stage = self._determine_growth_stage()
        
        # Update each physical attribute based on growth genes
        self.physical_attributes.size = self.calculate_growth_phase(
            1.0, self.age, self.growth_genes.size_multiplier)
            
        self.physical_attributes.energy_capacity = self.calculate_growth_phase(
            100.0, self.age, self.growth_genes.energy_efficiency)
            
        self.physical_attributes.processing_speed = self.calculate_growth_phase(
            1.0, self.age, self.growth_genes.processing_multiplier)
            
        self.physical_attributes.memory_capacity = self.calculate_growth_phase(
            1000.0, self.age, self.growth_genes.memory_expansion_rate)
            
        self.physical_attributes.sensor_resolution = self.calculate_growth_phase(
            1.0, self.age, self.growth_genes.sensor_development_rate)
            
        self.physical_attributes.action_precision = self.calculate_growth_phase(
            1.0, self.age, self.growth_genes.action_refinement_rate)
    
    def _determine_growth_stage(self) -> str:
        """Determine current growth stage based on age"""
        normalized_age = (self.age / self.maturity_age) * 100
        for stage, (min_age, max_age) in self.growth_stages.items():
            if min_age <= normalized_age < max_age:
                return stage
        return "mature"
    
    def get_growth_constraints(self) -> Dict[str, Tuple[float, float]]:
        """Get current physical constraints based on development stage"""
        stage = self._determine_growth_stage()
        current_size = self.physical_attributes.size
        
        return {
            "max_energy_usage": (0, self.physical_attributes.energy_capacity),
            "processing_speed_range": (0, self.physical_attributes.processing_speed),
            "memory_limit": (0, self.physical_attributes.memory_capacity),
            "sensor_resolution_range": (0, self.physical_attributes.sensor_resolution),
            "action_precision_range": (0, self.physical_attributes.action_precision)
        }
    
    def can_perform_action(self, requirements: Dict[str, float]) -> Tuple[bool, str]:
        """Check if physical capabilities meet action requirements"""
        if requirements["energy_usage"] > self.physical_attributes.energy_capacity:
            return False, "Insufficient energy capacity"
            
        if requirements["processing_speed_range"] > self.physical_attributes.processing_speed:
            return False, "Processing speed too low"
            
        return True, "Action permitted"

def create_dna_guide(growth_pattern: str = "balanced") -> DNAGuide:
    """Factory function to create a new DNA guide"""
    guide = DNAGuide()
    guide.initialize_growth_pattern(growth_pattern)
    return guide