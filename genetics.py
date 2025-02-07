import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from enum import Enum
import math
import json
import random
import logging
import torch
from genetic_inheritance import DigitalNucleotide, GeneticTrait
import uuid
import datetime
import os

if TYPE_CHECKING:
    from embryo_generator import Embryo

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

@dataclass
class BaseTraits:
    resilience: float = 1.0
    adaptability: float = 1.0
    efficiency: float = 1.0
    complexity: float = 1.0
    stability: float = 1.0


@dataclass
class MindGenetics:
    # Basic mental attributes
    creativity: float = 1.0
    learning_efficiency: float = 1.0
    adaptation_rate: float = 1.0
    memory_retention: float = 1.0
    memory_capacity: float = 1.0
    problem_solving: float = 1.0
    strategic_thinking: float = 0.6
    cognitive_growth_rate: float = 1.0  # Added with a default value
    pattern_recognition: float = 1.0  # Added for mind processing
    development_plasticity: float = 0.6  # Added for development

    # Advanced mental traits
    risk_assessment: float = 0.5
    social_awareness: float = 0.5
    innovation_drive: float = 0.7
    curiosity_factor: float = 0.8

    # Make sure we initialize this in GeneticCore
    mutation_history: List[str] = field(default_factory=list)
    adaptation_score: float = 1.0

    def calculate_creativity(self) -> float:
        """Calculate creativity with innovation bonus"""
        base_creativity = self.creativity
        innovation_bonus = self.innovation_drive * 0.3
        curiosity_bonus = self.curiosity_factor * 0.2
        return base_creativity * (1 + innovation_bonus + curiosity_bonus)

    def calculate_learning_capacity(self) -> float:
        """Calculate learning capacity with efficiency bonus"""
        base_capacity = self.learning_efficiency
        retention_bonus = self.memory_retention * 0.25
        return base_capacity * (1 + retention_bonus)

    def calculate_decision_making(self) -> float:
        """Calculate decision making with risk assessment"""
        base_decision = self.strategic_thinking
        risk_bonus = self.risk_assessment * 0.2
        return base_decision * (1 + risk_bonus)

    def calculate_adaptation_rate(self) -> float:
        """Calculate adaptation rate with awareness bonus"""
        base_rate = self.adaptation_rate
        awareness_bonus = self.social_awareness * 0.15
        return base_rate * (1 + awareness_bonus)

    def calculate_memory_capacity(self) -> float:
        """Calculate memory capacity with retention bonus"""
        base_capacity = self.memory_capacity
        retention_bonus = self.memory_retention * 0.2
        return base_capacity * (1 + retention_bonus)


@dataclass
class HeartGenetics:
    """Genetic traits for heart and emotional systems"""
    base_rhythm: float = 1.0
    stress_sensitivity: float = 1.0
    resilience: float = 1.0
    recovery_resilience: float = 1.0
    adaptation_rate: float = 1.0
    trust_baseline: float = 0.5  # Initial trust tendency
    emotional_capacity: float = 1.0
    security_sensitivity: float = 1.0  # How strongly agent responds to threats


@dataclass
class BrainGenetics:
    # Base attributes
    processing_speed: float = 1.0
    pattern_recognition: float = 1.0
    neural_plasticity: float = 0.5
    learning_rate: float = 0.05
    memory_capacity: float = 0.7
    decision_speed: float = 1.0  # Added missing attribute
    multi_tasking: float = 1.0   # Added missing attribute
    spatial_awareness: float = 1.0  # Added missing attribute
    temporal_processing: float = 1.0  # Added missing attribute
    error_correction: float = 1.0  # Added missing attribute
    
    # Track mutations and adaptations
    mutation_history: List[str] = field(default_factory=list)
    adaptation_score: float = 1.0

    def mutate(self) -> 'BrainGenetics':
        """Create a mutated copy with tracking"""
        mutated = BrainGenetics()
        mutation_strength = random.uniform(0.8, 1.2)
        mutations = []
        
        # Apply mutations with tracking
        for field in self.__annotations__:
            if field not in ['mutation_history', 'adaptation_score']:
                old_value = getattr(self, field)
                new_value = max(0.1, min(1.0, old_value * mutation_strength))
                setattr(mutated, field, new_value)
                
                if abs(new_value - old_value) > 0.1:
                    mutations.append(f"{field}: {old_value:.2f} -> {new_value:.2f}")
        
        mutated.mutation_history = self.mutation_history + mutations
        return mutated
    
    def inherit_from(self, parent: 'BrainGenetics', strength: float = 0.7) -> None:
        """Inherit traits from parent with variable strength"""
        for field in self.__annotations__:
            if field not in ['mutation_history', 'adaptation_score']:
                parent_value = getattr(parent, field)
                current_value = getattr(self, field)
                inherited = (parent_value * strength) + (current_value * (1 - strength))
                setattr(self, field, inherited)

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate overall brain performance metrics"""
        return {
            'processing_efficiency': self.calculate_processing_power() * self.calculate_pattern_recognition(),
            'learning_potential': self.calculate_plasticity() * self.calculate_learning_rate(),
            'memory_effectiveness': self.calculate_memory_capacity() * self.pattern_recognition,
            'adaptation_capability': self.neural_plasticity * self.learning_rate
        }

    def calculate_plasticity(self) -> float:
        """Calculate neural plasticity with learning bonus"""
        base_plasticity = self.neural_plasticity
        learning_bonus = self.learning_rate * 0.2
        return base_plasticity * (1 + learning_bonus)

    def calculate_learning_rate(self) -> float:
        """Calculate learning rate with plasticity bonus"""
        base_rate = self.learning_rate
        plasticity_bonus = self.neural_plasticity * 0.1
        return base_rate * (1 + plasticity_bonus)

    def calculate_memory_capacity(self) -> float:
        """Calculate memory capacity with recognition bonus"""
        base_capacity = self.memory_capacity
        recognition_bonus = self.pattern_recognition * 0.15
        return base_capacity * (1 + recognition_bonus)

    def calculate_processing_power(self) -> float:
        """Calculate processing power with speed bonus"""
        base_power = self.processing_speed
        recognition_bonus = self.pattern_recognition * 0.25
        return base_power * (1 + recognition_bonus)

    def calculate_pattern_recognition(self) -> float:
        """Calculate pattern recognition with efficiency bonus"""
        base_recognition = self.pattern_recognition
        efficiency_bonus = self.processing_speed * 0.15
        return base_recognition * (1 + efficiency_bonus)


@dataclass
class PhysicalGenetics:
    # Basic physical attributes
    size: float = 1.0
    speed: float = 1.0
    strength: float = 1.0
    energy_efficiency: float = 1.0
    sensor_sensitivity: float = 1.0
    
    # Advanced physical traits
    regeneration_rate: float = 0.5
    immune_system: float = 0.5
    metabolic_rate: float = 1.0
    longevity_factor: float = 1.0
    adaptation_speed: float = 0.5
    action_precision: float = 1.0

    def calculate_metabolism(self) -> float:
        """Calculate metabolism rate with efficiency bonus"""
        base_rate = self.metabolic_rate
        efficiency_bonus = self.energy_efficiency * 0.2
        return base_rate * (1 + efficiency_bonus)
    
    def calculate_efficiency(self) -> float:
        """Calculate energy efficiency with size penalty"""
        base_efficiency = self.energy_efficiency
        size_penalty = max(0, (self.size - 1.0) * 0.1)
        return base_efficiency * (1 - size_penalty)
    
    def calculate_stamina(self) -> float:
        """Calculate stamina with metabolism and strength"""
        base_stamina = (self.strength + self.metabolic_rate) / 2
        efficiency_bonus = self.energy_efficiency * 0.15
        return base_stamina * (1 + efficiency_bonus)
    
    def calculate_adaptation_rate(self) -> float:
        """Calculate physical adaptation rate"""
        base_rate = self.adaptation_speed
        regen_bonus = self.regeneration_rate * 0.1
        immune_bonus = self.immune_system * 0.1
        return base_rate * (1 + regen_bonus + immune_bonus)


@dataclass
class EmbryoGenetics:
    """Genetics for embryonic development and inheritance"""
    development_rate: float = 1.0
    mutation_chance: float = 0.05
    inheritance_factor: float = 0.7
    trait_stability: float = 0.8
    neural_plasticity: float = 0.6
    cognitive_growth_rate: float = 1.0
    development_plasticity: float = 0.6  # Added missing trait
    
    def calculate_mutation_rate(self) -> float:
        return self.mutation_chance * (1 - self.trait_stability)
        
    def calculate_inheritance(self) -> float:
        return self.inheritance_factor * self.trait_stability
        
    def calculate_development_speed(self) -> float:
        return self.development_rate * self.neural_plasticity
        
    def calculate_trait_stability(self) -> float:
        return self.trait_stability * (1 + self.inheritance_factor * 0.2)


def create_genetic_core(seed: Optional[int] = None) -> 'GeneticCore':
    """Factory function to create and initialize genetic core"""
    genetics = GeneticCore()
    genetics.initialize_random_genetics(seed)
    return genetics


class EmergentTrait:
    def __init__(self, name: str, value: float, influence_map: Dict[str, float]):
        self.name = name
        self.value = value
        self.influence_map = influence_map  # How this trait affects others
        self.generation_discovered = 0
        self.stability = 0.5  # How stable this trait is

class GeneticCore:
    def __init__(self):
        # Add validation
        self.brain_genetics = BrainGenetics()
        self.mind_genetics = MindGenetics()
        self.physical_genetics = PhysicalGenetics()
        self.embryo_genetics = EmbryoGenetics(
            development_rate=random.uniform(0.8, 1.2),
            mutation_chance=random.uniform(0.01, 0.1),
            inheritance_factor=random.uniform(0.5, 0.9),
            trait_stability=random.uniform(0.7, 0.9),
            neural_plasticity=random.uniform(0.4, 0.8),
            cognitive_growth_rate=random.uniform(0.8, 1.5),
            development_plasticity=random.uniform(0.4, 0.8)
        )
        
        # Initialize development and mutation variables
        self.development_progress = 0.0
        self.mutation_rate = 0.05  # Base mutation rate
        
        # Validate initialization
        self._validate_genetics()
        
        # Initialize traits
        self.traits = self._initialize_traits()
        self.emergent_traits: Dict[str, EmergentTrait] = {}
        self.trait_mutation_chance = 0.05
        
    def _validate_genetics(self):
        """Validate all genetics are properly initialized"""
        if not hasattr(self.mind_genetics, 'creativity'):
            logger.error("Mind genetics missing creativity trait")
            raise ValueError("Mind genetics initialization failed")
            
        if not hasattr(self.brain_genetics, 'pattern_recognition'):
            logger.error("Brain genetics missing pattern recognition trait")
            raise ValueError("Brain genetics initialization failed")

    def _initialize_traits(self):
        return {
            'neural': self._calculate_neural_traits(),
            'processing': self._calculate_processing_traits(), 
            'physical': self._calculate_physical_traits(),
            'development': self._calculate_development_traits()
        }
    
    def _calculate_neural_traits(self) -> Dict:
        """Calculate neural network traits from genetics"""
        return {
            'plasticity': self.brain_genetics.calculate_plasticity(),
            'learning_rate': self.brain_genetics.calculate_learning_rate(),
            'memory_capacity': self.brain_genetics.calculate_memory_capacity(),
            'processing_power': self.brain_genetics.calculate_processing_power()
        }

    def _calculate_processing_traits(self) -> Dict:
        """Calculate mental processing traits from genetics"""
        return {
            'creativity': self.mind_genetics.calculate_creativity(),
            'learning_capacity': self.mind_genetics.calculate_learning_capacity(),
            'decision_making': self.mind_genetics.calculate_decision_making(),
            'pattern_recognition': self.brain_genetics.calculate_pattern_recognition(),  # From brain not mind
            'problem_solving': self.mind_genetics.problem_solving,
            'memory_retention': self.mind_genetics.memory_retention
        }

    def _calculate_physical_traits(self) -> Dict:
        """Calculate physical traits from genetics"""
        return {
            'metabolism_rate': self.physical_genetics.calculate_metabolism(),
            'energy_efficiency': self.physical_genetics.calculate_efficiency(),
            'stamina': self.physical_genetics.calculate_stamina(),
            'adaptation_rate': self.physical_genetics.calculate_adaptation_rate()
        }

    def _calculate_development_traits(self) -> Dict:
        """Calculate development and inheritance traits"""
        return {
            'mutation_rate': self.embryo_genetics.calculate_mutation_rate(),
            'inheritance_strength': self.embryo_genetics.calculate_inheritance(),
            'development_speed': self.embryo_genetics.calculate_development_speed(),
            'trait_stability': self.embryo_genetics.calculate_trait_stability()
        }

    def initialize_random_genetics(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        def random_trait() -> float:
            return np.random.normal(1.0, 0.2)

        self.base_traits = BaseTraits(
            resilience=random_trait(),
            adaptability=random_trait(),
            efficiency=random_trait(),
            complexity=random_trait(),
            stability=random_trait()
        )

        self._initialize_mind_genetics()
        self._initialize_heart_genetics()
        self._initialize_brain_genetics()
        self._initialize_physical_genetics()

    def _initialize_mind_genetics(self) -> None:
        self.mind_genetics = MindGenetics(
            creativity=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            learning_efficiency=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            adaptation_rate=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            memory_retention=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            problem_solving=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            risk_assessment=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            social_awareness=self.base_traits.resilience *
            np.random.normal(1.0, 0.1),
            innovation_drive=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            curiosity_factor=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            strategic_thinking=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            cognitive_growth_rate=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            pattern_recognition=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            development_plasticity=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1)
        )

    def _initialize_heart_genetics(self) -> None:
        self.heart_genetics = HeartGenetics(
            base_rhythm=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            stress_sensitivity=self.base_traits.resilience *
            np.random.normal(1.0, 0.1),
            resilience=self.base_traits.resilience *
            np.random.normal(1.0, 0.1),
            recovery_resilience=self.base_traits.resilience *
            np.random.normal(1.0, 0.1),
            adaptation_rate=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            trust_baseline=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            emotional_capacity=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            security_sensitivity=self.base_traits.resilience *
            np.random.normal(1.0, 0.1)
        )

    def _initialize_brain_genetics(self) -> None:
        self.brain_genetics = BrainGenetics(
            processing_speed=self.base_traits.efficiency * np.random.normal(1.0, 0.1),
            pattern_recognition=self.base_traits.complexity * np.random.normal(1.0, 0.1),
            neural_plasticity=self.base_traits.stability * np.random.normal(1.0, 0.1),
            learning_rate=self.base_traits.adaptability * np.random.normal(1.0, 0.1),
            memory_capacity=self.base_traits.stability * np.random.normal(1.0, 0.1),
            decision_speed=self.base_traits.efficiency * np.random.normal(1.0, 0.1),
            multi_tasking=self.base_traits.complexity * np.random.normal(1.0, 0.1),
            spatial_awareness=self.base_traits.adaptability * np.random.normal(1.0, 0.1),
            temporal_processing=self.base_traits.stability * np.random.normal(1.0, 0.1),
            error_correction=self.base_traits.resilience * np.random.normal(1.0, 0.1)
        )

    def _initialize_physical_genetics(self) -> None:
        self.physical_genetics = PhysicalGenetics(
            size=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            speed=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            strength=self.base_traits.resilience *
            np.random.normal(1.0, 0.1),
            energy_efficiency=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            sensor_sensitivity=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            regeneration_rate=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            immune_system=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            metabolic_rate=self.base_traits.resilience *
            np.random.normal(1.0, 0.1),
            longevity_factor=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            adaptation_speed=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            action_precision=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1)
        )

    def get_mind_parameters(self) -> Dict[str, float]:
        stage_modifier = self._get_stage_modifier()
        return {
            "creativity": self.mind_genetics.creativity * stage_modifier,
            "learning_rate": self.mind_genetics.learning_efficiency * stage_modifier,
            "memory_retention": self.mind_genetics.memory_retention * 1000,
            "adaptation_rate": self.mind_genetics.adaptation_rate * stage_modifier,
            "problem_solving": self.mind_genetics.problem_solving * stage_modifier,
            "risk_assessment": self.mind_genetics.risk_assessment * stage_modifier,
            "social_awareness": self.mind_genetics.social_awareness * stage_modifier,
            "innovation_drive": self.mind_genetics.innovation_drive * stage_modifier,
            "curiosity_factor": self.mind_genetics.curiosity_factor * stage_modifier,
            "strategic_thinking": self.mind_genetics.strategic_thinking * stage_modifier
        }

    def get_heart_parameters(self) -> Dict[str, float]:
        return {
            "trust_threshold": max(0.3, min(0.9, self.heart_genetics.trust_baseline)),
            "security_threshold": self.heart_genetics.security_sensitivity,
            "adaptation_rate": self.heart_genetics.adaptation_rate,
            "check_interval": max(0.1, 1.0 / self.heart_genetics.integrity_check_frequency),
            "recovery_factor": self.heart_genetics.recovery_resilience
        }

    def get_brain_parameters(self) -> Dict[str, float]:
        return {
            "processing_speed": self.brain_genetics.processing_speed,
            "memory_capacity": self.brain_genetics.memory_capacity * 100,
            "pattern_recognition": self.brain_genetics.pattern_recognition * 100,
            "learning_rate": self.brain_genetics.learning_rate * 100,
            "neural_plasticity": self.brain_genetics.neural_plasticity * 100,
            "decision_speed": self.brain_genetics.decision_speed * 100,
            "multi_tasking": self.brain_genetics.multi_tasking * 100,
            "spatial_awareness": self.brain_genetics.spatial_awareness * 100,
            "temporal_processing": self.brain_genetics.temporal_processing * 100,
            "error_correction": self.brain_genetics.error_correction * 100
        }

    def get_physical_parameters(self) -> Dict[str, float]:
        stage_modifier = self._get_stage_modifier()
        return {
            "size": self.physical_genetics.size * stage_modifier,
            "speed": self.physical_genetics.speed * stage_modifier,
            "strength": self.physical_genetics.strength * stage_modifier,
            "energy_efficiency": self.physical_genetics.energy_efficiency * stage_modifier,
            "sensor_sensitivity": self.physical_genetics.sensor_sensitivity * stage_modifier,
            "regeneration_rate": self.physical_genetics.regeneration_rate * stage_modifier,
            "immune_system": self.physical_genetics.immune_system * stage_modifier,
            "metabolic_rate": self.physical_genetics.metabolic_rate * stage_modifier,
            "longevity_factor": self.physical_genetics.longevity_factor * stage_modifier,
            "adaptation_speed": self.physical_genetics.adaptation_speed * stage_modifier
        }

    def _get_stage_modifier(self) -> float:
        for stage, (min_prog, max_prog) in self.stages.items():
            if min_prog <= self.development_progress < max_prog:
                stage_progress = (self.development_progress -
                                  min_prog) / (max_prog - min_prog)
                return 1.0 + math.log(1 + stage_progress)
        return 1.0

    def update_development(self, time_delta: float) -> None:
        growth_rate = (self.base_traits.adaptability *
                       self.physical_genetics.size *
                       0.1)
        self.development_progress = min(1.0,
                                        self.development_progress + time_delta * growth_rate)

        if random.random() < self.mutation_rate * time_delta:
            self._apply_random_mutation()

    def _apply_random_mutation(self) -> None:
        categories = ['base', 'mind', 'heart', 'brain', 'physical']
        category = random.choice(categories)

        mutation_strength = np.random.normal(0, 0.1)

        if category == 'base':
            traits = vars(self.base_traits)
            trait = random.choice(list(traits.keys()))
            current_value = getattr(self.base_traits, trait)
            setattr(self.base_traits, trait, max(
                0.1, current_value + mutation_strength))
            logging.info(
                f"Applied mutation to base.{trait}: {mutation_strength:+.3f}")

        elif category == 'mind':
            traits = vars(self.mind_genetics)
            trait = random.choice(list(traits.keys()))
            current_value = getattr(self.mind_genetics, trait)
            setattr(self.mind_genetics, trait, max(
                0.1, current_value + mutation_strength))
            logging.info(
                f"Applied mutation to mind.{trait}: {mutation_strength:+.3f}")

        elif category == 'heart':
            traits = vars(self.heart_genetics)
            trait = random.choice(list(traits.keys()))
            current_value = getattr(self.heart_genetics, trait)
            setattr(self.heart_genetics, trait, max(
                0.1, current_value + mutation_strength))
            logging.info(
                f"Applied mutation to heart.{trait}: {mutation_strength:+.3f}")

        elif category == 'brain':
            traits = vars(self.brain_genetics)
            trait = random.choice(list(traits.keys()))
            current_value = getattr(self.brain_genetics, trait)
            setattr(self.brain_genetics, trait, max(
                0.1, current_value + mutation_strength))
            logging.info(
                f"Applied mutation to brain.{trait}: {mutation_strength:+.3f}")

        elif category == 'physical':
            traits = vars(self.physical_genetics)
            trait = random.choice(list(traits.keys()))
            current_value = getattr(self.physical_genetics, trait)
            setattr(self.physical_genetics, trait, max(
                0.1, current_value + mutation_strength))
            logging.info(
                f"Applied mutation to physical.{trait}: {mutation_strength:+.3f}")

    def save_genetics(self, file_path: str) -> None:
        """Save genetic configuration to file"""
        genetic_data = {
            "base_traits": vars(self.base_traits),
            "mind_genetics": vars(self.mind_genetics),
            "heart_genetics": vars(self.heart_genetics),
            "brain_genetics": vars(self.brain_genetics),
            "physical_genetics": vars(self.physical_genetics),
            "development_progress": self.development_progress
        }

        with open(file_path, 'w') as f:
            json.dump(genetic_data, f, indent=2)

    def load_genetics(self, file_path: str) -> None:
        """Load genetic configuration from file"""
        with open(file_path, 'r') as f:
            genetic_data = json.load(f)

        self.base_traits = BaseTraits(**genetic_data["base_traits"])
        self.mind_genetics = MindGenetics(**genetic_data["mind_genetics"])
        self.heart_genetics = HeartGenetics(**genetic_data["heart_genetics"])
        self.brain_genetics = BrainGenetics(**genetic_data["brain_genetics"])
        self.physical_genetics = PhysicalGenetics(
            **genetic_data["physical_genetics"])
        self.development_progress = genetic_data["development_progress"]

    @staticmethod
    def create_genetic_core(seed: Optional[int] = None) -> 'GeneticCore':
        """Factory function to create and initialize genetic core"""
        genetics = GeneticCore()
        genetics.initialize_random_genetics(seed)
        return genetics

    def _initialize_random_genes(self):
        """Initialize random genetic traits with gaussian distribution"""
        self.physical_genetics = PhysicalGenetics(
            **{field: max(0.1, min(2.0, random.gauss(1.0, 0.3))) 
               for field in PhysicalGenetics.__annotations__}
        )
        self.brain_genetics = BrainGenetics(
            **{field: max(0.1, min(2.0, random.gauss(1.0, 0.3))) 
               for field in BrainGenetics.__annotations__}
        )
        self.mind_genetics = MindGenetics(
            **{field: max(0.1, min(2.0, random.gauss(1.0, 0.3))) 
               for field in MindGenetics.__annotations__}
        )

    def _setup_trait_dependencies(self) -> Dict:
        """Setup genetic trait dependencies for realistic evolution"""
        return {
            'energy_efficiency': ['metabolic_rate', 'size'],
            'learning_efficiency': ['neural_plasticity', 'memory_capacity'],
            'adaptation_rate': ['innovation_drive', 'neural_plasticity'],
            'problem_solving': ['pattern_recognition', 'strategic_thinking'],
            'creativity': ['innovation_drive', 'neural_plasticity', 'curiosity_factor']
        }

    def mutate(self) -> List[str]:
        """Apply mutations with tracking"""
        mutations = []
        for genetics_class in [self.physical_genetics, self.brain_genetics, self.mind_genetics]:
            for field in genetics_class.__annotations__:
                if random.random() < self.mutation_rate:
                    old_value = getattr(genetics_class, field)
                    mutation = random.gauss(0, self.gene_variance)
                    new_value = max(0.1, min(2.0, old_value + mutation))
                    setattr(genetics_class, field, new_value)
                    mutations.append(f"{field}: {old_value:.2f} -> {new_value:.2f}")
        
        self.mutations.extend(mutations)
        
        # Emergent trait mutation
        if random.random() < self.trait_mutation_chance:
            new_trait = self._generate_emergent_trait()
            if new_trait:
                self.emergent_traits[new_trait.name] = new_trait
                mutations.append(f"Developed new trait: {new_trait.name}")
        
        return mutations

    def _generate_emergent_trait(self) -> Optional[EmergentTrait]:
        """Generate a completely new trait based on existing traits"""
        base_traits = {
            'creativity': self.mind_genetics.creativity,
            'processing_speed': self.brain_genetics.processing_speed,
            'energy_efficiency': self.physical_genetics.energy_efficiency,
            'neural_plasticity': self.brain_genetics.neural_plasticity
        }
        
        # Generate new trait name using combinations of existing traits
        trait_aspects = [
            ['hyper', 'meta', 'quantum', 'parallel'],
            ['processing', 'adaptation', 'synthesis', 'cognition'],
            ['resonance', 'alignment', 'coherence', 'integration']
        ]
        
        new_name = "_".join([random.choice(aspect) for aspect in trait_aspects])
        
        # Create influence map - how this trait affects others
        influence_map = {}
        for trait, value in base_traits.items():
            if random.random() < 0.3:  # 30% chance to influence each trait
                influence_map[trait] = random.uniform(-0.5, 0.5)
        
        # Initial value based on parent traits
        initial_value = sum(base_traits.values()) / len(base_traits)
        initial_value *= random.uniform(0.8, 1.2)  # Add some randomness
        
        return EmergentTrait(
            name=new_name,
            value=initial_value,
            influence_map=influence_map
        )

    def apply_emergent_traits(self):
        """Apply effects of emergent traits to base traits"""
        for trait in self.emergent_traits.values():
            for target_trait, influence in trait.influence_map.items():
                # Apply influence based on the trait's stability
                effect = influence * trait.value * trait.stability
                
                if hasattr(self.mind_genetics, target_trait):
                    current = getattr(self.mind_genetics, target_trait)
                    setattr(self.mind_genetics, target_trait, current + effect)
                elif hasattr(self.brain_genetics, target_trait):
                    current = getattr(self.brain_genetics, target_trait)
                    setattr(self.brain_genetics, target_trait, current + effect)
                elif hasattr(self.physical_genetics, target_trait):
                    current = getattr(self.physical_genetics, target_trait)
                    setattr(self.physical_genetics, target_trait, current + effect)

    def combine_genes(self, other_core: 'GeneticCore') -> 'GeneticCore':
        """Combine genes using digital DNA inheritance system"""
        offspring = GeneticCore()
        
        for genetics_type in ['physical_genetics', 'brain_genetics', 'mind_genetics']:
            current_traits = getattr(self, genetics_type)
            other_traits = getattr(other_core, genetics_type)
            
            for trait_name in current_traits.__annotations__:
                # Create digital DNA sequence for each trait
                parent1_trait = GeneticTrait(
                    trait_name,
                    getattr(current_traits, trait_name),
                    is_dominant=random.random() > 0.5
                )
                parent2_trait = GeneticTrait(
                    trait_name,
                    getattr(other_traits, trait_name),
                    is_dominant=random.random() > 0.5
                )
                
                # Create offspring trait using digital inheritance
                offspring_trait = GeneticTrait.from_parents(parent1_trait, parent2_trait)
                
                # Apply possible mutations
                if random.random() < self.mutation_rate:
                    for nucleotide in offspring_trait.dna_sequence:
                        if random.random() < 0.1:  # 10% chance per nucleotide
                            nucleotide.mutate()
                
                # Set the trait value on offspring
                offspring_genetics = getattr(offspring, genetics_type)
                setattr(offspring_genetics, trait_name, offspring_trait.value)
        
        # Track inheritance and generation
        offspring.generation = max(self.generation, other_core.generation) + 1
        
        # Create conception record with digital DNA information
        conception_data = {
            "id": str(uuid.uuid4())[:8],
            "parents": {
                "parent1_id": str(id(self))[:8],
                "parent2_id": str(id(other_core))[:8]
            },
            "dna_sequences": {},
            "generation": offspring.generation,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Record the event
        self._record_conception(conception_data)
        
        return offspring

    def _record_conception(self, data: Dict) -> None:
        """Record conception event with DNA information"""
        record_dir = "genetic_records"
        os.makedirs(record_dir, exist_ok=True)
        
        filepath = os.path.join(
            record_dir,
            f"conception_{data['id']}.json"
        )
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def inherit_from(self, parent_core: 'GeneticCore', mutation_rate: Optional[float] = None) -> None:
        """Inherit traits from a parent core with possible mutations"""
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
            
        combined = self.combine_genes(parent_core)
        
        # Apply combined genes to each genetics class
        for genetics_type, traits in combined.items():
            genetics_class = getattr(self, genetics_type)
            for field, value in traits.items():
                setattr(genetics_class, field, value)
        
        # Increment generation
        self.generation = parent_core.generation + 1

    def calculate_fitness_score(self) -> float:
        """Calculate overall fitness based on genetic traits"""
        physical_score = sum([
            getattr(self.physical_genetics, trait) 
            for trait in self.physical_genetics.__annotations__
        ]) / len(self.physical_genetics.__annotations__)
        
        brain_score = sum([
            getattr(self.brain_genetics, trait) 
            for trait in self.brain_genetics.__annotations__
        ]) / len(self.brain_genetics.__annotations__)
        
        mind_score = sum([
            getattr(self.mind_genetics, trait) 
            for trait in self.mind_genetics.__annotations__
        ]) / len(self.mind_genetics.__annotations__)
        
        return (physical_score * 0.3 + brain_score * 0.3 + mind_score * 0.4)

    def get_all_traits_list(self) -> List[Tuple[str, float]]:
        """Get all genetic traits as a list of (name, value) tuples"""
        traits = []
        for genetics_class in [self.physical_genetics, self.brain_genetics, self.mind_genetics]:
            class_name = genetics_class.__class__.__name__.lower().replace('genetics', '')
            for field, value in vars(genetics_class).items():
                traits.append((f"{class_name}.{field}", float(value)))
        return traits

    def create_offspring(self) -> 'GeneticCore':
        """Create a new genetic core with mutations"""
        offspring = GeneticCore()
        
        # Inherit and mutate embryo genetics
        for field in offspring.embryo_genetics.__dataclass_fields__:
            parent_value = getattr(self.embryo_genetics, field)
            mutation = random.uniform(-0.1, 0.1)
            new_value = max(0.1, min(2.0, parent_value + mutation))
            setattr(offspring.embryo_genetics, field, new_value)
            
        # Inherit other genetics similarly...
        return offspring

    def get_all_traits(self) -> Dict:
        """Get all genetic traits as a dictionary"""
        return {
            'embryo': {
                field: getattr(self.embryo_genetics, field)
                for field in self.embryo_genetics.__dataclass_fields__
            },
            'mind': {
                field: getattr(self.mind_genetics, field)
                for field in self.mind_genetics.__dataclass_fields__
            },
            'brain': {
                field: getattr(self.brain_genetics, field)
                for field in self.brain_genetics.__dataclass_fields__
            },
            'heart': {
                field: getattr(self.heart_genetics, field)
                for field in self.heart_genetics.__dataclass_fields__
            }
        }

    def calculate_potential(self) -> float:
        """Calculate overall genetic potential"""
        potentials = {
            'embryo': sum(getattr(self.embryo_genetics, f) 
                         for f in self.embryo_genetics.__dataclass_fields__),
            'mind': sum(getattr(self.mind_genetics, f) 
                       for f in self.mind_genetics.__dataclass_fields__),
            'brain': sum(getattr(self.brain_genetics, f) 
                        for f in self.brain_genetics.__dataclass_fields__),
            'heart': sum(getattr(self.heart_genetics, f) 
                        for f in self.heart_genetics.__dataclass_fields__)
        }
        return sum(potentials.values()) / len(potentials)

    def _develop_genetic_core(self, embryo: 'Embryo') -> 'GeneticCore':
        """Develop genetic core from embryo genetics"""
        genetic_core = GeneticCore()
        
        # Transfer embryo genetics to core systems
        genetic_core.mind_genetics.cognitive_growth_rate = embryo.genetics.cognitive_growth_rate
        genetic_core.brain_genetics.neural_plasticity = embryo.genetics.neural_plasticity
        genetic_core.mind_genetics.memory_capacity = embryo.genetics.memory_capacity
        genetic_core.mind_genetics.adaptation_rate = embryo.genetics.adaptation_rate
        
        return genetic_core

def create_offspring(parent1: 'GeneticCore', parent2: 'GeneticCore') -> 'GeneticCore':
    """Convenience function to create offspring from two parents"""
    # Verify parents are valid
    if not isinstance(parent1, GeneticCore) or not isinstance(parent2, GeneticCore):
        raise ValueError("Both parents must be GeneticCore instances")
        
    # Create offspring through genetic combination
    offspring = parent1.combine_genes(parent2)
    
    # Apply random mutations to each genetics class
    if random.random() < 0.1:  # 10% mutation chance
        offspring.mutate()  # Use existing mutation system
    
    return offspring
pass

# genetics.py

@dataclass
class GeneticTraits:
    mind_genetics: MindGenetics
    brain_genetics: BrainGenetics
    physical_genetics: PhysicalGenetics
    heart_genetics: HeartGenetics

@dataclass 
class Lineage:
    generation: int
    parent_id: Optional[str]
    birth_time: float
    genetic_heritage: List[str]
    mutations: List[Dict]  
    achievements: List[Dict]

class GeneticBehavior:
    """Handles genetic influence on behavior"""
    def __init__(self, genetic_core: GeneticCore):
        self.genetic_core = genetic_core
        
    def calculate_success_probability(self, action: str, structural_integrity: float) -> float:
        adaptation_rate = self.genetic_core.mind_genetics.adaptation_rate
        precision = self.genetic_core.physical_genetics.action_precision
        learning_efficiency = self.genetic_core.mind_genetics.learning_efficiency
        return float(min(1.0, (adaptation_rate + precision + learning_efficiency) / 3.0))
        
    def calculate_energy_cost(self, action: str, base_cost: float) -> float:
        energy_efficiency = self.genetic_core.physical_genetics.energy_efficiency
        metabolic_rate = self.genetic_core.physical_genetics.metabolic_rate
        return base_cost * (1.0 / energy_efficiency) * metabolic_rate