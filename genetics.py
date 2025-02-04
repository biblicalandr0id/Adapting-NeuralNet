import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math
import json
import random
import logging
import torch
from .genetic_inheritance import DigitalNucleotide, GeneticTrait

# Remove direct torchvision import
# Instead, use torch.nn.functional for any needed operations

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
    problem_solving: float = 1.0
    
    # Advanced mental traits
    risk_assessment: float = 0.5
    social_awareness: float = 0.5
    innovation_drive: float = 0.7
    curiosity_factor: float = 0.8
    strategic_thinking: float = 0.6


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


class BrainGenetics:
    def __init__(self):
        """Initialize brain-related genetic traits"""
        # Change from random.uniform to float type hints
        self.processing_speed: float = random.uniform(0.1, 1.0)
        self.pattern_recognition: float = random.uniform(0.1, 1.0)
        self.neural_plasticity: float = random.uniform(0.2, 0.8)
        self.learning_rate: float = random.uniform(0.01, 0.1)
        self.memory_capacity: float = random.uniform(0.3, 1.0)
        
    def mutate(self) -> 'BrainGenetics':
        """Create a mutated copy of brain genetics"""
        mutated = BrainGenetics()
        mutation_strength: float = random.uniform(0.8, 1.2)
        
        mutated.processing_speed = max(0.1, min(1.0, self.processing_speed * mutation_strength))
        mutated.pattern_recognition = max(0.1, min(1.0, self.pattern_recognition * mutation_strength))
        mutated.neural_plasticity = max(0.2, min(0.8, self.neural_plasticity * mutation_strength))
        mutated.learning_rate = max(0.01, min(0.1, self.learning_rate * mutation_strength))
        mutated.memory_capacity = max(0.3, min(1.0, self.memory_capacity * mutation_strength))
        
        return mutated


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


def create_genetic_core(seed: Optional[int] = None) -> 'GeneticCore':
    """Factory function to create and initialize genetic core"""
    genetics = GeneticCore()
    genetics.initialize_random_genetics(seed)
    return genetics


class GeneticCore:
    def __init__(self, parent_genes: Dict = None):
        """Initialize genetic core with optional parent genes"""
        self.base_traits = BaseTraits()
        self.mind_genetics = MindGenetics()
        self.brain_genetics = BrainGenetics()
        self.physical_genetics = PhysicalGenetics()
        self.heart_genetics = HeartGenetics()
        self.generation = 0
        self.mutations = []
        self.gene_variance = 0.1
        self.mutation_rate = 0.1
        self.trait_dependencies = self._setup_trait_dependencies()
        self.development_progress = 0.0
        self.stages = {
            'embryo': (0.0, 0.2),
            'infant': (0.2, 0.4),
            'juvenile': (0.4, 0.6),
            'adolescent': (0.6, 0.8),
            'adult': (0.8, 1.0)
        }

        if parent_genes:
            self._inherit_from_parent(parent_genes)
        else:
            self._initialize_random_genes()

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
            processing_speed=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            memory_capacity=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            pattern_recognition=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            learning_rate=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            neural_plasticity=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            decision_speed=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            multi_tasking=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            spatial_awareness=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            temporal_processing=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            error_correction=self.base_traits.resilience *
            np.random.normal(1.0, 0.1)
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
        return mutations

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
            "timestamp": datetime.now().isoformat()
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

    def get_all_traits(self) -> List[Tuple[str, float]]:
        """Get all genetic traits as a list of (name, value) tuples"""
        traits = []
        for genetics_class in [self.physical_genetics, self.brain_genetics, self.mind_genetics]:
            class_name = genetics_class.__class__.__name__.lower().replace('genetics', '')
            for field, value in vars(genetics_class).items():
                traits.append((f"{class_name}.{field}", float(value)))
        return traits


def create_offspring(parent1: GeneticCore, parent2: GeneticCore) -> GeneticCore:
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
