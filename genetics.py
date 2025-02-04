import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math
import json
import random
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class BaseTraits:
    resilience: float = 1.0
    adaptability: float = 1.0
    efficiency: float = 1.0
    complexity: float = 1.0
    stability: float = 1.0


@dataclass
class MindGenetics:
    cognitive_growth_rate: float = 1.0
    learning_efficiency: float = 1.0
    memory_capacity: float = 1.0
    neural_plasticity: float = 1.0
    pattern_recognition: float = 1.0


@dataclass
class HeartGenetics:
    trust_baseline: float = 0.5
    security_sensitivity: float = 1.0
    adaptation_rate: float = 1.0
    integrity_check_frequency: float = 1.0
    recovery_resilience: float = 1.0


@dataclass
class BrainGenetics:
    processing_speed: float = 1.0
    emotional_stability: float = 1.0
    focus_capacity: float = 1.0
    ui_responsiveness: float = 1.0
    interaction_capability: float = 1.0


@dataclass
class PhysicalGenetics:
    growth_rate: float = 1.0
    energy_efficiency: float = 1.0
    structural_integrity: float = 1.0
    sensor_sensitivity: float = 1.0
    action_precision: float = 1.0


class GeneticCore:
    def __init__(self):
        self.base_traits = BaseTraits()
        self.mind_genetics = MindGenetics()
        self.heart_genetics = HeartGenetics()
        self.brain_genetics = BrainGenetics()
        self.physical_genetics = PhysicalGenetics()

        self.stages = {
            "embryonic": (0, 0.25),
            "juvenile": (0.25, 0.5),
            "adolescent": (0.5, 0.75),
            "mature": (0.75, 1.0)
        }

        self.development_progress = 0.0
        self.mutation_rate = 0.01

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
            cognitive_growth_rate=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            learning_efficiency=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            memory_capacity=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            neural_plasticity=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            pattern_recognition=self.base_traits.complexity *
            np.random.normal(1.0, 0.1)
        )

    def _initialize_heart_genetics(self) -> None:
        self.heart_genetics = HeartGenetics(
            trust_baseline=0.5 * self.base_traits.stability,
            security_sensitivity=self.base_traits.resilience *
            np.random.normal(1.0, 0.1),
            adaptation_rate=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            integrity_check_frequency=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            recovery_resilience=self.base_traits.resilience *
            np.random.normal(1.0, 0.1)
        )

    def _initialize_brain_genetics(self) -> None:
        self.brain_genetics = BrainGenetics(
            processing_speed=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            emotional_stability=self.base_traits.stability *
            np.random.normal(1.0, 0.1),
            focus_capacity=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            ui_responsiveness=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            interaction_capability=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1)
        )

    def _initialize_physical_genetics(self) -> None:
        self.physical_genetics = PhysicalGenetics(
            growth_rate=self.base_traits.adaptability *
            np.random.normal(1.0, 0.1),
            energy_efficiency=self.base_traits.efficiency *
            np.random.normal(1.0, 0.1),
            structural_integrity=self.base_traits.resilience *
            np.random.normal(1.0, 0.1),
            sensor_sensitivity=self.base_traits.complexity *
            np.random.normal(1.0, 0.1),
            action_precision=self.base_traits.stability *
            np.random.normal(1.0, 0.1)
        )

    def get_mind_parameters(self) -> Dict[str, float]:
        stage_modifier = self._get_stage_modifier()
        return {
            "growth_rate": self.mind_genetics.cognitive_growth_rate * stage_modifier,
            "learning_rate": self.mind_genetics.learning_efficiency * stage_modifier,
            "memory_limit": self.mind_genetics.memory_capacity * 1000,
            "adaptation_threshold": 0.5 / self.mind_genetics.neural_plasticity,
            "pattern_recognition_threshold": 0.3 / self.mind_genetics.pattern_recognition
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
            "emotional_variance": 1.0 / self.brain_genetics.emotional_stability,
            "focus_duration": self.brain_genetics.focus_capacity * 100,
            "ui_update_interval": max(0.1, 1.0 / self.brain_genetics.ui_responsiveness),
            "interaction_threshold": 0.5 / self.brain_genetics.interaction_capability
        }

    def get_physical_parameters(self) -> Dict[str, float]:
        stage_modifier = self._get_stage_modifier()
        return {
            "size_multiplier": self.physical_genetics.growth_rate * stage_modifier,
            "energy_efficiency": self.physical_genetics.energy_efficiency,
            "structural_threshold": self.physical_genetics.structural_integrity,
            "sensor_resolution": self.physical_genetics.sensor_sensitivity,
            "action_precision": self.physical_genetics.action_precision
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
                       self.physical_genetics.growth_rate *
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

    def create_genetic_core(seed: Optional[int] = None) -> 'GeneticCore':
        """Factory function to create and initialize genetic core"""
        genetics = GeneticCore()
        genetics.initialize_random_genetics(seed)
        return genetics
