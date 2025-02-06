from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class NucleotideMutation:
    """Track mutations in nucleotides"""
    generation: int = 0
    previous_value: int = 0
    new_value: int = 0
    mutation_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    neural_activity: float = 0.0

class DigitalNucleotide:
    def __init__(self, value: Optional[int] = None):
        self.value = value if value is not None else random.randint(0, 3)
        self.generation = 0
        self.mutation_history: List[NucleotideMutation] = []
        self.creation_time = datetime.now()

    def mutate(self) -> 'DigitalNucleotide':
        old_value = self.value
        mutation_type = random.random()
        
        if mutation_type < 0.7:  # Point mutation
            self.value = random.randint(0, 3)
            mutation_name = 'point'
        elif mutation_type < 0.85:  # Bit flip
            self.value = self.value ^ (0b01 if random.random() < 0.5 else 0b10)
            mutation_name = 'bit-flip'
        else:  # Complement mutation
            self.value = self.value ^ 0b11
            mutation_name = 'complement'
        
        self.mutation_history.append(
            NucleotideMutation(
                generation=self.generation,
                previous_value=old_value,
                new_value=self.value,
                mutation_type=mutation_name
            )
        )
        self.generation += 1
        return self

    def get_mutation_stats(self) -> Dict[str, int]:
        """Get statistics about mutations"""
        return {
            'total_mutations': len(self.mutation_history),
            'point_mutations': sum(1 for m in self.mutation_history if m.mutation_type == 'point'),
            'bit_flips': sum(1 for m in self.mutation_history if m.mutation_type == 'bit-flip'),
            'complements': sum(1 for m in self.mutation_history if m.mutation_type == 'complement')
        }

@dataclass
class TraitExpression:
    """Controls how traits are expressed"""
    strength: float = 1.0
    suppression: float = 0.0
    variance: float = 0.1

class GeneticTrait:
    def __init__(self, name: str, value: float, 
                 is_dominant: bool = False, 
                 is_suppressed: bool = False,
                 expression: Optional[TraitExpression] = None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.value = value
        self.is_dominant = is_dominant
        self.is_suppressed = is_suppressed
        self.expression = expression or TraitExpression()
        self.dna_sequence = self._generate_dna_sequence()
        self.creation_time = datetime.now()
    
    def _generate_dna_sequence(self) -> List[DigitalNucleotide]:
        sequence_length = 8
        return [DigitalNucleotide(random.randint(0, 3)) for _ in range(sequence_length)]
    
    @classmethod
    def from_parents(cls, parent1_trait: 'GeneticTrait', 
                    parent2_trait: 'GeneticTrait',
                    mutation_chance: float = 0.1) -> 'GeneticTrait':
        # Inheritance logic
        if parent1_trait.is_dominant and not parent2_trait.is_dominant:
            primary_parent = parent1_trait
        elif parent2_trait.is_dominant and not parent1_trait.is_dominant:
            primary_parent = parent2_trait
        else:
            primary_parent = random.choice([parent1_trait, parent2_trait])
        
        # Mix traits
        mixed_value = (parent1_trait.value + parent2_trait.value) / 2
        if random.random() < mutation_chance:
            mixed_value += random.gauss(0, 0.1)
        
        new_trait = cls(
            name=primary_parent.name,
            value=max(0, min(1, mixed_value)),
            is_dominant=random.random() > 0.5,
            expression=TraitExpression(
                strength=(parent1_trait.expression.strength + 
                         parent2_trait.expression.strength) / 2
            )
        )
        
        # DNA inheritance
        for i in range(len(new_trait.dna_sequence)):
            donor = parent1_trait if random.random() < 0.5 else parent2_trait
            new_trait.dna_sequence[i] = donor.dna_sequence[i]
            if random.random() < mutation_chance:
                new_trait.dna_sequence[i].mutate()
        
        return new_trait
    
    def get_expression_value(self) -> float:
        if self.is_suppressed:
            return 0.0
        base_value = self.value * self.expression.strength
        variation = random.gauss(0, self.expression.variance)
        return max(0, min(1, base_value + variation))