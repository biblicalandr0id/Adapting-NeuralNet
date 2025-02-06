import json
import random
from datetime import datetime
from pathlib import Path
import uuid
import math

class DigitalNucleotide:
    def __init__(self, value):
        self.value = value
        self.generation = 0
        self.mutation_history = []
        
    def complement(self):
        return DigitalNucleotide(self.value ^ 0b11)
        
    def mutate(self):
        old_value = self.value
        mutation_type = random.random()
        
        if mutation_type < 0.7:  # Point mutation
            self.value = random.randint(0, 3)
        elif mutation_type < 0.85:  # Bit flip
            self.value = self.value ^ (0b01 if random.random() < 0.5 else 0b10)
        else:  # Complement mutation
            self.value = self.value ^ 0b11
            
        self.mutation_history.append({
            'generation': self.generation,
            'from': old_value,
            'to': self.value,
            'type': 'point' if mutation_type < 0.7 else 'bit-flip' if mutation_type < 0.85 else 'complement'
        })
        self.generation += 1
        return self

class GeneticTrait:
    def __init__(self, name, value, is_dominant=False, is_suppressed=False):
        self.name = name
        self.value = value
        self.is_dominant = is_dominant
        self.is_suppressed = is_suppressed
        self.dna_sequence = self._generate_dna_sequence()
    
    def _generate_dna_sequence(self):
        """Generate DNA sequence representing the trait"""
        sequence_length = 8  # Each trait represented by 8 nucleotides
        return [DigitalNucleotide(random.randint(0, 3)) for _ in range(sequence_length)]
    
    @classmethod
    def from_parents(cls, parent1_trait, parent2_trait):
        """Create a new trait by combining parent traits"""
        # Randomly select dominance and suppression from parents
        is_dominant = random.choice([parent1_trait.is_dominant, parent2_trait.is_dominant])
        is_suppressed = random.choice([parent1_trait.is_suppressed, parent2_trait.is_suppressed])
        
        # Value inheritance with slight mutation
        base_value = random.choice([parent1_trait.value, parent2_trait.value])
        mutation = random.uniform(-0.2, 0.2)  # Allow small variations
        new_value = max(1.5, min(3.0, base_value + mutation))
        
        return cls(parent1_trait.name, round(new_value, 2), is_dominant, is_suppressed)

class GeneticDominanceHandler:
    def __init__(self):
        self.specialization_dominance = {
            "learning_capacity": {"dominant": [3, 2], "recessive": [0, 1]},
            "pattern_recognition": {"dominant": [3, 2], "recessive": [0, 1]},
            "decision_making": {"dominant": [1, 2], "recessive": [0, 3]},
            "memory_capacity": {"dominant": [0, 3], "recessive": [1, 2]},
            "adaptability": {"dominant": [3, 2], "recessive": [1, 0]},
            "social_interaction": {"dominant": [2, 3], "recessive": [0, 1]},
            "task_specialization": {"dominant": [1, 3], "recessive": [0, 2]},
            "resource_management": {"dominant": [0, 1], "recessive": [2, 3]},
            "processing_speed": {"dominant": [3, 2], "recessive": [0, 1]},
            "energy_efficiency": {"dominant": [2, 3], "recessive": [0, 1]},
            "error_tolerance": {"dominant": [3, 1], "recessive": [2, 0]},
            "parallel_processing": {"dominant": [2, 3], "recessive": [0, 1]}
        }
    
    def determine_trait(self, trait1, trait2, specialization=None):
        """Determine trait expression when combining two genetic traits"""
        if trait1.is_suppressed and trait2.is_suppressed:
            return GeneticTrait(trait1.name, 0, False, True)
            
        if trait1.is_suppressed:
            return GeneticTrait(trait2.name, trait2.value, trait2.is_dominant, False)
        if trait2.is_suppressed:
            return GeneticTrait(trait1.name, trait1.value, trait1.is_dominant, False)
            
        if specialization and specialization in self.specialization_dominance:
            return self._determine_specialized_trait(trait1, trait2, specialization)
            
        if trait1.is_dominant and trait2.is_dominant:
            # Co-dominance: average the values
            return GeneticTrait(
                trait1.name,
                (trait1.value + trait2.value) / 2,
                True,
                False
            )
        elif trait1.is_dominant:
            return GeneticTrait(trait1.name, trait1.value, True, False)
        elif trait2.is_dominant:
            return GeneticTrait(trait2.name, trait2.value, True, False)
        else:
            # Both recessive, take lower value
            return GeneticTrait(
                trait1.name,
                min(trait1.value, trait2.value),
                False,
                False
            )

def generate_genetic_data(parent1_data=None, parent2_data=None):
    """Generate genetic traits either randomly or through inheritance"""
    dominance_handler = GeneticDominanceHandler()
    
    trait_names = [
        "learning_capacity", "pattern_recognition", "decision_making",
        "memory_capacity", "adaptability", "social_interaction",
        "task_specialization", "resource_management", "processing_speed",
        "energy_efficiency", "error_tolerance", "parallel_processing"
    ]
    
    if parent1_data and parent2_data:
        # Create traits through inheritance
        traits = {}
        for name in trait_names:
            parent1_trait = GeneticTrait(
                name, 
                parent1_data["combined_traits"][name],
                random.random() > 0.5,
                random.random() < 0.1
            )
            parent2_trait = GeneticTrait(
                name,
                parent2_data["combined_traits"][name],
                random.random() > 0.5,
                random.random() < 0.1
            )
            traits[name] = GeneticTrait.from_parents(parent1_trait, parent2_trait)
    else:
        # Generate random traits
        traits = {
            name: GeneticTrait(
                name,
                round(random.uniform(1.5, 3.0), 2),
                random.random() > 0.5,
                random.random() < 0.1
            )
            for name in trait_names
        }
    
    # Combine traits using dominance rules
    combined_traits = {}
    for name, trait in traits.items():
        # Create a second trait for combination
        second_trait = GeneticTrait(
            name,
            round(random.uniform(1.5, 3.0), 2),
            random.random() > 0.5,
            random.random() < 0.1
        )
        combined_trait = dominance_handler.determine_trait(trait, second_trait, name)
        combined_traits[name] = combined_trait.value
    
    # Inherit or generate specializations
    if parent1_data and parent2_data:
        # Combine specializations from parents with possible mutations
        all_specializations = set(parent1_data["specializations"] + parent2_data["specializations"])
        num_specializations = random.randint(2, min(4, len(all_specializations)))
        specializations = random.sample(list(all_specializations), num_specializations)
    else:
        specializations = random.sample([
            "pattern_analysis", "decision_optimization", "multi_task_processing",
            "collaborative_learning", "resource_optimization", "error_correction",
            "adaptive_learning", "parallel_computation"
        ], k=random.randint(2, 4))
    
    # Generate or inherit potential capabilities
    if parent1_data and parent2_data:
        potential_capabilities = {}
        for capability in ["learning_potential", "adaptation_capacity", 
                         "processing_capability", "social_capability"]:
            # Average parents' values with small random variation
            base_value = (parent1_data["potential_capabilities"][capability] + 
                         parent2_data["potential_capabilities"][capability]) / 2
            variation = random.uniform(-0.2, 0.2)
            potential_capabilities[capability] = round(
                max(2.0, min(3.0, base_value + variation)), 
                2
            )
    else:
        potential_capabilities = {
            "learning_potential": round(random.uniform(2.0, 3.0), 2),
            "adaptation_capacity": round(random.uniform(2.0, 3.0), 2),
            "processing_capability": round(random.uniform(2.0, 3.0), 2),
            "social_capability": round(random.uniform(2.0, 3.0), 2)
        }
    
    return {
        "combined_traits": combined_traits,
        "specializations": specializations,
        "potential_capabilities": potential_capabilities,
        "growth_rate": round(random.uniform(0.1, 0.2), 2),
        "dna_information": {
            "mutation_rates": {
                "point": 0.001,
                "insertion": 0.0005,
                "deletion": 0.0005,
                "duplication": 0.0002,
                "inversion": 0.0002
            },
            "generation": 0 if not parent1_data else max(
                parent1_data.get("dna_information", {}).get("generation", 0),
                parent2_data.get("dna_information", {}).get("generation", 0)
            ) + 1,
            "trait_sequences": {
                name: [n.value for n in trait.dna_sequence]
                for name, trait in traits.items()
            }
        }
    }

def create_conception_record(embryo_id, genetic_data, parent1_id=None, parent2_id=None):
    """Create a record of the conception with optional parent information"""
    record = {
        "embryo_id": embryo_id,
        "conception_time": datetime.now().isoformat(),
        "genetic_data": genetic_data,
        "parentage": {
            "parent1_id": parent1_id,
            "parent2_id": parent2_id
        } if parent1_id and parent2_id else None
    }
    
    records_dir = Path("conception_records")
    records_dir.mkdir(exist_ok=True)
    
    record_file = records_dir / f"conception_{embryo_id}.json"
    record_file.write_text(json.dumps(record, indent=2))

def conceive_embryo(parent1_id=None, parent2_id=None):
    """Create a new embryo either randomly or from parents"""
    embryo_id = str(uuid.uuid4())[:8]
    
    # Load parent data if provided
    parent1_data = None
    parent2_data = None
    
    if parent1_id and parent2_id:
        records_dir = Path("conception_records")
        try:
            with open(records_dir / f"conception_{parent1_id}.json") as f:
                parent1_data = json.loads(f)["genetic_data"]
            with open(records_dir / f"conception_{parent2_id}.json") as f:
                parent2_data = json.loads(f)["genetic_data"]
        except FileNotFoundError:
            print("Warning: Parent data not found, generating random embryo")
    
    genetic_data = generate_genetic_data(parent1_data, parent2_data)
    create_conception_record(embryo_id, genetic_data, parent1_id, parent2_id)
    
    return embryo_id

if __name__ == "__main__":
    # Example usage
    # Random embryo:
    embryo_id = conceive_embryo()
    print(f"Random Embryo ID: {embryo_id}")
    
    # Create two parent embryos and then a child
    parent1_id = conceive_embryo()
    parent2_id = conceive_embryo()
    child_id = conceive_embryo(parent1_id, parent2_id)
    print(f"\nParent 1 ID: {parent1_id}")
    print(f"Parent 2 ID: {parent2_id}")
    print(f"Child Embryo ID: {child_id}")
