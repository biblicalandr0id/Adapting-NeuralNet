import random


# --- EMBRYO NAMER (embryo_namer.py) ---
class EmbryoNamer:
    def __init__(self):
        self.prefixes = [
            "Neo", "Syn", "Quantum", "Cyber", "Digital", "Neural", "Binary",
            "Data", "Logic", "Matrix", "Vector", "Alpha", "Beta", "Delta",
            "Echo", "Gamma", "Omega", "Prime", "Core", "Node"
        ]

        self.suffixes = [
            "Wave", "Core", "Mind", "Net", "Flux", "Node", "Link", "Path",
            "Web", "Grid", "Flow", "Stream", "Pulse", "Spark", "Point",
            "Base", "Loop", "Chain", "Sync", "Edge"
        ]

        self.unique_identifiers = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"
        ]

    def generate_random_name(self):
        """Generate a random name for an embryo"""
        prefix = random.choice(self.prefixes)
        suffix = random.choice(self.suffixes)
        identifier = random.choice(self.unique_identifiers)
        sequence = random.randint(100, 999)

        return f"{prefix}-{suffix}-{identifier}{sequence}"

    def generate_inherited_name(self, parent1_name, parent2_name):
        """Generate a name for an embryo based on its parents' names"""
        # Extract components from parent names
        p1_prefix = parent1_name.split('-')[0]
        p2_prefix = parent2_name.split('-')[0]

        # Combine first two letters of each parent's prefix
        new_prefix = p1_prefix[:2] + p2_prefix[:2]

        # Choose a random suffix
        new_suffix = random.choice(self.suffixes)

        # Generate a unique sequence
        sequence = random.randint(100, 999)

        return f"{new_prefix}-{new_suffix}-{sequence}"
