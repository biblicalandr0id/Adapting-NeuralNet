import os
from datetime import datetime
from embryo_namer import EmbryoNamer
from genetics import GeneticCore
import json


class EmbryoGenerator:
    def __init__(self, genetic_core: GeneticCore, embryo_namer: EmbryoNamer):
        self.genetic_core = genetic_core
        self.embryo_namer = embryo_namer
        self.embryo_id = self.embryo_namer.generate_random_name()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_embryo_file(self, output_dir="embryos"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{output_dir}/embryo_{self.embryo_id}_{self.timestamp}.py"

        template = self._create_embryo_template()

        with open(filename, 'w') as f:
            f.write(template)

        return filename

    def _create_embryo_template(self):
        physical_genetics = self.genetic_core.physical_genetics
        mind_genetics = self.genetic_core.mind_genetics
        base_traits = self.genetic_core.base_traits
        heart_genetics = self.genetic_core.heart_genetics
        brain_genetics = self.genetic_core.brain_genetics
        growth_rate = physical_genetics.growth_rate
        specializations_data = {
            "pattern_recognition_focus": mind_genetics.pattern_recognition,
            "learning_style": mind_genetics.learning_efficiency,
            "cognitive_capacity": mind_genetics.memory_capacity,
            "neural_adaptability": mind_genetics.neural_plasticity
        }
        potential_capabilities_data = {
            "adaptability_index": base_traits.adaptability,
            "resilience_factor": base_traits.resilience,
            "processing_power": base_traits.complexity,
            "efficiency_level": base_traits.efficiency
        }
        specializations_json = json.dumps(specializations_data, indent=8)
        potential_capabilities_json = json.dumps(
            potential_capabilities_data, indent=8)

        return f'''
import random
import time
from datetime import datetime
import numpy as np
from pathlib import Path

class AutonomousEmbryo:
    def __init__(self):
        self.embryo_id = "{self.embryo_id}"
        self.conception_time = "{self.timestamp}"
        self.genetic_traits = {{
            'growth_rate': {physical_genetics.growth_rate},
            'energy_efficiency': {physical_genetics.energy_efficiency},
            'structural_integrity': {physical_genetics.structural_integrity},
            'sensor_sensitivity': {physical_genetics.sensor_sensitivity},
            'action_precision': {physical_genetics.action_precision},
            'cognitive_growth_rate': {mind_genetics.cognitive_growth_rate},
            'learning_efficiency': {mind_genetics.learning_efficiency},
            'memory_capacity': {mind_genetics.memory_capacity},
            'neural_plasticity': {mind_genetics.neural_plasticity},
            'pattern_recognition': {mind_genetics.pattern_recognition},
            'trust_baseline': {heart_genetics.trust_baseline},
            'security_sensitivity': {heart_genetics.security_sensitivity},
            'adaptation_rate': {heart_genetics.adaptation_rate},
            'integrity_check_frequency': {heart_genetics.integrity_check_frequency},
            'recovery_resilience': {heart_genetics.recovery_resilience},
            'processing_speed': {brain_genetics.processing_speed},
            'emotional_stability': {brain_genetics.emotional_stability},
            'focus_capacity': {brain_genetics.focus_capacity},
            'ui_responsiveness': {brain_genetics.ui_responsiveness},
            'interaction_capability': {brain_genetics.interaction_capability},
            'resilience': {base_traits.resilience},
            'adaptability': {base_traits.adaptability},
            'efficiency': {base_traits.efficiency},
            'complexity': {base_traits.complexity},
            'stability': {base_traits.stability}
        }}
        self.specializations = {specializations_json}
        self.potential_capabilities = {potential_capabilities_json}
        self.growth_rate = {growth_rate}

        self.age = 0
        self.experiences = []
        self.learned_patterns = {{}}
        self.development_stage = 0
        self.neural_connections = self._initialize_neural_connections()

        self.log_file = Path(f"development_logs/{{self.embryo_id}}.json")
        self.log_file.parent.mkdir(exist_ok=True)
        self._log_development("Embryo initialized")

    def _initialize_neural_connections(self):
        base_connections = int(self.genetic_traits['sensor_sensitivity'] * 100)
        return {{
            'cognitive': base_connections * self.genetic_traits['pattern_recognition'],
            'behavioral': base_connections * self.genetic_traits['adaptability'],
            'processing': base_connections * self.genetic_traits['processing_speed']
        }}

    def learn_from_experience(self, experience_data):
        learning_effectiveness = self.genetic_traits['learning_capacity'] / 3.0
        processed_data = self._process_experience(experience_data)
        self.experiences.append({{
            'timestamp': datetime.now().isoformat(),
            'data': processed_data,
            'learning_impact': learning_effectiveness
        }})
        self._update_neural_connections(processed_data)
        self._log_development(f"Learned from experience: {{len(self.experiences)}}")
        return processed_data

    def _process_experience(self, experience_data):
        processing_quality = self.genetic_traits['processing_speed'] / 3.0
        pattern_recognition = self.genetic_traits['pattern_recognition'] / 3.0
        processed_result = {{
            'original_data': experience_data,
            'processing_quality': processing_quality,
            'patterns_recognized': pattern_recognition,
            'timestamp': datetime.now().isoformat()
        }}
        return processed_result

    def _update_neural_connections(self, processed_data):
        growth_factor = self.growth_rate * 0.1
        for connection_type in self.neural_connections:
            current_connections = self.neural_connections[connection_type]
            new_connections = int(current_connections * (1 + growth_factor))
            self.neural_connections[connection_type] = new_connections

    def develop(self):
        self.age += 1
        development_rate = self.growth_rate * (
            1 + len(self.experiences) * 0.01 * self.genetic_traits['adaptability']
        )
        self.development_stage += development_rate
        if self.development_stage >= 10 and len(self.specializations) > 0:
            self._develop_specializations()
        self._log_development(f"Development progressed: Stage {{self.development_stage:.2f}}")
        return self.development_stage

    def _develop_specializations(self):
        for specialization in self.specializations:
            if specialization not in self.learned_patterns:
                self.learned_patterns[specialization] = {{
                    'development_level': 0,
                    'activation_count': 0,
                    'effectiveness': self.genetic_traits['task_specialization'] / 3.0
                }}
            self.learned_patterns[specialization]['development_level'] += (
                self.growth_rate * self.genetic_traits['task_specialization'] * 0.1
            )

    def _log_development(self, event):
        log_entry = {{
            'timestamp': datetime.now().isoformat(),
            'age': self.age,
            'development_stage': self.development_stage,
            'event': event,
            'neural_connections': self.neural_connections,
            'specializations': self.learned_patterns
        }}
        if not self.log_file.exists():
            self.log_file.write_text(json.dumps([log_entry], indent=2))
        else:
            logs = json.loads(self.log_file.read_text())
            logs.append(log_entry)
            self.log_file.write_text(json.dumps(logs, indent=2))

    def get_status(self):
        return {{
            'embryo_id': self.embryo_id,
            'age': self.age,
            'development_stage': self.development_stage,
            'experiences_count': len(self.experiences),
            'neural_connections': self.neural_connections,
            'specializations': self.learned_patterns,
            'potential_capabilities': self.potential_capabilities
        }}
'''


def generate_embryo(genetic_data, output_dir="embryos"):
    generator = EmbryoGenerator(genetic_data)
    return generator.generate_embryo_file(output_dir)
