import os
from datetime import datetime
from genetics import (
    GeneticCore, 
    EmbryoGenetics,
    MindGenetics,
    BrainGenetics,
    HeartGenetics,
    PhysicalGenetics,
    BaseTraits
)
import json
import uuid
from typing import Dict, Optional, Tuple, Any
import logging
from neural_networks import (
    NeuralAdaptiveNetwork, 
    GeneticMemoryCell  # Add this import
)
from dataclasses import dataclass
from mind import Mind
from brain import Brain
from heart import HeartSystem
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from agent import AdaptiveAgent


logger = logging.getLogger(__name__)


@dataclass
class Embryo:
    """Represents a developing agent embryo"""
    genetics: EmbryoGenetics
    parent: Optional[Any] = None
    position: Optional[tuple] = None
    environment: Optional[Any] = None
    gender: str = "undefined"
    genetic_core: Optional[GeneticCore] = None  # Fixed capitalization

    def __post_init__(self):
        """Initialize additional embryo attributes"""
        self.development_progress = 0.0
        self.maturity_threshold = 1.0 / self.genetics.development_rate
        self.gender = self._determine_gender()
        
        # Initialize genetic_core if not provided
        if self.genetic_core is None:  # Fixed capitalization
            self.genetic_core = GeneticCore()  # Fixed capitalization
            self.genetic_core.embryo_genetics = self.genetics  # Fixed capitalization

    def _determine_gender(self) -> str:
        """Determine embryo gender"""
        import random
        return random.choice(["male", "female"])


class EmbryoGenerator:
    """Handles the creation of new embryos with genetic traits"""
    def __init__(self):
        self.successful_embryos = 0
        self.failed_attempts = 0
        self.logger = logging.getLogger(__name__)

    def generate_embryo(self, parent_genetics: Optional[Dict] = None) -> Tuple[GeneticCore, NeuralAdaptiveNetwork]:
        """Generate a new embryo with optional parent genetics"""
        try:
            # Create genetic core
            genetic_core = GeneticCore()
            if parent_genetics:
                genetic_core.inherit_from(parent_genetics)
            else:
                genetic_core.initialize_random_genetics()

            # Generate neural network based on genetics
            network_params = self._calculate_network_architecture(genetic_core)
            neural_net = NeuralAdaptiveNetwork(
                input_size=network_params['input_size'],
                hidden_size=network_params['hidden_size'],
                output_size=network_params['output_size']
            )

            self.successful_embryos += 1
            return genetic_core, neural_net

        except Exception as e:
            self.failed_attempts += 1
            logger.error(f"Failed to generate embryo: {str(e)}")
            raise

    def create_embryo(self, 
                     parent=None, 
                     position=None,
                     environment=None) -> 'Embryo':
        """Create new embryo using existing genetics system"""
        try:
            # Create genetic core first with complete traits
            genetic_core = GeneticCore()
            genetic_core.initialize_random_genetics()  # Ensure all genetics are initialized

            # Initialize embryo genetics with all required parameters
            embryo_genetics = EmbryoGenetics(
                development_rate=random.uniform(0.8, 1.2),
                mutation_chance=random.uniform(0.01, 0.1),
                inheritance_factor=random.uniform(0.5, 0.9),
                trait_stability=random.uniform(0.7, 0.9),
                neural_plasticity=random.uniform(0.4, 0.8),
                cognitive_growth_rate=random.uniform(0.8, 1.5),
                development_plasticity=random.uniform(0.4, 0.8)
            )

            # Ensure genetic core has all required genetics initialized
            genetic_core.mind_genetics = MindGenetics(
                creativity=random.uniform(0.7, 1.3),
                learning_efficiency=random.uniform(0.8, 1.2),
                adaptation_rate=random.uniform(0.6, 1.4),
                memory_capacity=random.uniform(0.7, 1.3),
                problem_solving=random.uniform(0.8, 1.2),
                strategic_thinking=random.uniform(0.7, 1.3),
                risk_assessment=random.uniform(0.6, 1.4),
                social_awareness=random.uniform(0.5, 1.5),
                innovation_drive=random.uniform(0.7, 1.3),
                curiosity_factor=random.uniform(0.8, 1.2)
            )
            
            genetic_core.brain_genetics = BrainGenetics(
                processing_speed=random.uniform(0.8, 1.2),
                pattern_recognition=random.uniform(0.7, 1.3),
                neural_plasticity=random.uniform(0.6, 1.4),
                learning_rate=random.uniform(0.05, 0.15),
                memory_capacity=random.uniform(0.6, 1.4),
                decision_speed=random.uniform(0.8, 1.2),
                multi_tasking=random.uniform(0.7, 1.3),
                spatial_awareness=random.uniform(0.6, 1.4),
                temporal_processing=random.uniform(0.7, 1.3),
                error_correction=random.uniform(0.8, 1.2)
            )

            genetic_core.heart_genetics = HeartGenetics(
                base_rhythm=random.uniform(0.8, 1.2),
                stress_sensitivity=random.uniform(0.6, 1.4),
                resilience=random.uniform(0.7, 1.3),
                recovery_resilience=random.uniform(0.8, 1.2),
                adaptation_rate=random.uniform(0.7, 1.3),
                trust_baseline=random.uniform(0.4, 0.8),
                emotional_capacity=random.uniform(0.6, 1.4),
                security_sensitivity=random.uniform(0.7, 1.3)
            )

            genetic_core.physical_genetics = PhysicalGenetics(
                size=random.uniform(0.8, 1.2),
                speed=random.uniform(0.7, 1.3),
                strength=random.uniform(0.8, 1.2),
                energy_efficiency=random.uniform(0.7, 1.3),
                sensor_sensitivity=random.uniform(0.6, 1.4),
                regeneration_rate=random.uniform(0.5, 1.5),
                immune_system=random.uniform(0.7, 1.3),
                metabolic_rate=random.uniform(0.8, 1.2),
                longevity_factor=random.uniform(0.7, 1.3),
                adaptation_speed=random.uniform(0.6, 1.4),
                action_precision=random.uniform(0.7, 1.3)
            )

            # Set embryo genetics in genetic core
            genetic_core.embryo_genetics = embryo_genetics

            # Create embryo with complete genetic initialization
            embryo = Embryo(
                genetics=embryo_genetics,
                genetic_core=genetic_core,  # Pass genetic_core with correct field name
                parent=parent,
                position=position or (0, 0),
                environment=environment,
                gender=random.choice(["male", "female"])
            )

            self.successful_embryos += 1
            self.logger.info(f"Created embryo with development rate: {embryo_genetics.development_rate}")
            return embryo

        except Exception as e:
            self.failed_attempts += 1
            self.logger.error(f"Failed to create embryo: {str(e)}")
            raise

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
            self.learned_patterns[specialization['development_level'] += (
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


class EmbryoToAgentDevelopment:
    """Handles the development pipeline from embryo to assembled agent"""
    def __init__(self, agent_assembler):
        self.agent_assembler = agent_assembler
        self.logger = logging.getLogger(__name__)
        self.development_stages = {
            'embryonic': 0.0,
            'neural_formation': 0.25,
            'cognitive_development': 0.5,
            'system_integration': 0.75,
            'maturation': 1.0
        }
    
    def develop_and_assemble(self, embryo: Embryo) -> Optional['AdaptiveAgent']:
        """Develop embryo through stages and assemble into agent"""
        try:
            # Track development progress
            development_log = []
            
            # 1. Embryonic Stage - Basic genetic expression
            genetic_core = self._develop_genetic_core(embryo)
            development_log.append(self._log_stage('embryonic', embryo))
            
            # 2. Neural Formation - Create neural network
            neural_net = self._develop_neural_network(embryo, genetic_core)
            development_log.append(self._log_stage('neural_formation', embryo))
            
            # 3. Cognitive Development - Initialize systems
            cognitive_systems = self._develop_cognitive_systems(embryo, genetic_core)
            development_log.append(self._log_stage('cognitive_development', embryo))
            
            # 4. System Integration - Connect all components
            self._integrate_systems(embryo, genetic_core, neural_net, cognitive_systems)
            development_log.append(self._log_stage('system_integration', embryo))
            
            # 5. Final Assembly - Create agent with all required components
            assembled = self.agent_assembler.assemble(
                position=embryo.position,
                parent=embryo.parent,
                gender=embryo.gender,
                environment=embryo.environment
            )
            
            if assembled.success:
                # Attach the developed systems
                agent = assembled.agent
                agent.neural_net = neural_net
                agent.genetic_core = genetic_core
                agent.brain = cognitive_systems['brain']
                agent.mind = cognitive_systems['mind']
                agent.heart = cognitive_systems['heart']
                
                self.logger.info(f"Successfully developed embryo into agent: {agent.id}")
                return agent
            else:
                self.logger.error(f"Failed to assemble agent: {assembled.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"Development failed: {str(e)}")
            return None

    def _develop_genetic_core(self, embryo: 'Embryo') -> GeneticCore:
        """Transform embryo genetics into full genetic core traits"""
        # Use existing genetic core if embryo has one, otherwise create new
        genetic_core = embryo.genetic_core if embryo.genetic_core else GeneticCore()

        # Initialize all required genetics classes if they don't exist
        genetic_core.mind_genetics = MindGenetics()
        genetic_core.brain_genetics = BrainGenetics()
        genetic_core.heart_genetics = HeartGenetics()
        genetic_core.physical_genetics = PhysicalGenetics()
        genetic_core.base_traits = BaseTraits()
        
        embryo_traits = embryo.genetics

        # Mind Genetics - Transform learning and cognitive traits
        # Use brain_genetics to store cognitive growth since mind_genetics doesn't have it
        genetic_core.brain_genetics.learning_rate = embryo_traits.cognitive_growth_rate
        genetic_core.mind_genetics.adaptation_rate = embryo_traits.development_plasticity
        genetic_core.mind_genetics.learning_efficiency = embryo_traits.development_rate
        genetic_core.mind_genetics.memory_capacity = embryo_traits.neural_plasticity * 1.5
        genetic_core.mind_genetics.creativity = embryo_traits.development_plasticity * 1.2
        genetic_core.mind_genetics.pattern_recognition = embryo_traits.cognitive_growth_rate * 1.3
        genetic_core.mind_genetics.problem_solving = embryo_traits.cognitive_growth_rate
        genetic_core.mind_genetics.strategic_thinking = embryo_traits.neural_plasticity
        genetic_core.mind_genetics.risk_assessment = embryo_traits.trait_stability
        genetic_core.mind_genetics.social_awareness = embryo_traits.development_plasticity
        genetic_core.mind_genetics.innovation_drive = embryo_traits.cognitive_growth_rate
        genetic_core.mind_genetics.curiosity_factor = embryo_traits.development_plasticity

        # Brain Genetics - Transform neural and processing traits
        genetic_core.brain_genetics.neural_plasticity = embryo_traits.neural_plasticity
        genetic_core.brain_genetics.learning_rate = embryo_traits.development_rate
        genetic_core.brain_genetics.memory_capacity = embryo_traits.cognitive_growth_rate
        genetic_core.brain_genetics.processing_speed = embryo_traits.development_rate * 1.4
        genetic_core.brain_genetics.pattern_recognition = embryo_traits.cognitive_growth_rate * 1.2
        genetic_core.brain_genetics.development_plasticity = embryo_traits.development_plasticity
        genetic_core.brain_genetics.decision_speed = embryo_traits.development_rate
        genetic_core.brain_genetics.multi_tasking = embryo_traits.cognitive_growth_rate
        genetic_core.brain_genetics.spatial_awareness = embryo_traits.neural_plasticity
        genetic_core.brain_genetics.temporal_processing = embryo_traits.development_rate
        genetic_core.brain_genetics.error_correction = embryo_traits.trait_stability

        # Heart Genetics - Transform emotional and metabolic traits
        genetic_core.heart_genetics.base_rhythm = embryo_traits.development_rate
        genetic_core.heart_genetics.stress_sensitivity = embryo_traits.neural_plasticity
        genetic_core.heart_genetics.resilience = embryo_traits.trait_stability
        genetic_core.heart_genetics.recovery_resilience = embryo_traits.trait_stability
        genetic_core.heart_genetics.adaptation_rate = embryo_traits.development_plasticity
        genetic_core.heart_genetics.trust_baseline = embryo_traits.trait_stability * 0.8
        genetic_core.heart_genetics.emotional_capacity = embryo_traits.neural_plasticity
        genetic_core.heart_genetics.security_sensitivity = embryo_traits.development_plasticity

        # Physical Genetics - Transform physical capabilities
        genetic_core.physical_genetics.size = embryo_traits.cognitive_growth_rate
        genetic_core.physical_genetics.speed = embryo_traits.development_rate
        genetic_core.physical_genetics.strength = embryo_traits.trait_stability
        genetic_core.physical_genetics.energy_efficiency = embryo_traits.development_rate
        genetic_core.physical_genetics.sensor_sensitivity = embryo_traits.neural_plasticity
        genetic_core.physical_genetics.regeneration_rate = embryo_traits.trait_stability
        genetic_core.physical_genetics.immune_system = embryo_traits.trait_stability
        genetic_core.physical_genetics.metabolic_rate = embryo_traits.cognitive_growth_rate
        genetic_core.physical_genetics.longevity_factor = embryo_traits.trait_stability
        genetic_core.physical_genetics.adaptation_speed = embryo_traits.development_plasticity
        genetic_core.physical_genetics.action_precision = embryo_traits.neural_plasticity

        # Base Traits
        genetic_core.base_traits.resilience = embryo_traits.trait_stability
        genetic_core.base_traits.adaptability = embryo_traits.development_plasticity
        genetic_core.base_traits.efficiency = embryo_traits.development_rate
        genetic_core.base_traits.complexity = embryo_traits.cognitive_growth_rate
        genetic_core.base_traits.stability = embryo_traits.trait_stability

        # Store original embryo genetics
        genetic_core.embryo_genetics = embryo_traits

        return genetic_core

    def _develop_neural_network(self, embryo: Embryo, genetic_core: GeneticCore) -> NeuralAdaptiveNetwork:
        """Develop neural network with embryonic traits"""
        network_params = self._calculate_network_architecture(embryo, genetic_core)
        
        return NeuralAdaptiveNetwork(
            input_size=network_params['input_size'],
            output_size=network_params['output_size'],
            hidden_size=network_params['hidden_size'],
            genetic_core=genetic_core
        )

    def _develop_cognitive_systems(self, embryo: Embryo, genetic_core: GeneticCore) -> Dict:
        """Develop cognitive systems (mind, brain, heart)"""
        # Create memory system first
        memory_system = GeneticMemoryCell(
            input_size=64,
            hidden_size=128,
            genetic_traits=genetic_core.mind_genetics
        )

        # Create neural network for mind
        neural_net = self._develop_neural_network(embryo, genetic_core)
        
        # Create mind properly using embryo's cognitive growth rate
        mind = Mind(
            genetic_core=genetic_core,
            neural_net=neural_net,
            memory_system=memory_system,
            growth_rate=embryo.genetics.cognitive_growth_rate  # Use embryo genetics directly
        )

        # Initialize all systems with proper parameters
        return {
            'mind': mind,
            'brain': Brain(
                genetic_core=genetic_core,
                neural_net=neural_net,
                plasticity=genetic_core.brain_genetics.neural_plasticity,
                learning_rate=genetic_core.brain_genetics.learning_rate
            ),
            'heart': HeartSystem(
                genetics=genetic_core.heart_genetics,
                state=GeneticCore.base_traits,
                energy_efficiency=genetic_core.physical_genetics.energy_efficiency,
                metabolic_rate=genetic_core.physical_genetics.metabolic_rate
            )
        }

    def _integrate_systems(self, embryo: Embryo, genetic_core: GeneticCore,
                         neural_net: NeuralAdaptiveNetwork,
                         cognitive_systems: Dict) -> Dict:
        """Integrate all systems according to embryo development stage"""
        # Connect brain to neural network
        cognitive_systems['brain'].neural_net = neural_net
        
        # Connect mind to brain
        cognitive_systems['mind'].brain = cognitive_systems['brain']
        
        # Connect heart to brain
        cognitive_systems['heart'].brain_link = cognitive_systems['brain']
        
        return cognitive_systems

    def _calculate_network_architecture(self, embryo: Embryo, genetic_core: GeneticCore) -> Dict:
        """Calculate neural network architecture based on embryo genetics"""
        base_size = 32
        growth_factor = embryo.genetics.cognitive_growth_rate
        
        return {
            'input_size': int(base_size * growth_factor),
            'output_size': int(base_size/2 * growth_factor),
            'hidden_size': int(base_size * 2 * growth_factor)
        }

    def _log_stage(self, stage: str, embryo: Embryo) -> Dict:
        """Log development stage information"""
        return {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'progress': self.development_stages[stage],
            'genetics': embryo.genetics.__dict__
        }
