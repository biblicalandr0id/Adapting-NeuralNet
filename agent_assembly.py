from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any, List
import logging
import uuid
from datetime import datetime
import time
from genetics import GeneticCore
from neural_networks import GeneticMemoryCell, NeuralAdaptiveNetwork
from augmentation import AdaptiveDataAugmenter
from agent import AdaptiveAgent, ActionDecoder
from brain import Brain
from heart import HeartSystem
from mind import Mind
from birth_records import BirthRegistry
from actions import ActionSystem
from diagnostics import NeuralDiagnostics
import torch
import json
import os
import random
import pathlib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AssemblyResult:
    """Assembly result with all necessary components"""
    agent: Optional[AdaptiveAgent]
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    assembly_time: Optional[float] = None

    def get_genetic_traits(self) -> Dict[str, Dict[str, float]]:
        """Get all genetic traits properly organized"""
        if not self.agent:
            return {}
            
        return {
            'base': vars(self.agent.genetic_core.base_traits),
            'mind': vars(self.agent.genetic_core.mind_genetics),
            'brain': vars(self.agent.genetic_core.brain_genetics),
            'heart': vars(self.agent.genetic_core.heart_genetics),
            'physical': vars(self.agent.genetic_core.physical_genetics),
            'embryo': vars(self.agent.genetic_core.embryo_genetics)
        }

@dataclass
class ComponentVerification:
    neural_network: bool = False
    genetic_core: bool = False
    mind: bool = False
    brain: bool = False
    heart: bool = False
    memory_system: bool = False
    action_system: bool = False
    birth_record: bool = False

class AgentAssembler:
    """Assembles agents while preserving all genetic and neural properties"""
    
    def __init__(self, agent_class, birth_registry: BirthRegistry):
        self.agent_class = agent_class
        self.birth_registry = birth_registry
        self.assembly_stats = {
            'successful_assemblies': 0,
            'failed_assemblies': 0,
            'assembly_times': []
        }
        
    def assemble(self, position: Tuple[float, float], 
                parent: Optional[AdaptiveAgent] = None,
                gender: Optional[str] = None,
                environment: Optional[Dict] = None,
                genetic_core: Optional[GeneticCore] = None) -> AssemblyResult:  # Added genetic_core parameter
        start_time = time.time()
        verification = ComponentVerification()
        birth_id = str(uuid.uuid4())
        
        try:
            # Create agent directory and subdirectories
            agent_id = str(uuid.uuid4())
            paths = self._create_agent_directory(agent_id)

            # 1. Initialize Genetic Core with complete embryo traits
            if genetic_core is None:
                genetic_core = (parent.genetic_core.create_offspring(mutation_rate=0.1) if parent else GeneticCore())
            genetic_core.embryo_genetics.development_plasticity = random.uniform(0.4, 0.8)
            verification.genetic_core = True

            # 2. Neural Network with enhanced genetic influence
            neural_net = self._create_neural_network(genetic_core)
            neural_net.plasticity = genetic_core.embryo_genetics.neural_plasticity
            neural_net.development_rate = genetic_core.embryo_genetics.development_rate
            neural_net.adaptation_rate = genetic_core.embryo_genetics.development_plasticity
            verification.neural_network = True

            # 3. Memory System with genetic scaling
            memory_capacity = int(64 * genetic_core.mind_genetics.memory_capacity)
            memory_system = GeneticMemoryCell(
                input_size=memory_capacity,
                hidden_size=memory_capacity * 2,
                genetic_traits=genetic_core.mind_genetics
            )
            verification.memory_system = True

            # 4. Action System with genetic influence
            action_system = ActionSystem(
                neural_net=neural_net,
                genetic_core=genetic_core,
                precision=genetic_core.physical_genetics.action_precision,
                efficiency=genetic_core.physical_genetics.energy_efficiency
            )
            verification.action_system = True

            # 5. Mind System with complete genetic traits
            mind = Mind(
                genetic_core=genetic_core,
                neural_net=neural_net,
                memory_system=memory_system,
                growth_rate=genetic_core.mind_genetics.cognitive_growth_rate,
                creativity=genetic_core.mind_genetics.creativity,
                learning_efficiency=genetic_core.mind_genetics.learning_efficiency,
                plasticity=genetic_core.embryo_genetics.development_plasticity
            )
            verification.mind = True

            # 6. Brain System with complete genetic influence
            brain = Brain(
                neural_net=neural_net,
                genetic_core=genetic_core,
                plasticity=genetic_core.brain_genetics.neural_plasticity,
                learning_rate=genetic_core.brain_genetics.learning_rate,
                processing_speed=genetic_core.brain_genetics.processing_speed,
                memory_capacity=genetic_core.brain_genetics.memory_capacity,
                pattern_recognition=genetic_core.brain_genetics.pattern_recognition
            )
            verification.brain = True

            # 7. Heart System with complete metabolic traits
            heart = HeartSystem(
                genetic_core=genetic_core,
                energy_efficiency=genetic_core.physical_genetics.energy_efficiency,
                metabolic_rate=genetic_core.physical_genetics.metabolic_rate,
                recovery_rate=genetic_core.heart_genetics.recovery_resilience,
                adaptation_rate=genetic_core.heart_genetics.adaptation_rate,
                trust_baseline=genetic_core.heart_genetics.trust_baseline
            )
            verification.heart = True

            # 8. Data Augmentation System
            data_augmenter = AdaptiveDataAugmenter(
                creativity=genetic_core.mind_genetics.creativity,
                pattern_recognition=genetic_core.brain_genetics.pattern_recognition
            )

            # 9. Neural Diagnostics
            neural_diagnostics = NeuralDiagnostics(
                check_frequency=int(100 * genetic_core.brain_genetics.processing_speed),
                alert_threshold=0.8 * genetic_core.mind_genetics.adaptation_rate
            )

            # 10. Birth Record and Registration
            birth_record = self.birth_registry.register_birth(
                parent_id=parent.id if parent else None,
                genetic_core=genetic_core,
                position=position,
                gender=gender or self._determine_gender()
            )
            verification.birth_record = True

            # Save all component states
            self._save_component_states(
                paths=paths,
                genetic_core=genetic_core,
                neural_net=neural_net,
                memory_system=memory_system,
                action_system=action_system,
                mind=mind,
                brain=brain,
                heart=heart
            )

            # Correct agent creation order - id MUST be first
            try:
                agent = self.agent_class(
                    id=agent_id,  # MUST be first parameter
                    genetic_core=genetic_core,
                    neural_net=neural_net,
                    position=position,
                    gender=birth_record.gender,
                    memory_system=memory_system,
                    action_system=action_system,
                    data_augmenter=data_augmenter,
                    neural_diagnostics=neural_diagnostics,
                    birth_record=birth_record,
                    parent_ids={'parent': parent.id} if parent else {}
                )
            except Exception as e:
                logger.error(f"Failed to create agent instance: {str(e)}")
                raise

            agent.environment = environment  # Set environment here
            agent.birth_id = birth_id  # Set birth_id here

            # Attach all systems
            agent.brain = brain
            agent.heart = heart
            agent.mind = mind

            # Final verification
            try:
                self._validate_assembly(agent)
                self._generate_basic_documentation(agent, paths, verification)
            except Exception as e:
                logger.error(f"Failed final validation: {str(e)}")
                raise

            return AssemblyResult(
                agent=agent,
                success=True,
                assembly_time=time.time() - start_time,
                metadata={
                    'agent_id': agent_id,
                    'path': paths['root'],
                    'verification': verification.__dict__
                }
            )

        except Exception as e:
            logger.error(f"Agent assembly failed: {str(e)}", exc_info=True)
            return AssemblyResult(
                agent=None,
                success=False,
                error_message=str(e),
                assembly_time=time.time() - start_time
            )

    def assemble_direct(self, position: Tuple[float, float],
                       parent: Optional[AdaptiveAgent] = None,
                       gender: Optional[str] = None,
                       environment: Optional[Dict] = None) -> AssemblyResult:
        """Directly assemble an agent without embryo stage"""
        start_time = time.time()
        verification = ComponentVerification()
        birth_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())

        try:
            # Create agent directory
            paths = self._create_agent_directory(agent_id)

            # 1. Initialize genetic core with all traits
            genetic_core = (parent.genetic_core.create_offspring(mutation_rate=0.1) if parent else GeneticCore())
            genetic_core.initialize_random_genetics()
            verification.genetic_core = True

            # 2. Create neural network with genetic parameters
            neural_net = NeuralAdaptiveNetwork(
                input_size=64,
                hidden_size=128,
                output_size=32,
                genetic_core=genetic_core
            )
            neural_net.plasticity = genetic_core.brain_genetics.neural_plasticity
            neural_net.development_rate = genetic_core.embryo_genetics.development_rate
            neural_net.adaptation_rate = genetic_core.mind_genetics.adaptation_rate
            verification.neural_network = True

            # 3. Memory System with genetic scaling
            memory_system = GeneticMemoryCell(
                input_size=int(64 * genetic_core.mind_genetics.memory_capacity),
                hidden_size=int(128 * genetic_core.mind_genetics.memory_capacity),
                genetic_traits=genetic_core.mind_genetics
            )
            verification.memory_system = True

            # 4. Action System with genetic traits
            action_system = ActionSystem(
                neural_net=neural_net,
                genetic_core=genetic_core,
                precision=genetic_core.physical_genetics.action_precision,
                efficiency=genetic_core.physical_genetics.energy_efficiency
            )
            verification.action_system = True

            # 5. Support Systems with genetic traits
            data_augmenter = AdaptiveDataAugmenter(
                creativity=genetic_core.mind_genetics.creativity,
                pattern_recognition=genetic_core.brain_genetics.pattern_recognition
            )

            neural_diagnostics = NeuralDiagnostics(
                check_frequency=int(100 * genetic_core.brain_genetics.processing_speed),
                alert_threshold=0.8 * genetic_core.mind_genetics.adaptation_rate
            )

            # 6. Birth Record
            birth_record = self.birth_registry.register_birth(
                parent_id=parent.id if parent else None,
                genetic_core=genetic_core,
                position=position,
                gender=gender or self._determine_gender()
            )
            verification.birth_record = True

            try:
                agent = self.agent_class(
                    id=agent_id,  # MUST be first parameter
                    genetic_core=genetic_core,
                    neural_net=neural_net,
                    position=position,
                    gender=birth_record.gender,
                    memory_system=memory_system,
                    action_system=action_system,
                    data_augmenter=data_augmenter,
                    neural_diagnostics=neural_diagnostics,
                    birth_record=birth_record,
                    parent_ids={'parent': parent.id} if parent else {}
                )
            except Exception as e:
                logger.error(f"Failed to create agent instance: {str(e)}")
                raise

            # Set environment and birth_id
            agent.environment = environment
            agent.birth_id = birth_id

            # Create and attach core systems with genetic traits
            agent.brain = Brain(
                neural_net=neural_net,
                genetic_core=genetic_core,
                plasticity=genetic_core.brain_genetics.neural_plasticity,
                learning_rate=genetic_core.brain_genetics.learning_rate
            )

            agent.heart = HeartSystem(
                genetic_core=genetic_core,
                energy_efficiency=genetic_core.physical_genetics.energy_efficiency,
                metabolic_rate=genetic_core.physical_genetics.metabolic_rate
            )

            agent.mind = Mind(
                genetic_core=genetic_core,
                neural_net=neural_net,
                memory_system=memory_system,
                growth_rate=genetic_core.mind_genetics.cognitive_growth_rate
            )

            # Set initial state based on genetic traits
            agent.energy = 100.0 * genetic_core.physical_genetics.energy_efficiency
            agent.max_energy = 150.0 * genetic_core.physical_genetics.energy_efficiency
            agent.sensor_sensitivity = genetic_core.physical_genetics.sensor_sensitivity
            agent.metabolic_rate = genetic_core.physical_genetics.metabolic_rate
            agent.action_precision = genetic_core.physical_genetics.action_precision
            agent.regeneration_rate = genetic_core.physical_genetics.regeneration_rate

            # Save component states
            self._save_component_states(
                paths=paths,
                genetic_core=genetic_core,
                neural_net=neural_net,
                memory_system=memory_system,
                action_system=action_system,
                mind=agent.mind,
                brain=agent.brain,
                heart=agent.heart
            )

            # Final validation
            self._validate_assembly(agent)
            self._generate_documentation(agent, paths, verification)

            return AssemblyResult(
                agent=agent,
                success=True,
                assembly_time=time.time() - start_time,
                metadata={
                    'agent_id': agent_id,
                    'path': paths['root'],
                    'verification': verification.__dict__,
                    'creation_type': 'direct'
                }
            )

        except Exception as e:
            logger.error(f"Direct agent assembly failed: {str(e)}", exc_info=True)
            return AssemblyResult(
                agent=None,
                success=False,
                error_message=str(e),
                assembly_time=time.time() - start_time
            )

    def _generate_basic_documentation(self, agent: 'AdaptiveAgent', 
                          path: str,
                          verification: ComponentVerification):
        """Generate basic agent documentation safely"""
        docs_path = Path(path)
        docs = {
            'agent_id': agent.id,
            'creation_time': datetime.now().isoformat(),
            'verification': verification.__dict__,
            'genetic_traits': self._get_genetic_traits(agent),
            'neural_architecture': self._get_neural_architecture(agent),
            'birth_record': self._get_birth_record(agent)
        }
        
        doc_path = docs_path / "agent_documentation.json"
        with open(doc_path, 'w') as f:
            json.dump(docs, f, indent=2)

    def _calculate_network_architecture(self, genetic_core: GeneticCore) -> Dict:
        """Calculate neural network architecture based on genetics"""
        # Get network dimensions from the genetic core's traits 
        input_layer_scale = genetic_core.brain_genetics.processing_power
        output_layer_scale = genetic_core.brain_genetics.pattern_recognition
        memory_capacity = genetic_core.mind_genetics.memory_capacity
        plasticity_bonus = genetic_core.embryo_genetics.development_plasticity * 0.3

        # Scale architecture based on both cognitive growth and plasticity
        return {
            'input_size': int(32 * max(1.0, input_layer_scale + plasticity_bonus)),
            'output_size': int(16 * max(1.0, output_layer_scale + plasticity_bonus)),
            'hidden_size': int(64 * memory_capacity * (1 + plasticity_bonus)),
            'num_layers': max(2, int(2 * genetic_core.mind_genetics.pattern_recognition))
        }

    def _create_neural_network(self, genetic_core: GeneticCore) -> NeuralAdaptiveNetwork:
        """Create and configure neural network with genetic traits"""
        try:
            # Get network architecture based on genetics
            architecture = self._calculate_network_architecture(genetic_core)
            
            # Create network with genetic parameters
            network = NeuralAdaptiveNetwork(
                input_size=architecture['input_size'],
                hidden_size=architecture['hidden_size'],
                output_size=architecture['output_size'],
                genetic_core=genetic_core
            )
            
            # Apply embryo genetic influences
            network.plasticity = genetic_core.embryo_genetics.neural_plasticity
            network.development_rate = genetic_core.embryo_genetics.development_rate
            network.adaptation_rate = genetic_core.embryo_genetics.development_plasticity
            
            # Initialize states
            network.reset_states()
            
            return network

        except Exception as e:
            logger.error(f"Neural network creation failed: {str(e)}")
            raise

    def _determine_gender(self) -> str:
        """Randomly determine agent gender"""
        return 'female' if random.random() < 0.5 else 'male'
        
    def _validate_assembly(self, agent: AdaptiveAgent) -> None:
        """Validate all required components are present and properly initialized"""
        required = [
            'genetic_core', 'neural_net', 'brain', 'heart', 
            'mind', 'position', 'birth_record', 'action_system', 
            'neural_diagnostics', 'data_augmenter'
        ]
        
        missing = [attr for attr in required if not hasattr(agent, attr)]
        if missing:
            raise ValueError(f"Agent missing required components: {missing}")

        # Validate genetic core contains all required genetics
        genetic_required = [
            'base_traits', 'mind_genetics', 'brain_genetics', 
            'heart_genetics', 'physical_genetics', 'embryo_genetics'
        ]
        missing_genetics = [g for g in genetic_required if not hasattr(agent.genetic_core, g)]
        if missing_genetics:
            raise ValueError(f"Genetic core missing required genetics: {missing_genetics}")

        # Validate all genetics have required traits
        self._validate_mind_genetics(agent.genetic_core.mind_genetics)
        self._validate_brain_genetics(agent.genetic_core.brain_genetics)  
        self._validate_heart_genetics(agent.genetic_core.heart_genetics)
        self._validate_physical_genetics(agent.genetic_core.physical_genetics)
        self._validate_embryo_genetics(agent.genetic_core.embryo_genetics)

    def _validate_mind_genetics(self, mind_genetics):
        required = [
            'creativity', 'learning_efficiency', 'adaptation_rate',
            'memory_retention', 'memory_capacity', 'problem_solving',
            'strategic_thinking', 'cognitive_growth_rate', 'pattern_recognition',
            'development_plasticity', 'risk_assessment', 'social_awareness',
            'innovation_drive', 'curiosity_factor'
        ]
        self._validate_traits(mind_genetics, required, 'mind_genetics')

    def _validate_brain_genetics(self, brain_genetics):
        required = [
            'processing_speed', 'pattern_recognition', 'neural_plasticity',
            'learning_rate', 'memory_capacity', 'decision_speed',
            'multi_tasking', 'spatial_awareness', 'temporal_processing',
            'error_correction'
        ]
        self._validate_traits(brain_genetics, required, 'brain_genetics')

    def _validate_heart_genetics(self, heart_genetics):
        required = [
            'base_rhythm', 'stress_sensitivity', 'resilience',
            'recovery_resilience', 'adaptation_rate', 'trust_baseline',
            'emotional_capacity', 'security_sensitivity'
        ]
        self._validate_traits(heart_genetics, required, 'heart_genetics')

    def _validate_physical_genetics(self, physical_genetics):
        required = [
            'size', 'speed', 'strength', 'energy_efficiency',
            'sensor_sensitivity', 'regeneration_rate', 'immune_system',
            'metabolic_rate', 'longevity_factor', 'adaptation_speed',
            'action_precision'
        ]
        self._validate_traits(physical_genetics, required, 'physical_genetics')

    def _validate_embryo_genetics(self, embryo_genetics):
        required = [
            'development_rate', 'mutation_chance', 'inheritance_factor',
            'trait_stability', 'neural_plasticity', 'cognitive_growth_rate',
            'development_plasticity'
        ]
        self._validate_traits(embryo_genetics, required, 'embryo_genetics')

    def _validate_traits(self, genetics, required_traits, genetics_name):
        missing = [t for t in required_traits if not hasattr(genetics, t)]
        if missing:
            raise ValueError(f"{genetics_name} missing required traits: {missing}")

    def _validate_systems(self, brain: Brain, heart: HeartSystem, mind: Mind) -> None:
        """Validate proper integration of core systems"""
        if not all([brain, heart, mind]):
            raise ValueError("One or more core systems are None")
        
        if not all([hasattr(brain, 'neural_net'), hasattr(heart, 'state'), hasattr(mind, 'embryo')]):
            raise ValueError("Core systems are missing required attributes")

    def _validate_genetic_core(self, genetic_core: GeneticCore) -> None:
        """Validate genetic core"""
        required_traits = [
            'mind_genetics', 'brain_genetics', 'heart_genetics', 'physical_genetics'
        ]
        if not all(hasattr(genetic_core, trait) for trait in required_traits):
            raise ValueError(f"Genetic core missing required traits")

    def _validate_neural_network(self, neural_net: NeuralAdaptiveNetwork) -> None:
        """Validate neural network"""
        required_attributes = [
            'layers', 'state_manager', 'optimizer', 'criterion'
        ]
        if not all(hasattr(neural_net, attr) for attr in required_attributes):
            raise ValueError(f"Neural network missing required attributes")

    def _validate_action_decoder(self, action_decoder: ActionDecoder) -> None:
        """Validate action decoder"""
        if not isinstance(action_decoder, ActionDecoder):
            raise ValueError("Action decoder is not an instance of ActionDecoder")
        if not hasattr(action_decoder, 'action_prototypes'):
            raise ValueError("Action decoder missing action_prototypes attribute")

    def _validate_neural_diagnostics(self, neural_diagnostics: NeuralDiagnostics) -> None:
        """Validate neural diagnostics"""
        if not isinstance(neural_diagnostics, NeuralDiagnostics):
            raise ValueError("Neural diagnostics is not an instance of NeuralDiagnostics")
        if not hasattr(neural_diagnostics, 'diagnostic_history'):
            raise ValueError("Neural diagnostics missing diagnostic_history attribute")

    def _validate_data_augmenter(self, data_augmenter: AdaptiveDataAugmenter) -> None:
        """Validate data augmenter"""
        if not isinstance(data_augmenter, AdaptiveDataAugmenter):
            raise ValueError("Data augmenter is not an instance of AdaptiveDataAugmenter")
        if not hasattr(data_augmenter, 'augmentation_history'):
            raise ValueError("Data augmenter missing augmentation_history attribute")

    def _save_component_states(self, paths: Dict[str, str], **components):
        """Save all component states with comprehensive documentation"""
        timestamp = datetime.now().isoformat().replace(':', '-')  # Windows-safe timestamp
        
        for name, component in components.items():
            component_path = Path(paths[name])
            
            # Save state if component has state_dict
            if hasattr(component, 'state_dict'):
                state_path = component_path / f"state_{timestamp}.pth"
                torch.save(component.state_dict(), str(state_path))
                
            # Save configuration
            if hasattr(component, 'get_config'):
                config_path = component_path / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(component.get_config(), f, indent=2)
                    
            # Save genetic influences
            if hasattr(component, 'get_genetic_traits'):
                traits_path = component_path / "genetic_traits.json"
                with open(traits_path, 'w') as f:
                    json.dump(component.get_genetic_traits(), f, indent=2)

    def _get_genetic_traits(self, agent: AdaptiveAgent) -> Dict:
        """Extract genetic traits for documentation"""
        return {
            'brain': agent.genetic_core.brain_genetics.__dict__,
            'mind': agent.genetic_core.mind_genetics.__dict__,
            'physical': agent.genetic_core.physical_genetics.__dict__,
            'embryo': agent.genetic_core.embryo_genetics.__dict__
        }

    def _get_neural_architecture(self, agent: AdaptiveAgent) -> Dict:
        """Extract neural architecture for documentation"""
        return {
            'input_size': agent.neural_net.input_size,
            'hidden_size': agent.neural_net.hidden_size,
            'output_size': agent.neural_net.output_size
        }

    def _get_birth_record(self, agent: AdaptiveAgent) -> Dict:
        """Extract birth record for documentation"""
        return agent.birth_record.__dict__

    def _create_agent_directory(self, agent_id: str) -> Dict[str, str]:
        """Create full agent directory structure and return paths"""
# Use Path for cross-platform compatibility
        base_path = Path("agents") / str(agent_id)
        
        # Create main subdirectories using proper Windows paths
        directories = {
            'root': str(base_path),
            'genetic': str(base_path / "genetic"),
            'neural': str(base_path / "neural"),
            'mind': str(base_path / "mind"), 
            'brain': str(base_path / "brain"),
            'heart': str(base_path / "heart"),
            'memory': str(base_path / "memory"),
            'actions': str(base_path / "actions"),
            'diagnostics': str(base_path / "diagnostics"),
            'docs': str(base_path / "documentation")
        }

        # Create each directory using Path
        for path in directories.values():
            Path(path).mkdir(parents=True, exist_ok=True)

        # Create additional subdirectories using Path
        (Path(directories['genetic']) / "traits").mkdir(exist_ok=True)
        (Path(directories['neural']) / "checkpoints").mkdir(exist_ok=True)
        (Path(directories['memory']) / "experiences").mkdir(exist_ok=True)
        (Path(directories['actions']) / "history").mkdir(exist_ok=True)
        (Path(directories['diagnostics']) / "logs").mkdir(exist_ok=True)
        
        return directories

    def _generate_documentation(self, agent: AdaptiveAgent, paths: Dict[str, str], verification: ComponentVerification):
        """Generate comprehensive agent documentation"""
        docs_path = Path(paths['docs'])
        
        # Basic agent info
        agent_info = {
            'id': agent.id,
            'birth_id': agent.birth_id,
            'creation_time': datetime.now().isoformat(),
            'gender': agent.gender,
            'parent_ids': agent.parent_ids,
            'verification': verification.__dict__
        }
        
        # Genetic documentation
        genetic_doc = {
            'base_traits': vars(agent.genetic_core.base_traits),
            'mind_genetics': vars(agent.genetic_core.mind_genetics),
            'brain_genetics': vars(agent.genetic_core.brain_genetics),
            'heart_genetics': vars(agent.genetic_core.heart_genetics),
            'physical_genetics': vars(agent.genetic_core.physical_genetics),
            'embryo_genetics': vars(agent.genetic_core.embryo_genetics)
        }
        
        # Neural architecture
        neural_doc = {
            'architecture': {
                'input_size': agent.neural_net.input_size,
                'hidden_size': agent.neural_net.hidden_size,
                'output_size': agent.neural_net.output_size,
            },
            'parameters': {
                'plasticity': agent.neural_net.plasticity,
                'learning_rate': agent.neural_net.learning_rate,
                'adaptation_rate': agent.neural_net.adaptation_rate
            }
        }
        
        # System configurations
        systems_doc = {
            'brain': {
                'reasoning_type': agent.brain.reasoning_type,
                'cognitive_params': agent.brain.cognitive_params
            },
            'mind': {
                'cognitive_metrics': agent.mind.cognitive_metrics,
                'development_stages': agent.mind.development_stages
            },
            'heart': {
                'trust_baseline': agent.heart.trust_baseline,
                'energy_efficiency': agent.heart.energy_efficiency
            }
        }
        
        # Save each documentation component
        documentation = {
            'agent': agent_info,
            'genetics': genetic_doc,
            'neural': neural_doc,
            'systems': systems_doc
        }
        
        # Save main documentation using Windows-safe paths
        doc_path = docs_path / "agent_documentation.json"
        with open(doc_path, 'w') as f:
            json.dump(documentation, f, indent=2)
            
        # Save verification report with Windows-safe paths
        verification_path = docs_path / "verification_report.json"
        verification_doc = {
            'timestamp': datetime.now().isoformat().replace(':', '-'),  # Windows-safe timestamp
            'verification_results': verification.__dict__,
            'validation_checks': self._get_validation_checks(agent)
        }
        
        with open(verification_path, 'w') as f:
            json.dump(verification_doc, f, indent=2)

    def _get_validation_checks(self, agent: AdaptiveAgent) -> Dict:
        """Get detailed validation check results"""
        checks = {}
        
        # Check genetic traits
        checks['genetics'] = {
            category: {
                trait: hasattr(getattr(agent.genetic_core, f"{category}_genetics"), trait)
                for trait in required
            }
            for category, required in {
                'mind': self._get_mind_trait_requirements(),
                'brain': self._get_brain_trait_requirements(),
                'heart': self._get_heart_trait_requirements(),
                'physical': self._get_physical_trait_requirements()
            }.items()
        }
        
        # Check systems
        checks['systems'] = {
            'brain': all(hasattr(agent.brain, attr) for attr in ['neural_net', 'state', 'cognitive_params']),
            'mind': all(hasattr(agent.mind, attr) for attr in ['neural_net', 'memory_system', 'cognitive_metrics']),
            'heart': all(hasattr(agent.heart, attr) for attr in ['trust_baseline', 'energy_efficiency', 'state'])
        }
        
        return checks

    def _get_mind_trait_requirements(self) -> List[str]:
        return [
            'creativity', 'learning_efficiency', 'adaptation_rate',
            'memory_retention', 'memory_capacity', 'problem_solving',
            'strategic_thinking', 'cognitive_growth_rate', 'pattern_recognition',
            'development_plasticity', 'risk_assessment', 'social_awareness',
            'innovation_drive', 'curiosity_factor'
        ]

    def _get_brain_trait_requirements(self) -> List[str]:
        return [
            'processing_speed', 'pattern_recognition', 'neural_plasticity',
            'learning_rate', 'memory_capacity', 'decision_speed',
            'multi_tasking', 'spatial_awareness', 'temporal_processing',
            'error_correction'
        ]
        
    def _get_heart_trait_requirements(self) -> List[str]:
        return [
            'base_rhythm', 'stress_sensitivity', 'resilience',
            'recovery_resilience', 'adaptation_rate', 'trust_baseline',
            'emotional_capacity', 'security_sensitivity'
        ]

    def _get_physical_trait_requirements(self) -> List[str]:
        return [
            'size', 'speed', 'strength', 'energy_efficiency',
            'sensor_sensitivity', 'regeneration_rate', 'immune_system',
            'metabolic_rate', 'longevity_factor', 'adaptation_speed',
            'action_precision'
        ]
    
    def load_agent(self, agent_id: str) -> Optional[AdaptiveAgent]:
        """Load an agent from saved files"""
        try:
            # Construct paths
            base_path = Path("agents") / str(agent_id)
            if not base_path.exists():
                raise FileNotFoundError(f"Agent directory not found: {base_path}")

            # Load documentation to get basic info
            doc_path = base_path / "documentation" / "agent_documentation.json"
            with open(doc_path, 'r') as f:
                documentation = json.load(f)

            # Load genetic core state
            genetic_path = base_path / "genetic" / "state_latest.pth"
            genetic_core = GeneticCore()
            genetic_core.load_state_dict(torch.load(str(genetic_path)))

            # Load neural network
            neural_path = base_path / "neural" / "state_latest.pth"
            neural_net = NeuralAdaptiveNetwork(
                input_size=documentation['neural']['architecture']['input_size'],
                hidden_size=documentation['neural']['architecture']['hidden_size'],
                output_size=documentation['neural']['architecture']['output_size'],
                genetic_core=genetic_core
            )
            neural_net.load_state_dict(torch.load(str(neural_path)))

            # Load memory system
            memory_path = base_path / "memory" / "state_latest.pth"
            memory_system = GeneticMemoryCell(
                input_size=documentation['neural']['architecture']['input_size'],
                hidden_size=documentation['neural']['architecture']['hidden_size'],
                genetic_traits=genetic_core.mind_genetics
            )
            memory_system.load_state_dict(torch.load(str(memory_path)))

            # Create core systems
            brain = Brain(
                neural_net=neural_net,
                genetic_core=genetic_core,
                plasticity=genetic_core.brain_genetics.neural_plasticity,
                learning_rate=genetic_core.brain_genetics.learning_rate
            )

            heart = HeartSystem(
                genetic_core=genetic_core,
                energy_efficiency=genetic_core.physical_genetics.energy_efficiency,
                metabolic_rate=genetic_core.physical_genetics.metabolic_rate
            )

            mind = Mind(
                genetic_core=genetic_core,
                neural_net=neural_net,
                memory_system=memory_system,
                growth_rate=genetic_core.mind_genetics.cognitive_growth_rate
            )

            # Create agent instance
            agent = self.agent_class(
                id=documentation['agent']['id'],
                genetic_core=genetic_core,
                neural_net=neural_net,
                position=documentation.get('position', (0, 0)),
                gender=documentation['agent']['gender'],
                memory_system=memory_system,
                action_system=ActionSystem(
                    neural_net=neural_net,
                    genetic_core=genetic_core,
                    precision=genetic_core.physical_genetics.action_precision,
                    efficiency=genetic_core.physical_genetics.energy_efficiency
                ),
                data_augmenter=AdaptiveDataAugmenter(
                    creativity=genetic_core.mind_genetics.creativity,
                    pattern_recognition=genetic_core.brain_genetics.pattern_recognition
                ),
                neural_diagnostics=NeuralDiagnostics(
                    check_frequency=int(100 * genetic_core.brain_genetics.processing_speed),
                    alert_threshold=0.8 * genetic_core.mind_genetics.adaptation_rate
                ),
                birth_record=self.birth_registry.get_record(documentation['agent']['birth_id']),
                parent_ids=documentation['agent']['parent_ids']
            )

            # Attach systems
            agent.brain = brain
            agent.heart = heart
            agent.mind = mind
            agent.birth_id = documentation['agent']['birth_id']

            logger.info(f"Successfully loaded agent {agent_id}")
            return agent

        except Exception as e:
            logger.error(f"Failed to load agent {agent_id}: {str(e)}", exc_info=True)
            return None