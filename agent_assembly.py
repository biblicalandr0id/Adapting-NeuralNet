from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import logging
import uuid
from datetime import datetime
import time
from genetics import GeneticCore
from neural_networks import NeuralAdaptiveNetwork
from agent import AdaptiveAgent
from brain import Brain, BrainState
from heart import HeartSystem, HeartState
from mind import Mind, create_embryo
from birth_records import BirthRecord, BirthRegistry
from action_system import ActionDecoder
import random  # Add this import for _determine_gender

logger = logging.getLogger(__name__)

@dataclass
class AssemblyResult:
    """Contains the assembled agent and metadata"""
    agent: Optional[AdaptiveAgent]
    success: bool
    error_message: Optional[str] = None
    assembly_time: float = 0.0
    metadata: Dict = None

class AgentAssembler:
    """Assembles agents from components with validation"""
    
    def __init__(self, birth_registry: BirthRegistry):
        self.birth_registry = birth_registry
        self.assembly_stats = {
            'successful_assemblies': 0,
            'failed_assemblies': 0,
            'mutation_rates': [],
            'assembly_times': []
        }
        
    def create_agent(self, 
                    position: Tuple[float, float],
                    parent: Optional[AdaptiveAgent] = None,
                    gender: Optional[str] = None) -> AssemblyResult:
        start_time = time.time()
        try:
            # 1. Create genetic core
            genetic_core = (
                parent.genetic_core.create_offspring() if parent 
                else GeneticCore()
            )
            
            # 2. Create neural network with architecture based on genetics
            network_params = self._calculate_network_architecture(genetic_core)
            neural_net = NeuralAdaptiveNetwork(**network_params)
            
            if parent:
                neural_net.inherit_weights(
                    parent.neural_net,
                    mutation_rate=genetic_core.mind_genetics.creativity * 0.1,
                    adaptation_rate=genetic_core.mind_genetics.adaptation_rate
                )
            
            # 3. Create core systems
            brain = Brain(genetic_core.brain_genetics)
            heart = HeartSystem(genetic_core.heart_genetics)
            mind = Mind(
                embryo=create_embryo(genetic_core),
                brain_interface=brain.interface
            )
            
            # 4. Generate birth record
            birth_record = BirthRecord(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                genetic_traits=genetic_core.get_all_traits(),
                parent_id=parent.id if parent else None,
                position=position,
                gender=gender or self._determine_gender()
            )
            
            # 5. Register birth
            self.birth_registry.register_birth(birth_record)
            
            # 6. Create agent
            agent = AdaptiveAgent(
                genetic_core=genetic_core,
                neural_net=neural_net,
                position=position,
                gender=birth_record.gender,
                birth_record=birth_record,
                parent_ids={'parent': parent.id} if parent else {}
            )
            
            # 7. Initialize core systems
            agent.brain = brain
            agent.heart = heart 
            agent.mind = mind
            
            # 8. Initialize additional systems
            self._initialize_memory_system(agent, genetic_core)
            self._setup_action_system(agent)
            self._initialize_diagnostics(agent)
            
            # 9. Validate assembly
            self._validate_assembly(agent)
            self._validate_systems(agent)
            
            # Update assembly stats
            self.assembly_stats['successful_assemblies'] += 1
            self.assembly_stats['assembly_times'].append(time.time() - start_time)
            
            return AssemblyResult(
                agent=agent,
                success=True,
                assembly_time=time.time() - start_time,
                metadata={
                    'birth_id': birth_record.id,
                    'genetic_potential': genetic_core.calculate_potential(),
                    'network_complexity': len(neural_net.parameters()),
                    'inheritance': bool(parent),
                    'memory_size': agent.neural_net.memory_size,  # Updated to use neural net memory
                    'action_count': len(agent.actions)
                }
            )
            
        except Exception as e:
            self.assembly_stats['failed_assemblies'] += 1
            logger.error(f"Agent assembly failed: {str(e)}")
            return AssemblyResult(
                agent=None,
                success=False,
                error_message=str(e),
                assembly_time=time.time() - start_time
            )
    
    def _calculate_network_architecture(self, genetic_core: GeneticCore) -> Dict:
        """Calculate neural network architecture based on genetics"""
        processing_speed = genetic_core.brain_genetics.processing_speed
        plasticity = genetic_core.brain_genetics.neural_plasticity
        
        return {
            'input_size': int(32 * processing_speed),
            'hidden_size': int(64 * plasticity),
            'output_size': 16,
            'num_layers': max(2, int(3 * plasticity)),
            'learning_rate': genetic_core.mind_genetics.learning_efficiency * 0.01
        }
    
    def _determine_gender(self) -> str:
        """Randomly determine agent gender"""
        return 'female' if random.random() < 0.5 else 'male'
        
    def _validate_assembly(self, agent: AdaptiveAgent) -> None:
        """Validate all required components are present and initialized"""
        required = [
            'genetic_core', 'neural_net', 'brain', 'heart', 
            'mind', 'position', 'birth_record'
        ]
        
        missing = [attr for attr in required if not hasattr(agent, attr)]
        
        if missing:
            raise ValueError(f"Agent missing required components: {missing}")

    def _initialize_memory_system(self, agent: AdaptiveAgent, genetic_core: GeneticCore) -> None:
        """Configure neural network memory parameters based on genetics"""
        memory_capacity = genetic_core.mind_genetics.memory_capacity
        learning_rate = genetic_core.mind_genetics.learning_efficiency
        
        # Configure neural network memory
        agent.neural_net.configure_memory(
            memory_size=int(128 * memory_capacity),
            attention_heads=max(1, int(4 * memory_capacity)),
            learning_rate=learning_rate * 0.01,
            forget_rate=1.0 - (memory_capacity * 0.5)  # Higher capacity = lower forget rate
        )

    def _setup_action_system(self, agent: AdaptiveAgent) -> None:
        """Initialize action system with basic capability frameworks"""
        action_decoder = ActionDecoder(hidden_size=32)
        agent.action_decoder = action_decoder
        
        # Initialize action prototypes - actual implementations are in AdaptiveAgent
        action_prototypes = {
            'move': torch.randn(32),      # Movement prototype vector
            'gather': torch.randn(32),    # Gathering prototype vector
            'process': torch.randn(32)    # Processing prototype vector
        }
        
        # Register action prototypes with decoder
        for name, prototype in action_prototypes.items():
            agent.action_decoder.add_action(
                name=name,
                prototype_vector=prototype,
                method=getattr(agent, f"_{name}_action")  # Link to agent's implementation
            )

    def _initialize_diagnostics(self, agent: AdaptiveAgent) -> None:
        """Setup diagnostic systems"""
        agent.neural_diagnostics = NeuralDiagnostics(
            check_frequency=100,
            alert_threshold=0.8
        )
        agent.agent_diagnostics = AgentDiagnostics(
            track_metrics=['energy', 'age', 'resources'],
            log_frequency=50
        )

    def _validate_systems(self, agent: AdaptiveAgent) -> None:
        """Perform extended validation of agent systems"""
        validations = {
            'neural_network': lambda: agent.neural_net(
                torch.randn(1, agent.neural_net.input_size)
            ),
            'action_system': lambda: len(agent.actions) > 0,
            'brain_interface': lambda: agent.brain.interface is not None,
            'heart_state': lambda: agent.heart.state is not None,
            'mind_embryo': lambda: agent.mind.embryo is not None,
            'diagnostics': lambda: all([
                hasattr(agent, 'neural_diagnostics'),
                hasattr(agent, 'agent_diagnostics')
            ])
        }
        
        failures = []
        for system, validation in validations.items():
            try:
                if not validation():
                    failures.append(f"{system} validation failed")
            except Exception as e:
                failures.append(f"{system} error: {str(e)}")
        
        if failures:
            raise ValueError(f"System validation failures: {', '.join(failures)}")