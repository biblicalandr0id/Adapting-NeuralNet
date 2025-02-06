# heart.py
import hashlib
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field
import threading
import logging
#from watchdog.observers import Observer # Removed watchdog imports
#from watchdog.events import FileSystemEventHandler #Removed watchdog imports
import shutil

logger = logging.getLogger(__name__)

@dataclass
class SecurityMemory:
    """Stores security experiences and trust levels"""
    trusted_patterns: Dict[str, float] = field(default_factory=dict)
    threat_patterns: Dict[str, float] = field(default_factory=dict)
    interaction_history: List[Dict] = field(default_factory=list)
    component_checksums: Dict[str, str] = field(default_factory=dict)

@dataclass
class HeartState:
    """Represents the current state of the heart system"""
    rhythm: float = 1.0
    energy_level: float = 100.0
    stress_level: float = 0.0
    recovery_rate: float = 1.0
    health: float = 100.0
    adaptation_progress: float = 0.0
    trust_baseline: float = 0.5  # Base trust level
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        'fear': 0.0,
        'trust': 0.5,
        'cooperation': 0.5,
        'aggression': 0.0
    })

class HeartSecurity:
    def __init__(self, component_paths: Dict[str, str]):
        """
        Initialize heart security system
        component_paths: Dictionary mapping component names to their file paths
        """
        self.memory = SecurityMemory()
        self.component_paths = {name: os.path.abspath(path) for name, path in component_paths.items()} # Make component paths absolute
        self.trust_threshold = 0.6 # Lowered trust threshold
        self.threat_threshold = 0.4 # Increased threat threshold to allow more interactions
        self.learning_rate = 0.1
        self.is_alive = True
        self.last_integrity_check = time.time() # Record the time of the last integrity check
        self.integrity_check_interval = 5 # Check component integrity every 5 seconds
        
        # Initialize component monitoring
        self.setup_component_monitoring()
        
        # Start security threads
        self.start_security_monitoring()
        
    def setup_component_monitoring(self):
        """Set up initial component verification"""
        for name, path in self.component_paths.items():
            if os.path.exists(path):
                self.memory.component_checksums[name] = self.calculate_checksum(path)
            else:
                logging.error(f"Component {name} not found at {path}")
                if self.is_alive:
                  self.self_destruct()
                return
                
    def calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def evaluate_interaction(self, interaction_data: Dict) -> Tuple[bool, float]:
        """Evaluate if an interaction is safe based on past experiences"""
        interaction_pattern = str(sorted(interaction_data.items()))
        
        trust_score = self.memory.trusted_patterns.get(interaction_pattern, 0.5)
        threat_score = self.memory.threat_patterns.get(interaction_pattern, 0.5)
        
        # Adjusted safety score calculation to be less conservative
        safety_score = (trust_score * 0.7)  + (1 - threat_score * 0.3) # Give greater weight to trust scores
        is_safe = safety_score > self.trust_threshold
        
        return is_safe, safety_score
    
    def learn_from_outcome(self, interaction_data: Dict, was_positive: bool, impact_level: float):
        """Update trust/threat patterns based on interaction outcomes"""
        interaction_pattern = str(sorted(interaction_data.items()))
        
        if was_positive:
            current_trust = self.memory.trusted_patterns.get(interaction_pattern, 0.5)
            self.memory.trusted_patterns[interaction_pattern] = min(
                1.0, current_trust + (self.learning_rate * impact_level * 1.2) # Increased learning for positive
            )
            # If negative interactions were reduced, slowly make them be seen as not as bad
            current_threat = self.memory.threat_patterns.get(interaction_pattern, 0.5)
            self.memory.threat_patterns[interaction_pattern] = max(
                0.0, current_threat - (self.learning_rate * impact_level * 0.1)
            )
        else:
            current_threat = self.memory.threat_patterns.get(interaction_pattern, 0.5)
            self.memory.threat_patterns[interaction_pattern] = min(
                1.0, current_threat + (self.learning_rate * impact_level)
            )
            # If positive interactions were reduced, slowly make them be seen as not as good
            current_trust = self.memory.trusted_patterns.get(interaction_pattern, 0.5)
            self.memory.trusted_patterns[interaction_pattern] = max(
                0.0, current_trust - (self.learning_rate * impact_level * 0.1)
            )
            
        # Store interaction history
        self.memory.interaction_history.append({
            "interaction": interaction_data,
            "outcome": "positive" if was_positive else "negative",
            "impact": impact_level,
            "timestamp": time.time()
        })
        
    def verify_component_integrity(self) -> bool:
        """Verify all components are present and unmodified"""
        for name, path in self.component_paths.items():
            if not os.path.exists(path):
                logging.error(f"Component {name} missing")
                return False
                
            current_checksum = self.calculate_checksum(path)
            if current_checksum != self.memory.component_checksums[name]:
                logging.error(f"Component {name} has been modified")
                return False
                
        return True
    
    def self_destruct(self):
        """Initiate self-destruct sequence"""
        logging.warning("Self-destruct sequence initiated")
        self.is_alive = False
        
        # Remove all component files
        for path in self.component_paths.values():
            try:
                if os.path.exists(path):
                    secure_delete(path)
            except Exception as e:
                logging.error(f"Error during self-destruct: {e}")
            
    def start_security_monitoring(self):
        """Start background security monitoring threads"""
        self.monitor_thread = threading.Thread(target=self._security_monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        self.clean_thread = threading.Thread(target=self._clean_interaction_history_loop)
        self.clean_thread.daemon = True
        self.clean_thread.start()
        
    def _security_monitor_loop(self):
        """Continuous security monitoring loop"""
        while self.is_alive:
            try:
                current_time = time.time()
                if current_time - self.last_integrity_check > self.integrity_check_interval:
                    if not self.verify_component_integrity():
                        self.self_destruct()
                        break
                    self.last_integrity_check = current_time
                time.sleep(1)  # Check interval
            except Exception as e:
                logging.error(f"Error in security monitoring loop: {e}")
                self.self_destruct()
                break
    def _clean_interaction_history_loop(self):
      """Continuously clean old interactions"""
      while self.is_alive:
        try:
          self._clean_old_interactions()
          time.sleep(10) #Clean every 10 seconds
        except Exception as e:
          logging.error(f"Error cleaning old interactions: {e}")
          
    def _clean_old_interactions(self, max_age: float = 3600):
        """Remove old interactions from history"""
        current_time = time.time()
        self.memory.interaction_history = [
            interaction for interaction in self.memory.interaction_history
            if current_time - interaction["timestamp"] < max_age
        ]
        
    def process_interaction(self, interaction_data: Dict) -> Tuple[bool, str]:
        """Process and evaluate an incoming interaction"""
        if not self.is_alive:
            return False, "System is not active"
            
        is_safe, safety_score = self.evaluate_interaction(interaction_data)
        
        if not is_safe:
            logging.warning(f"Potentially unsafe interaction detected: {interaction_data}")
            return False, f"Interaction blocked (safety score: {safety_score:.2f})"
            
        return True, f"Interaction approved (safety score: {safety_score:.2f})"

# Removed ComponentChangeHandler class

def secure_delete(path: str):
    """Securely delete a file by overwriting with random data"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        with open(path, "wb") as f:
            f.write(os.urandom(size))
        os.remove(path)

def create_heart_security(component_paths: Dict[str, str]) -> HeartSecurity:
    """Factory function to create heart security system"""
    return HeartSecurity(component_paths)

class HeartSystem:
    """Manages the heart functions and emotional state of an agent"""
    def __init__(self, genetics):
        self.genetics = genetics
        self.state = HeartState()
        self.memory = []  # Stores recent heart events
        self.adaptation_threshold = 0.7
        self.trust_memory = []  # Track trust-related interactions
        
    def update(self, environment_state: Dict, agent_activity: float) -> Dict:
        """Update heart state and emotional responses"""
        try:
            # Calculate base metrics
            stress_factor = self._calculate_stress(environment_state, agent_activity)
            self.state.rhythm = self._adjust_rhythm(stress_factor)
            
            # Update trust and emotional state
            self._update_emotional_state(environment_state)
            
            # Calculate energy consumption
            energy_consumption = self._calculate_energy_consumption(
                self.state.rhythm, 
                agent_activity
            )
            self.state.energy_level -= energy_consumption
            
            # Recovery if needed
            if self.state.stress_level > self.adaptation_threshold:
                self._trigger_recovery()
            
            self._apply_recovery()
            
            return self._get_heart_metrics()
            
        except Exception as e:
            logger.error(f"Heart update error: {str(e)}")
            return {}
    
    def _update_emotional_state(self, env_state: Dict) -> None:
        """Update emotional state based on environment and genetics"""
        # Update trust baseline based on experiences
        if self.trust_memory:
            recent_experiences = self.trust_memory[-10:]  # Last 10 interactions
            trust_adjustment = sum(exp['outcome'] for exp in recent_experiences) / len(recent_experiences)
            self.state.trust_baseline = max(0.1, min(1.0, 
                self.state.trust_baseline + trust_adjustment * self.genetics.heart_genetics.adaptation_rate
            ))
        
        # Update other emotional states
        threat_level = len(env_state.get('threats', []))
        self.state.emotional_state['fear'] = min(1.0, threat_level * 0.2)
        self.state.emotional_state['aggression'] = max(0.0, 
            self.state.stress_level - self.genetics.heart_genetics.resilience
        )
    
    def record_interaction(self, interaction_type: str, outcome: float) -> None:
        """Record social interaction outcomes for trust adaptation"""
        self.trust_memory.append({
            'type': interaction_type,
            'outcome': outcome,
            'timestamp': datetime.now()
        })
        
        # Trim old memories
        if len(self.trust_memory) > 100:
            self.trust_memory = self.trust_memory[-100:]

class EmotionalCore:
    """Deep emotional processing system"""
    def __init__(self, heart_system: HeartSystem):
        self.heart = heart_system
        self.emotional_memory = []
        self.bonding_threshold = 0.7
        self.emotional_states = {
            'joy': 0.0,
            'sadness': 0.0,
            'fear': 0.0,
            'anger': 0.0,
            'curiosity': 0.5,
            'attachment': 0.0
        }

# Example usage:
if __name__ == "__main__":
    components = {
        "mind": "mind.py",
        "dna": "dna.py",
        "brain": "brainui.py"
    }
    
    heart = create_heart_security(components)
    
    # Example interaction
    interaction = {
        "type": "external_input",
        "source": "user_interface",
        "action": "modify_thought_patterns"
    }
    
    is_allowed, message = heart.process_interaction(interaction)
    print(f"Interaction allowed: {is_allowed}, Message: {message}")