from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import json
import os
import uuid
import logging
from agent import AdaptiveAgent
logger = logging.getLogger(__name__)

@dataclass
class BirthRecord:
    """Records the birth details of an agent"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    gender: str = ""
    birth_time: datetime = field(default_factory=datetime.now)
    parent_ids: Dict[str, str] = field(default_factory=dict)
    genetic_traits: Dict = field(default_factory=dict)
    generation: int = 1
    birth_location: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    species_type: str = "agent"  # Can be "agent" or "predator"
    
    def to_dict(self) -> Dict:
        """Convert birth record to dictionary format"""
        return {
            'id': self.id,
            'name': self.name,
            'gender': self.gender,
            'birth_time': self.birth_time.isoformat(),
            'parent_ids': self.parent_ids,
            'genetic_traits': self.genetic_traits,
            'generation': self.generation,
            'birth_location': self.birth_location,
            'species_type': self.species_type
        }

class BirthRegistry:
    """Manages birth records for the simulation"""
    def __init__(self, registry_dir: str = 'birth_records'):
        self.registry_dir = registry_dir
        self.records: Dict[str, BirthRecord] = {}
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create registry directory if it doesn't exist"""
        if not os.path.exists(self.registry_dir):
            os.makedirs(self.registry_dir)
            logger.info(f"Created birth registry directory: {self.registry_dir}")
    
    def register_birth(self, record: BirthRecord) -> bool:
        """Register a new birth record"""
        try:
            # Store in memory
            self.records[record.id] = record
            
            # Save to file
            date_dir = os.path.join(
                self.registry_dir,
                record.birth_time.strftime('%Y%m%d')
            )
            os.makedirs(date_dir, exist_ok=True)
            
            filepath = os.path.join(date_dir, f"{record.id}.json")
            with open(filepath, 'w') as f:
                json.dump(record.to_dict(), f, indent=2)
            
            logger.info(f"Registered birth: {record.name} ({record.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register birth: {str(e)}")
            return False
    
    def get_record(self, record_id: str) -> Optional[BirthRecord]:
        """Retrieve a birth record by ID"""
        return self.records.get(record_id)
    
    def get_lineage(self, record_id: str) -> List[BirthRecord]:
        """Get the complete lineage for a given record"""
        lineage = []
        current = self.get_record(record_id)
        
        while current:
            lineage.append(current)
            # Follow maternal line (can be modified to include paternal)
            mother_id = current.parent_ids.get('mother')
            current = self.get_record(mother_id) if mother_id else None
            
        return lineage
    
    def get_generation_stats(self) -> Dict:
        """Get statistics about generations"""
        stats = {}
        for record in self.records.values():
            if record.generation not in stats:
                stats[record.generation] = {
                    'count': 0,
                    'male_count': 0,
                    'female_count': 0,
                    'avg_genetic_fitness': 0.0
                }
            
            gen_stats = stats[record.generation]
            gen_stats['count'] += 1
            if record.gender == 'male':
                gen_stats['male_count'] += 1
            else:
                gen_stats['female_count'] += 1
                
        return stats
    
    def backup_records(self) -> bool:
        """Create a backup of all birth records"""
        try:
            backup_dir = os.path.join(
                self.registry_dir,
                f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            os.makedirs(backup_dir)
            
            for record in self.records.values():
                filepath = os.path.join(backup_dir, f"{record.id}.json")
                with open(filepath, 'w') as f:
                    json.dump(record.to_dict(), f, indent=2)
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup records: {str(e)}")
            return False

# Example usage:
birth_registry = BirthRegistry()

# When creating a new agent
birth_record = BirthRecord(
    name="Agent_001",
    gender="female",
    parent_ids={'mother': 'mother_id', 'father': 'father_id'},
    genetic_traits=agent.genetic_core.get_all_traits(),
    birth_location=agent.position
)

# Register the birth
birth_registry.register_birth(birth_record)

# Later, get lineage information
lineage = birth_registry.get_lineage(birth_record.id)