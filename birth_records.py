from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import json
import os
import uuid
import logging

logger = logging.getLogger(__name__)

@dataclass
class BirthRecord:
    """Record of an agent's birth details"""
    id: str
    timestamp: datetime
    genetic_traits: Dict
    parent_id: Optional[str]
    position: tuple[float, float]
    gender: str
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert birth record to dictionary format"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'genetic_traits': self.genetic_traits,
            'parent_id': self.parent_id,
            'position': self.position,
            'gender': self.gender,
            'metadata': self.metadata or {}
        }

class BirthRegistry:
    """Manages birth records for all agents"""
    
    def __init__(self, save_dir: str = 'birth_records'):
        self.save_dir = save_dir
        self.records: Dict[str, BirthRecord] = {}
        os.makedirs(save_dir, exist_ok=True)
        
    def register_birth(self, record: BirthRecord) -> None:
        """Register a new birth record"""
        self.records[record.id] = record
        self._save_record(record)
        
    def get_record(self, agent_id: str) -> Optional[BirthRecord]:
        """Retrieve a birth record by agent ID"""
        return self.records.get(agent_id)
        
    def _save_record(self, record: BirthRecord) -> None:
        """Save birth record to file"""
        try:
            filename = f"{self.save_dir}/{record.id}.json"
            with open(filename, 'w') as f:
                json.dump(record.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save birth record: {str(e)}")
            
    def load_records(self) -> None:
        """Load all birth records from files"""
        try:
            for filename in os.listdir(self.save_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.save_dir, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        record = BirthRecord(
                            id=data['id'],
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            genetic_traits=data['genetic_traits'],
                            parent_id=data['parent_id'],
                            position=tuple(data['position']),
                            gender=data['gender'],
                            metadata=data.get('metadata')
                        )
                        self.records[record.id] = record
        except Exception as e:
            logger.error(f"Failed to load birth records: {str(e)}")

    def get_lineage(self, agent_id: str) -> list[BirthRecord]:
        """Get the lineage chain for an agent"""
        lineage = []
        current_id = agent_id
        
        while current_id:
            record = self.get_record(current_id)
            if not record:
                break
            lineage.append(record)
            current_id = record.parent_id
            
        return lineage

    def get_generation_stats(self) -> Dict:
        """Get statistics about generations"""
        stats = {
            'total_births': len(self.records),
            'generations': {},
            'gender_ratio': {'male': 0, 'female': 0}
        }
        
        for record in self.records.values():
            # Count generations
            generation = len(self.get_lineage(record.id))
            stats['generations'][generation] = stats['generations'].get(generation, 0) + 1
            
            # Track gender ratio
            stats['gender_ratio'][record.gender] += 1
            
        return stats