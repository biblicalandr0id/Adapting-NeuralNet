from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import json
import os
import uuid
import logging
from genetics import GeneticCore

logger = logging.getLogger(__name__)

@dataclass
class BirthRecord:
    """Record of an agent's birth details"""
    id: str
    parent_id: Optional[str]
    genetic_core: GeneticCore
    position: Tuple[float, float]
    gender: str
    birth_time: datetime = field(default_factory=datetime.now)
    generation: int = 0
    genetic_heritage: List[Dict] = field(default_factory=list)
    neural_adaptations: List[Dict] = field(default_factory=list)
    significant_mutations: List[Dict] = field(default_factory=list)

class BirthRegistry:
    """Manages birth records for all agents"""
    
    def __init__(self, save_dir: str = 'birth_records'):
        self.records: Dict[str, BirthRecord] = {}
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def register_birth(self, genetic_core: GeneticCore, position: Tuple[float, float],
                      gender: str, parent_id: Optional[str] = None) -> BirthRecord:
        """Register a new birth with enhanced genetic and neural tracking"""
        record_id = str(uuid.uuid4())
        
        # Get parent record if exists
        parent_record = self.records.get(parent_id) if parent_id else None
        generation = (parent_record.generation + 1) if parent_record else 0
        
        # Track genetic heritage
        genetic_heritage = []
        if parent_record:
            genetic_heritage.extend(parent_record.genetic_heritage)
            
        # Add current genetic state
        genetic_heritage.append({
            'time': datetime.now().isoformat(),
            'traits': genetic_core.get_all_traits(),
            'potential': genetic_core.calculate_potential(),
            'generation': generation
        })
        
        # Track significant mutations
        significant_mutations = []
        if parent_record:
            mutations = genetic_core.get_significant_mutations(parent_record.genetic_core)
            if mutations:
                significant_mutations.extend(mutations)
        
        # Create birth record
        record = BirthRecord(
            id=record_id,
            parent_id=parent_id,
            genetic_core=genetic_core,
            position=position,
            gender=gender,
            generation=generation,
            genetic_heritage=genetic_heritage,
            significant_mutations=significant_mutations
        )
        
        # Save record
        self.records[record_id] = record
        self._save_record(record)
        
        return record
        
    def _save_record(self, record: BirthRecord):
        """Save birth record with genetic and neural information"""
        record_path = os.path.join(self.save_dir, f"{record.id}.json")
        
        record_data = {
            'id': record.id,
            'parent_id': record.parent_id,
            'birth_time': record.birth_time.isoformat(),
            'generation': record.generation,
            'position': record.position,
            'gender': record.gender,
            'genetic_heritage': record.genetic_heritage,
            'significant_mutations': record.significant_mutations,
            'genetic_traits': {
                'mind': record.genetic_core.mind_genetics.__dict__,
                'brain': record.genetic_core.brain_genetics.__dict__,
                'heart': record.genetic_core.heart_genetics.__dict__,
                'physical': record.genetic_core.physical_genetics.__dict__
            }
        }
        
        with open(record_path, 'w') as f:
            json.dump(record_data, f, indent=2)

    def get_lineage(self, agent_id: str) -> List[BirthRecord]:
        """Get complete lineage for an agent"""
        lineage = []
        current_id = agent_id
        
        while current_id:
            record = self.records.get(current_id)
            if not record:
                break
            lineage.append(record)
            current_id = record.parent_id
            
        return lineage
        
    def analyze_evolution(self, agent_id: str) -> Dict:
        """Analyze evolutionary progress through generations"""
        lineage = self.get_lineage(agent_id)
        if not lineage:
            return {}
            
        analysis = {
            'generations': len(lineage),
            'trait_evolution': {},
            'significant_mutations': [],
            'genetic_potential_trend': []
        }
        
        # Analyze trait evolution
        for record in lineage:
            # Track genetic potential
            potential = record.genetic_core.calculate_potential()
            analysis['genetic_potential_trend'].append({
                'generation': record.generation,
                'potential': potential
            })
            
            # Track significant mutations
            analysis['significant_mutations'].extend(record.significant_mutations)
            
        return analysis