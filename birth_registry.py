import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BirthRecord:
    id: str
    parent_id: Optional[str]
    timestamp: datetime
    generation: int
    genetic_heritage: List[str] = None
    
    def __post_init__(self):
        if self.genetic_heritage is None:
            self.genetic_heritage = []

class BirthRegistry:
    def __init__(self):
        self.records: Dict[str, BirthRecord] = {}
        self.lineage_depth = 5
        
    def create_record(self, parent_id: Optional[str] = None, generation: int = 0) -> BirthRecord:
        """Create a new birth record"""
        record = BirthRecord(
            id=str(uuid.uuid4())[:8],
            parent_id=parent_id,
            timestamp=datetime.now(),
            generation=generation
        )
        
        # Add parent's heritage if exists
        if parent_id and parent_id in self.records:
            parent_record = self.records[parent_id]
            record.genetic_heritage = (
                parent_record.genetic_heritage[-self.lineage_depth:] + [parent_id]
            )
            
        self.records[record.id] = record
        logger.info(f"Created birth record: {record.id} (Gen: {generation})")
        return record
    
    def get_lineage(self, agent_id: str) -> List[str]:
        """Get agent's lineage history"""
        if agent_id not in self.records:
            return []
            
        return self.records[agent_id].genetic_heritage
    
    def get_generation(self, agent_id: str) -> int:
        """Get agent's generation number"""
        if agent_id not in self.records:
            return 0
            
        return self.records[agent_id].generation