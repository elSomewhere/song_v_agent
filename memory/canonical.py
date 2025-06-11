"""
Canonical memory management - loads and parses static documents.
"""
from pathlib import Path
from typing import Dict, Any, List
import re


class CanonicalMemory:
    """Manages static canonical memory from base documents."""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.style_content = self._load_style()
        self.entities_content = self._load_entities()
        self.entity_appearances = self._parse_entity_appearances()
        
    def _load_style(self) -> str:
        """Load the style guide document."""
        style_path = self.data_path / "style.md"
        if style_path.exists():
            return style_path.read_text(encoding='utf-8')
        return ""
    
    def _load_entities(self) -> str:
        """Load the entities document."""
        entities_path = self.data_path / "entites.md"  # Note: filename has typo in original
        if entities_path.exists():
            return entities_path.read_text(encoding='utf-8')
        # Try correct spelling as fallback
        entities_path = self.data_path / "entities.md"
        if entities_path.exists():
            return entities_path.read_text(encoding='utf-8')
        return ""
    
    def _parse_entity_appearances(self) -> Dict[str, Dict[str, str]]:
        """Parse entity appearances from the entities document."""
        entities = {}
        current_entity = None
        current_section = None
        
        lines = self.entities_content.split('\n')
        
        for line in lines:
            # Check for entity header (e.g., "## Helena Carter")
            entity_match = re.match(r'^##\s+(.+)$', line)
            if entity_match:
                current_entity = entity_match.group(1).strip()
                entities[current_entity] = {
                    'name': current_entity,
                    'description': '',
                    'appearance': '',
                    'personality': '',
                    'role': ''
                }
                current_section = 'description'
                continue
            
            # Check for section headers
            if current_entity:
                if re.match(r'^###\s+Physical Appearance', line, re.IGNORECASE):
                    current_section = 'appearance'
                    continue
                elif re.match(r'^###\s+Personality', line, re.IGNORECASE):
                    current_section = 'personality'
                    continue
                elif re.match(r'^###\s+Role', line, re.IGNORECASE):
                    current_section = 'role'
                    continue
                
                # Add content to current section
                if current_section and line.strip():
                    entities[current_entity][current_section] += line + '\n'
        
        # Clean up whitespace
        for entity in entities.values():
            for key in entity:
                if isinstance(entity[key], str):
                    entity[key] = entity[key].strip()
        
        return entities
    
    def get_entity_appearance(self, entity_name: str) -> Dict[str, str]:
        """Get appearance details for a specific entity."""
        # Try exact match first
        if entity_name in self.entity_appearances:
            return self.entity_appearances[entity_name]
        
        # Try case-insensitive match
        for name, details in self.entity_appearances.items():
            if name.lower() == entity_name.lower():
                return details
        
        # Try partial match
        for name, details in self.entity_appearances.items():
            if entity_name.lower() in name.lower() or name.lower() in entity_name.lower():
                return details
        
        return {
            'name': entity_name,
            'description': f'Unknown entity: {entity_name}',
            'appearance': '',
            'personality': '',
            'role': ''
        }
    
    def get_all_entities(self) -> List[str]:
        """Get list of all known entities."""
        return list(self.entity_appearances.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert canonical memory to dictionary format."""
        return {
            'style': self.style_content,
            'entities': self.entity_appearances,
            'entity_names': self.get_all_entities()
        } 