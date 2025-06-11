"""
Script parsing utilities for extracting scenes from markdown files.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import re


class ScriptParser:
    """Parses script files to extract individual scenes."""
    
    def __init__(self, script_path: str = "data/script.md"):
        self.script_path = Path(script_path)
        self.scenes = self._parse_script()
        
    def _parse_script(self) -> List[Dict[str, Any]]:
        """Parse the script file into individual scenes."""
        
        if not self.script_path.exists():
            print(f"Script file not found: {self.script_path}")
            return []
        
        content = self.script_path.read_text(encoding='utf-8')
        
        # Split by shot markers (e.g., "1. **Shot 1**", "Shot 1:", "SHOT 1:", etc.)
        # More flexible pattern to match various formats including markdown bold and numbered lists
        shot_pattern = r'(?:^|\n)\s*\d*\.?\s*\*?\*?(?:Shot|SHOT)\s*(\d+)\*?\*?\s*'
        
        scenes = []
        matches = list(re.finditer(shot_pattern, content, re.MULTILINE | re.IGNORECASE))
        
        for i, match in enumerate(matches):
            shot_number = int(match.group(1))
            
            # Get the content until the next shot or end of file
            start_pos = match.end()
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            
            shot_content = content[start_pos:end_pos].strip()
            
            # Extract title from the first line of content if it exists
            lines = shot_content.split('\n')
            shot_title = ""
            if lines:
                # Skip empty lines and extract bullet points
                for line in lines:
                    if line.strip().startswith('- **Action/Story Detail**'):
                        break
                    if line.strip():
                        shot_title = line.strip()
                        break
            
            # Extract entities from the scene
            entities = self._extract_entities(shot_content)
            
            scenes.append({
                'shot_id': shot_number,
                'title': shot_title,
                'text': shot_content,
                'entities': entities
            })
        
        return scenes
    
    def _extract_entities(self, scene_text: str) -> List[str]:
        """Extract entity names from scene text."""
        
        entities = []
        
        # Look for capitalized words that might be names
        # Exclude common words that are often capitalized
        exclude_words = {
            'The', 'A', 'An', 'In', 'On', 'At', 'To', 'From', 'With',
            'But', 'And', 'Or', 'As', 'If', 'Then', 'Shot', 'SHOT',
            'INT', 'EXT', 'CUT', 'FADE', 'CLOSE', 'WIDE'
        }
        
        # Find potential entity names (capitalized words)
        words = scene_text.split()
        for word in words:
            # Clean punctuation
            clean_word = word.strip('.,!?;:\'"')
            
            # Check if it's a capitalized word and not in exclude list
            if (clean_word and 
                clean_word[0].isupper() and 
                clean_word not in exclude_words and
                not clean_word.isupper()):  # Exclude all-caps words
                
                if clean_word not in entities:
                    entities.append(clean_word)
        
        # Also look for explicit character mentions
        # Pattern for "Character Name" in dialogue or action
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        
        for match in re.finditer(name_pattern, scene_text):
            name = match.group(1)
            if name not in entities and name not in exclude_words:
                # Check if it might be a character name (2-3 words max)
                word_count = len(name.split())
                if 1 <= word_count <= 3:
                    entities.append(name)
        
        return entities
    
    def get_scene(self, shot_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific scene by shot ID."""
        
        for scene in self.scenes:
            if scene['shot_id'] == shot_id:
                return scene
        
        return None
    
    def get_scene_count(self) -> int:
        """Get the total number of scenes."""
        return len(self.scenes)
    
    def get_all_entities(self) -> List[str]:
        """Get all unique entities mentioned in the script."""
        
        all_entities = set()
        for scene in self.scenes:
            all_entities.update(scene['entities'])
        
        return sorted(list(all_entities)) 