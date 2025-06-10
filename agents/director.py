"""
Director Agent - Coordinates the storyboard generation process.
Reads shot IDs sequentially, fetches shot text, detects entities, and manages workflow.
"""
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from agents.shared import get_shot_by_id, extract_entities_from_document

def get_available_entities() -> tuple[List[str], Dict[str, List[str]]]:
    """Get list of available entities from the data files.
    
    Returns:
        Tuple of (list of entities in refs, entity aliases dictionary)
    """
    # Get entities from refs folder
    refs_entities = []
    refs_path = Path("data/refs")
    
    if refs_path.exists():
        for subdir in refs_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                # Use directory name as entity name
                refs_entities.append(subdir.name)
    
    # Get entity aliases from entities document
    entity_aliases = extract_entities_from_document()
    
    return refs_entities, entity_aliases

def extract_entities(shot_text: str, base_context: Dict[str, Any] = None) -> List[str]:
    """Extract entity names from shot text using available entities.
    
    Args:
        shot_text: The shot description text
        base_context: Base context containing documents
        
    Returns:
        List of detected entity names
    """
    detected_entities = []
    
    # Get available entities and aliases
    if base_context and 'base_documents' in base_context:
        base_docs = base_context['base_documents']
        # Use cached entity aliases if available
        if hasattr(extract_entities, '_cached_aliases'):
            entity_aliases = extract_entities._cached_aliases
        else:
            entity_aliases = extract_entities_from_document()
            extract_entities._cached_aliases = entity_aliases
    else:
        entity_aliases = extract_entities_from_document()
    
    # Get refs entities
    refs_entities = []
    refs_path = Path("data/refs")
    
    if refs_path.exists():
        for subdir in refs_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                # Use directory name as entity name
                refs_entities.append(subdir.name)
    
    # Convert shot text to lowercase for comparison
    shot_text_lower = shot_text.lower()
    
    # Check for each entity from the entities document
    for canonical_name, aliases in entity_aliases.items():
        # Check if any alias appears in the text
        for alias in aliases:
            if alias.lower() in shot_text_lower:
                # Try to match with refs entities (case-insensitive)
                matched = False
                for ref_entity in refs_entities:
                    if ref_entity.lower() == canonical_name.lower():
                        detected_entities.append(ref_entity)
                        matched = True
                        break
                    # Also check if the ref entity matches any alias
                    elif ref_entity.lower() in [a.lower() for a in aliases]:
                        detected_entities.append(ref_entity)
                        matched = True
                        break
                
                # If no match in refs, use the canonical name if it exists in refs
                if not matched:
                    for ref_entity in refs_entities:
                        if canonical_name.lower() in ref_entity.lower() or ref_entity.lower() in canonical_name.lower():
                            detected_entities.append(ref_entity)
                            matched = True
                            break
                
                if matched:
                    break  # Move to next canonical entity
    
    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in detected_entities:
        if entity not in seen:
            seen.add(entity)
            unique_entities.append(entity)
    
    return unique_entities

async def director_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Director node for the LangGraph workflow.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with shot info
    """
    # Get current shot ID or start from 1
    current_shot_id = state.get("current_shot_id", 1)
    output_base_path = state.get("output_base_path", "output")
    variations_per_shot = state.get("variations_per_shot", 1)
    current_variation = state.get("current_variation", 1)
    
    # Check if we've already processed all variations for this shot
    shot_output_dir = Path(output_base_path) / f"Shot-{current_shot_id:03d}"
    if shot_output_dir.exists():
        # Count existing variations
        existing_variations = len(list(shot_output_dir.glob("var*.png")))
        if existing_variations >= variations_per_shot:
            print(f"Shot {current_shot_id} already has {existing_variations} variations, skipping...")
            return {
                **state,
                "current_shot_id": current_shot_id + 1,
                "current_variation": 1,
                "status": "skip"
            }
    
    # Fetch shot from database
    shot = get_shot_by_id(current_shot_id)
    
    if not shot:
        # No more shots to process
        return {
            **state,
            "status": "complete"
        }
    
    # Extract entities from shot text
    base_context = state.get('base_context', {})
    entities = extract_entities(shot['text'], base_context)
    
    print(f"Processing Shot {current_shot_id}")
    print(f"  Entities detected: {entities}")
    
    # Log available entities for debugging
    refs_entities, entity_aliases = get_available_entities()
    if not entities and refs_entities:
        print(f"  Available entities in refs: {refs_entities}")
        print(f"  Entity aliases from document: {list(entity_aliases.keys())}")
    
    # Initialize state for this shot
    result = {
        **state,
        "current_shot_id": current_shot_id,
        "shot_id": shot['shot_id'],
        "shot_text": shot['text'],
        "entities": entities,
        "attempt": 0,
        "current_variation": current_variation,  # Keep current variation
        "status": "ready",
        "output_base_path": output_base_path,
        "image_paths": state.get("image_paths", [])  # Preserve existing image paths
    }
    
    print(f"  Director returning state with keys: {list(result.keys())}")
    print(f"  Director shot_id: {result.get('shot_id')}")
    
    return result

# For testing
if __name__ == "__main__":
    # Test entity extraction
    test_text = """
    Medium shot of Helena walking calmly amid rubble and panicked townspeople. 
    Her white armor is still largely intact, but dirt smudges appear on the plating.
    Silicate tanks roll into town, crushing debris.
    """
    
    entities = extract_entities(test_text)
    print(f"Test entities: {entities}")
    
    # Show available entities
    refs_entities, entity_aliases = get_available_entities()
    print(f"\nAvailable entities in refs: {refs_entities}")
    print(f"\nEntity aliases from document:")
    for canonical, aliases in entity_aliases.items():
        print(f"  {canonical}: {aliases}") 