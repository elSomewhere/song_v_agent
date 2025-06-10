"""
Migration script to transition from old sliding window system to new semantic RAG system
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
import sys
sys.path.append('..')

from memory import MemoryService, EntityState, CameraSpec, EnvironmentSpec, ImageRef


def extract_shot_info_from_prompt(prompt_file: Path) -> Dict[str, Any]:
    """Extract scene information from saved prompt JSON files"""
    with open(prompt_file, 'r') as f:
        data = json.load(f)
    
    # Extract entities from prompt text
    prompt_text = data.get('prompt', '')
    entities = []
    
    # Look for character mentions
    if 'Helena' in prompt_text:
        entities.append('Helena')
    if 'Joy' in prompt_text:
        entities.append('Joy')
    # Add more entity detection as needed
    
    # Extract visual elements
    visual_elements = []
    if 'explosion' in prompt_text.lower():
        visual_elements.append('explosion')
    if 'debris' in prompt_text.lower():
        visual_elements.append('debris')
    
    return {
        'entities': entities,
        'visual_elements': visual_elements,
        'prompt': prompt_text
    }


def extract_critic_feedback(critic_file: Path) -> List[str]:
    """Extract critic tags from feedback files"""
    try:
        with open(critic_file, 'r') as f:
            data = json.load(f)
        
        tags = []
        if data.get('pass') == False:
            tags.append('failed_qa')
        
        # Extract specific issues
        feedback = data.get('feedback', '')
        if 'aspect ratio' in feedback.lower():
            tags.append('aspect_ratio_issue')
        if 'style' in feedback.lower():
            tags.append('style_issue')
        
        return tags
    except:
        return []


def migrate_run_directory(run_dir: Path, memory_service: MemoryService):
    """Migrate a single run directory to episodic memory"""
    print(f"\nMigrating {run_dir.name}...")
    
    # Find all shot directories
    shot_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("Shot-")])
    
    for shot_dir in shot_dirs:
        # Extract shot ID
        shot_id_match = re.match(r'Shot-(\d+)', shot_dir.name)
        if not shot_id_match:
            continue
        
        shot_id = int(shot_id_match.group(1))
        
        # Skip if already exists
        if memory_service.episodic.get_by_scene_id(shot_id):
            print(f"  Shot {shot_id} already in memory, skipping...")
            continue
        
        print(f"  Processing Shot {shot_id}...")
        
        # Find prompt and image files
        prompt_files = list(shot_dir.glob("prompt_var*.json"))
        image_files = list(shot_dir.glob("var*.png"))
        critic_files = list(shot_dir.glob("critic_var*_try*.json"))
        
        if not prompt_files or not image_files:
            print(f"    Missing files for Shot {shot_id}, skipping...")
            continue
        
        # Extract information from first prompt
        shot_info = extract_shot_info_from_prompt(prompt_files[0])
        
        # Collect all critic tags
        all_critic_tags = []
        for critic_file in critic_files:
            tags = extract_critic_feedback(critic_file)
            all_critic_tags.extend(tags)
        
        # Create image references
        image_refs = []
        for i, img_file in enumerate(image_files):
            image_refs.append(ImageRef(
                path=str(img_file),
                variation_id=i,
                retry_count=0,
                quality_score=0.8  # Default score
            ))
        
        # Create default entity states
        entity_states = []
        for entity_name in shot_info['entities']:
            entity_states.append(EntityState(
                name=entity_name,
                pose="as shown",
                emotion="as depicted",
                visual_ref_ids=[]
            ))
        
        # Create default camera and environment specs
        camera_spec = CameraSpec(
            type="medium shot",
            angle="eye level"
        )
        
        environment_spec = EnvironmentSpec(
            setting="as depicted",
            lighting="cinematic",
            mood="dramatic"
        )
        
        # Write episodic memory
        try:
            memory_service.write_episodic_memory(
                scene_id=shot_id,
                shot_description=shot_info['prompt'][:200] + "...",
                entities=shot_info['entities'],
                entity_states=entity_states,
                camera_spec=camera_spec,
                environment_spec=environment_spec,
                visual_elements=shot_info['visual_elements'],
                generated_images=image_refs,
                critic_tags=list(set(all_critic_tags))
            )
            print(f"    Successfully migrated Shot {shot_id}")
        except Exception as e:
            print(f"    Error migrating Shot {shot_id}: {e}")


def main():
    """Main migration function"""
    print("=== AI Storyboard Generator Migration Tool ===")
    print("Migrating from sliding window to semantic RAG architecture\n")
    
    # Initialize memory service
    memory_service = MemoryService("./memory_db")
    
    # Initialize canonical memories if needed
    if not Path("./memory_db").exists():
        print("Initializing canonical memories...")
        memory_service.initialize_canonical_memories("./data")
    
    # Find all run directories
    output_path = Path("./output")
    if not output_path.exists():
        print("No output directory found. Nothing to migrate.")
        return
    
    run_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    if not run_dirs:
        print("No run directories found. Nothing to migrate.")
        return
    
    print(f"Found {len(run_dirs)} run directories to migrate")
    
    # Migrate each run
    for run_dir in run_dirs:
        migrate_run_directory(run_dir, memory_service)
    
    # Summary
    episodic_count = len(memory_service.episodic.get_all())
    print(f"\n=== Migration Complete ===")
    print(f"Total episodic memories created: {episodic_count}")
    print("\nYou can now use run_v2.py with the new semantic RAG architecture!")
    print("The system will automatically retrieve relevant memories based on:")
    print("  - Semantic similarity of scenes")
    print("  - Entity overlap between scenes")
    print("  - Narrative distance (with decay)")
    
    print("\nExample usage:")
    print("  python run_v2.py --shot-id 1 --max-shots 5 --retrieval-k 5")


if __name__ == "__main__":
    main() 