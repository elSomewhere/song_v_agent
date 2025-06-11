"""
Simple test script to verify the RAG-based storyboard system components.
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_memory_service():
    """Test the memory service initialization and basic operations."""
    print("Testing Memory Service...")
    
    from memory.service import MemoryService
    
    # Initialize memory service
    memory_service = MemoryService()
    
    # Check canonical memory
    print(f"Canonical entities loaded: {len(memory_service.canonical_memory.get_all_entities())}")
    print(f"Entities: {memory_service.canonical_memory.get_all_entities()[:5]}...")
    
    # Test scene context
    test_scene = {
        "shot_id": 1,
        "scene_text": "Helena stands in the rain, looking determined",
        "entities": ["Helena"]
    }
    
    # Test retrieval (should return empty on first run)
    memories = await memory_service.retrieve_working_memory(test_scene, k=3)
    print(f"Retrieved {len(memories)} memories")
    
    print("✓ Memory Service test passed\n")


async def test_script_parser():
    """Test the script parser."""
    print("Testing Script Parser...")
    
    from utils.parsing import ScriptParser
    
    parser = ScriptParser()
    print(f"Total scenes found: {parser.get_scene_count()}")
    
    # Test getting first scene
    scene = parser.get_scene(1)
    if scene:
        print(f"First scene: Shot {scene['shot_id']} - {scene['title'][:50]}...")
        print(f"Entities detected: {scene['entities']}")
    
    print("✓ Script Parser test passed\n")


async def test_planner_agent():
    """Test the planner agent."""
    print("Testing Planner Agent...")
    
    from agents.planner import PlannerAgent
    from memory.service import MemoryService
    
    # Initialize
    planner = PlannerAgent()
    memory_service = MemoryService()
    
    # Get test scene
    from utils.parsing import ScriptParser
    parser = ScriptParser()
    scene = parser.get_scene(1)
    
    if scene:
        # Plan the scene
        plan = await planner.plan_scene(
            scene_text=scene['text'],
            scene_number=scene['shot_id'],
            working_memory=[],
            canonical_style=memory_service.canonical_memory.style_content,
            canonical_entities=memory_service.canonical_memory.entity_appearances
        )
        
        print(f"Scene plan created:")
        print(f"  - Entities: {len(plan.entities)}")
        print(f"  - Camera: {plan.camera.type}")
        print(f"  - Mood: {plan.mood}")
        print(f"  - Prompt length: {len(plan.image_prompt)} chars")
    
    print("✓ Planner Agent test passed\n")


async def test_workflow():
    """Test the workflow initialization."""
    print("Testing Workflow...")
    
    from workflow.graph import create_workflow
    
    config = {
        "variations_per_shot": 1,
        "max_retries": 1,
        "memory_k": 5,
        "memory_weights": {"semantic": 0.6, "entity": 0.3, "temporal": 0.1},
        "quality_threshold": 0.7,
        "cache_embeddings": True,
        "save_metadata": False,
        "db_path": "./lancedb_data",
        "data_path": "data"
    }
    
    workflow = create_workflow(config)
    print("✓ Workflow created successfully\n")


async def main():
    """Run all tests."""
    print("=== RAG Storyboard System Component Tests ===\n")
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    try:
        await test_memory_service()
        await test_script_parser()
        await test_planner_agent()
        await test_workflow()
        
        print("=== All tests passed! ===")
        print("\nYou can now run the full system with:")
        print("  python run_rag.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 