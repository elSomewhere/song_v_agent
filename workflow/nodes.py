"""
Workflow node implementations for the RAG-based storyboard system.
"""
from typing import Dict, Any
import asyncio

from workflow.state import WorkflowState
from agents.planner import PlannerAgent
from agents.renderer import RendererAgent
from agents.vision_qa import VisionQAAgent
from memory.service import MemoryService
from utils.parsing import ScriptParser


# Initialize global instances
planner_agent = PlannerAgent()
renderer_agent = RendererAgent()
vision_qa_agent = VisionQAAgent()
memory_service = None  # Will be initialized in the main workflow
script_parser = ScriptParser()


async def planner_node(state: WorkflowState) -> Dict[str, Any]:
    """Planning node - analyzes scene and creates generation plan."""
    
    # Get the current scene
    scene = script_parser.get_scene(state["current_shot_id"])
    
    if not scene:
        # No more scenes to process
        return {
            "status": "no_more_scenes",
            "error": f"No scene found for shot {state['current_shot_id']}"
        }
    
    # Update scene text and entities in state
    state["scene_text"] = scene["text"]
    
    # Retrieve working memory
    scene_context = {
        "shot_id": scene["shot_id"],
        "scene_text": scene["text"],
        "entities": scene["entities"]
    }
    
    # Calculate dynamic weights
    weights = memory_service.calculate_retrieval_weights(scene_context)
    
    # Retrieve relevant memories
    working_memory = await memory_service.retrieve_working_memory(
        scene_context,
        k=state["memory_k"],
        weights=weights
    )
    
    # Create scene plan
    scene_plan = await planner_agent.plan_scene(
        scene_text=scene["text"],
        scene_number=scene["shot_id"],
        working_memory=working_memory,
        canonical_style=state["canonical_memory"]["style"],
        canonical_entities=state["canonical_memory"]["entities"]
    )
    
    # Update state
    return {
        "scene_plan": scene_plan.model_dump(),
        "working_memory": working_memory,
        "status": "rendering",
        "current_variation": 1,
        "attempt_number": 1,
        "generated_images": [],
        "quality_scores": [],
        "feedback_history": []
    }


async def renderer_node(state: WorkflowState) -> Dict[str, Any]:
    """Rendering node - generates images based on scene plan."""
    
    scene_plan = state["scene_plan"]
    
    # Generate image
    result = await renderer_agent.render_image(
        scene_plan=scene_plan,  # Pass dict directly, will be converted internally
        variation_number=state["current_variation"],
        previous_attempts=state["feedback_history"],
        canonical_appearances=state["canonical_memory"]["entities"],
        output_path=state["output_base_path"]
    )
    
    if result.status == "success":
        # Add to generated images - return only the new item
        return {
            "status": "evaluating",
            "error": None,
            "generated_images": [result.image_data]  # operator.add will append this
        }
    else:
        # Handle error
        return {
            "status": "error",
            "error": result.error
        }


async def vision_qa_node(state: WorkflowState) -> Dict[str, Any]:
    """Vision QA node - evaluates generated images."""
    
    # Check if we have any generated images
    if not state.get("generated_images") or len(state["generated_images"]) == 0:
        return {
            "status": "error",
            "error": "No images available for evaluation"
        }
    
    # Get the latest generated image
    latest_image = state["generated_images"][-1]
    scene_plan = state["scene_plan"]
    
    # Evaluate the image
    evaluation = await vision_qa_agent.evaluate_image(
        scene_plan=scene_plan,  # Pass dict directly
        image_path=latest_image,
        attempt_number=state["attempt_number"],
        previous_feedback=state["feedback_history"],
        quality_threshold=state["quality_threshold"]
    )
    
    # Add to feedback history - return only new items for operator.add
    new_feedback = evaluation.model_dump()
    new_score = evaluation.quality_score
    
    # Determine next action based on evaluation
    if evaluation.status == "pass":
        # Check if we need more variations
        if state["current_variation"] < state["max_variations"]:
            return {
                "status": "rendering",
                "current_variation": state["current_variation"] + 1,
                "attempt_number": 1,  # Reset attempts for new variation
                "feedback_history": [new_feedback],
                "quality_scores": [new_score]
            }
        else:
            # All variations complete
            return {
                "status": "complete",
                "feedback_history": [new_feedback],
                "quality_scores": [new_score]
            }
    
    elif evaluation.status == "retry":
        # Retry current variation
        return {
            "status": "rendering",
            "attempt_number": state["attempt_number"] + 1,
            "feedback_history": [new_feedback],
            "quality_scores": [new_score]
        }
    
    else:  # fail
        # Check if we should try next variation or give up
        if state["current_variation"] < state["max_variations"]:
            return {
                "status": "rendering",
                "current_variation": state["current_variation"] + 1,
                "attempt_number": 1,
                "feedback_history": [new_feedback],
                "quality_scores": [new_score]
            }
        else:
            return {
                "status": "complete",  # Complete with failures
                "feedback_history": [new_feedback],
                "quality_scores": [new_score]
            }


async def memory_update_node(state: WorkflowState) -> Dict[str, Any]:
    """Memory update node - stores successful generations in episodic memory."""
    
    # Only store if we have successful images
    if state["generated_images"] and any(score >= state["quality_threshold"] for score in state["quality_scores"]):
        
        # Prepare scene data for storage
        scene_data = {
            "scene_id": state["current_scene_number"],
            "shot_id": state["current_shot_id"],
            "entities": [e["name"] if isinstance(e, dict) else e for e in state["scene_plan"].get("entities", [])],
            "entity_states": {
                e["name"]: {
                    "pose": e.get("pose", ""),
                    "emotion": e.get("emotion", ""),
                    "position": e.get("position", "")
                }
                for e in state["scene_plan"].get("entities", [])
                if isinstance(e, dict) and "name" in e
            },
            "camera": state["scene_plan"].get("camera", {}),
            "environment": state["scene_plan"].get("environment", {}),
            "visual_elements": state["scene_plan"].get("action_beats", []),
            "mood": state["scene_plan"].get("mood", ""),
            "visual_focus": state["scene_plan"].get("visual_focus", ""),
            "image_paths": state["generated_images"],
            "quality_score": max(state["quality_scores"]) if state["quality_scores"] else 0.0
        }
        
        # Store in episodic memory
        await memory_service.store_episodic_memory(scene_data)
    
    # Mark as complete
    return {
        "status": "scene_complete"
    }


def initialize_memory_service(db_path: str = "./lancedb_data", data_path: str = "data"):
    """Initialize the global memory service."""
    global memory_service
    memory_service = MemoryService(db_path, data_path) 