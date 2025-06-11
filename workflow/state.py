"""
Workflow state management for the RAG-based storyboard system.
"""
from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
import operator


class WorkflowState(TypedDict):
    """State passed between workflow nodes."""
    # Current processing state
    current_scene_number: int
    current_shot_id: int
    current_variation: int
    max_variations: int
    
    # Scene data
    scene_text: str
    scene_plan: Optional[Dict[str, Any]]
    
    # Memory context
    working_memory: List[Dict[str, Any]]
    canonical_memory: Dict[str, Any]
    
    # Generation state
    status: str  # planning|rendering|evaluating|complete|error|no_more_scenes
    attempt_number: int
    max_attempts: int
    
    # Results
    generated_images: Annotated[List[str], operator.add]
    quality_scores: Annotated[List[float], operator.add]
    feedback_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # Configuration
    output_base_path: str
    embedding_cache: Dict[str, List[float]]
    
    # Additional fields for compatibility
    error: Optional[str]
    memory_k: int
    memory_weights: Dict[str, float]
    quality_threshold: float 