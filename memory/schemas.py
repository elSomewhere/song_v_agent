"""
Pydantic schemas for memory management in the RAG-based storyboard system.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class EntityState(BaseModel):
    """Represents the state of an entity in a scene."""
    name: str
    pose: str
    emotion: str
    position: str
    clothing_state: Optional[str] = None


class CameraSpec(BaseModel):
    """Camera specifications for a scene."""
    type: str = Field(description="Camera shot type (e.g., 'close-up', 'wide shot')")
    angle: str = Field(description="Camera angle (e.g., 'low angle', 'eye level')")
    distance: str = Field(description="Distance from subject")
    movement: Optional[str] = Field(None, description="Camera movement (e.g., 'pan', 'dolly')")


class EnvironmentSpec(BaseModel):
    """Environment specifications for a scene."""
    location: str
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    lighting: str


class ScenePlan(BaseModel):
    """Complete plan for generating a scene."""
    scene_id: int
    shot_id: int
    entities: List[EntityState]
    camera: CameraSpec
    environment: EnvironmentSpec
    action_beats: List[str] = Field(description="Key action moments to capture")
    visual_focus: str = Field(description="Primary visual element to emphasize")
    mood: str
    continuity_notes: Optional[str] = None
    image_prompt: str = Field(description="Optimized prompt for image generation")


class EpisodicMemory(BaseModel):
    """Stored memory of a successfully generated scene."""
    scene_id: int
    shot_id: int
    timestamp: datetime
    entities: List[str] = Field(description="List of entity names in the scene")
    entity_states: Dict[str, Dict[str, str]] = Field(description="Detailed state of each entity")
    camera: Dict[str, str]
    environment: Dict[str, str]
    visual_elements: List[str] = Field(description="Key visual elements in the scene")
    embedding: List[float] = Field(description="Vector embedding of the scene")
    image_paths: List[str] = Field(description="Paths to generated images")
    quality_score: float = Field(ge=0.0, le=1.0, description="Quality score from Vision QA")


class GenerationResult(BaseModel):
    """Result from the Renderer agent."""
    status: str = Field(pattern="^(success|error)$")
    image_data: Optional[str] = Field(None, description="Base64 encoded image or URL")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class VisionQAResult(BaseModel):
    """Result from the Vision QA agent."""
    status: str = Field(pattern="^(pass|retry|fail)$")
    quality_score: float = Field(ge=0.0, le=1.0)
    feedback: Dict[str, str] = Field(
        description="Detailed feedback on composition, accuracy, continuity, quality"
    )
    specific_issues: List[str] = Field(default_factory=list)
    retry_guidance: Optional[str] = None 