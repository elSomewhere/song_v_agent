"""
Structured schemas for the AI Storyboard Generator
Defines all data models for agent communication
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class EntitySpec(BaseModel):
    """Specification for an entity in a scene"""
    name: str
    canonical_id: str
    pose: str
    emotion: str
    visual_ref_ids: List[str] = Field(default_factory=list)


class CameraSpec(BaseModel):
    """Camera specification for a scene"""
    type: str
    angle: str
    movement: Optional[str] = None
    fov: Optional[int] = None


class EnvironmentSpec(BaseModel):
    """Environment specification for a scene"""
    setting: str
    lighting: str
    mood: str


class ContextRef(BaseModel):
    """Reference to retrieved context"""
    scene_id: int
    relevance: float
    summary: Optional[str] = None


class ScenePlan(BaseModel):
    """Complete scene plan output from Planner agent"""
    scene_id: int
    entities: List[EntitySpec]
    camera: CameraSpec
    environment: EnvironmentSpec
    visual_beats: List[str]
    retrieved_context: List[ContextRef]
    raw_description: str
    
    class Config:
        schema_extra = {
            "example": {
                "scene_id": 42,
                "entities": [
                    {
                        "name": "Helena",
                        "canonical_id": "CHAR_HELENA",
                        "pose": "crouching behind debris",
                        "emotion": "determined, alert",
                        "visual_ref_ids": ["helena_ref_01", "helena_ref_03"]
                    }
                ],
                "camera": {
                    "type": "medium shot",
                    "angle": "eye level",
                    "movement": "slow push in"
                },
                "environment": {
                    "setting": "destroyed urban street",
                    "lighting": "harsh shadows, fire glow",
                    "mood": "tense, apocalyptic"
                },
                "visual_beats": [
                    "explosion illuminates background",
                    "debris particles in air"
                ],
                "retrieved_context": [
                    {"scene_id": 38, "relevance": 0.89},
                    {"scene_id": 15, "relevance": 0.76}
                ],
                "raw_description": "Helena crouches behind debris..."
            }
        }


class VisionQAResult(BaseModel):
    """Result from Vision QA agent analysis"""
    status: str  # "pass" or "fail"
    scores: Dict[str, float] = Field(default_factory=lambda: {
        "composition_match": 0.0,
        "style_adherence": 0.0,
        "entity_accuracy": 0.0,
        "narrative_coherence": 0.0
    })
    issues: List[Dict[str, str]] = Field(default_factory=list)
    retry_guidance: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        """Check if the QA passed"""
        return self.status == "pass"
    
    @property
    def overall_score(self) -> float:
        """Calculate overall score from individual scores"""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


class GeneratedImage(BaseModel):
    """Information about a generated image"""
    path: str
    prompt: str
    scene_id: int
    variation_id: int
    retry_count: int = 0
    generation_params: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    qa_result: Optional[VisionQAResult] = None


class AgentState(BaseModel):
    """Shared state between agents in the workflow"""
    # Current processing info
    current_scene_id: int
    current_shot_description: str
    
    # Scene plan from Planner
    scene_plan: Optional[ScenePlan] = None
    
    # Generated images
    generated_images: List[GeneratedImage] = Field(default_factory=list)
    
    # Retry tracking
    current_retry_count: int = 0
    max_retries: int = 2
    
    # Variation tracking
    current_variation: int = 0
    total_variations: int = 3
    
    # Error tracking
    errors: List[str] = Field(default_factory=list)
    
    # Workflow metadata
    run_id: str
    output_dir: str
    
    def add_error(self, error: str):
        """Add an error to the state"""
        self.errors.append(f"[Scene {self.current_scene_id}] {error}")
    
    def should_retry(self) -> bool:
        """Check if we should retry the current generation"""
        return self.current_retry_count < self.max_retries
    
    def next_variation(self) -> bool:
        """Move to next variation, return True if more variations exist"""
        self.current_variation += 1
        self.current_retry_count = 0  # Reset retry count for new variation
        return self.current_variation < self.total_variations
    
    def reset_for_scene(self, scene_id: int, shot_description: str):
        """Reset state for a new scene"""
        self.current_scene_id = scene_id
        self.current_shot_description = shot_description
        self.scene_plan = None
        self.generated_images = []
        self.current_retry_count = 0
        self.current_variation = 0


class RAGConfig(BaseModel):
    """Configuration for the RAG system"""
    # Memory settings
    memory_db_path: str = "./memory_db"
    
    # Retrieval settings
    retrieval_k: int = 5
    semantic_weight: float = 0.6
    entity_weight: float = 0.3
    distance_weight: float = 0.1
    min_similarity: float = 0.5
    
    # Entity importance
    entity_boost: Dict[str, float] = Field(default_factory=lambda: {
        "protagonist": 1.5,
        "antagonist": 1.3,
        "supporting": 1.0
    })
    
    # Generation settings
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4000
    
    # Image settings
    image_size: str = "1792x1024"
    image_quality: str = "hd"
    image_style: str = "natural"
    
    # Processing settings
    variations_per_shot: int = 3
    max_retries_per_variation: int = 2
    parallel_variations: bool = False
    
    # Quality thresholds
    min_overall_qa_score: float = 0.7
    min_individual_qa_score: float = 0.6 